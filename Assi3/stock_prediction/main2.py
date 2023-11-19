import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# 加载数据
# df = pd.read_csv("data/000001.csv")

df = pd.read_excel("data/stock_data.xlsx")

df_main = df[['open', 'high', 'low', 'close']]

# 缺失值填充，使用上一个有效值
df_main = df_main.fillna(method='ffill')

plt.figure(figsize=(15, 6))
plt.plot(df_main['open'])
plt.title('Close Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig("show.png", dpi=300)


# Time Series Line Plot
plt.figure(figsize=(14, 7))
for c in df_main.columns.values:
    plt.plot(df_main.index, df_main[c], label=c)
plt.title('Stock Price History')
plt.xlabel('Dates')
plt.ylabel('Price')
plt.legend()
plt.savefig("sph.png", dpi=300)

# Moving Averages
moving_averages = [20, 50, 100]
plt.figure(figsize=(14, 7))
for ma in moving_averages:
    df_main['MA' + str(ma)] = df_main['close'].rolling(window=ma).mean()
plt.plot(df_main['close'], label='Actual Prices')
for ma in moving_averages:
    plt.plot(df_main['MA' + str(ma)], label=f'MA for {ma} days')
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig("ma.png", dpi=300)

# Histogram of Daily Price Changes
daily_changes = df_main['close'].pct_change().dropna()
plt.figure(figsize=(10, 7))
plt.hist(daily_changes, bins=50, alpha=0.75)
plt.title('Histogram of Daily Price Changes')
plt.xlabel('Percentage Change')
plt.savefig("dpc.png", dpi=300)


# 数据缩放
scaler = MinMaxScaler(feature_range=(-1, 1))
print(df_main.head())
# 这里不能进行统一进行缩放，因为fit_transform返回值是numpy类型
for col in ['open', 'high', 'low', 'close']:
    df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1,1))

# 将下一日的收盘价作为本日的标签
df_main['target'] = df_main['close'].shift(-1)

print(df_main.head())

# 使用了shift函数，在最后必然是有缺失值的，这里去掉缺失值所在行
df_main.dropna()

# 修改数据类型
df_main = df_main.astype(np.float32)

print(df_main)

input_dim = 4      # 数据的特征数
hidden_dim = 32    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out)

        return out


# 创建两个列表，用来存储数据的特征和标签
data_feat, data_target = [],[]

# 设每条数据序列有20组数据
seq = 20

for index in range(len(df_main) - seq):
    # 构建特征集
    data_feat.append(df_main[['open', 'high', 'low', 'close']][index: index + seq].values)
    # 构建target集
    data_target.append(df_main['target'][index:index + seq])

# 将特征集和标签集整理成numpy数组
data_feat = np.array(data_feat)
data_target = np.array(data_target)


# 这里按照8:2的比例划分训练集和测试集
test_set_size = int(np.round(0.2*df_main.shape[0]))  # np.round(1)是四舍五入，
train_size = data_feat.shape[0] - (test_set_size)
print(test_set_size)  # 输出测试集大小
print(train_size)     # 输出训练集大小


trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,4)).type(torch.Tensor)
# 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
testX = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,4)).type(torch.Tensor)
trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
testY = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor)

print('x_train.shape = ', trainX.shape)
print('y_train.shape = ', trainY.shape)
print('x_test.shape = ', testX.shape)
print('y_test.shape = ', testY.shape)


# 实例化模型
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法
loss_fn = torch.nn.MSELoss(size_average=True)             # 使用均方差作为损失函数

# 设定数据遍历次数
num_epochs = 100

# 打印模型结构
print(model)

# train model
hist = np.zeros(num_epochs)
for t in range(num_epochs):
    # Forward pass
    y_train_pred = model(trainX)

    loss = loss_fn(y_train_pred, trainY)
    if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs 将梯度归零
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()


plt.figure(figsize=(10, 4))
plt.plot(hist)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.savefig("loss.png", dpi=300)

# make predictions
y_test_pred = model(testX)
y_test_pred_denorm = scaler.inverse_transform(y_test_pred[:, -1, :].detach().numpy().reshape(-1, 1)).reshape(-1)
testY_denorm = scaler.inverse_transform(testY[:, -1, :].detach().numpy().reshape(-1, 1)).reshape(-1)

# Plotting the predictions against actual values
plt.figure(figsize=(15, 6))
plt.plot(y_test_pred_denorm, label='Predicted')
plt.plot(testY_denorm, label='Actual')
plt.title('Predicted vs Actual Closing Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig("1.png", dpi=300)
# plt.show()

# Evaluation metrics
test_mae = mean_absolute_error(testY_denorm, y_test_pred_denorm)
test_r2 = r2_score(testY_denorm, y_test_pred_denorm)
print(f"Mean Absolute Error: {test_mae}")
print(f"R2 Score: {test_r2}")