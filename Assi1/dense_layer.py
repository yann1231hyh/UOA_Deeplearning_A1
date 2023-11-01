import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

# 定义数据读取函数
def read_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空格
            if line:
                parts = line.split(' ')  # 使用空格分隔行中的部分
                label = int(parts[0])  # 第一个部分为标签值
                features = {}
                for feature in parts[1:]:
                    index, value = feature.split(':')  # 使用冒号分隔特征的索引和值
                    index = int(index)
                    value = float(value)
                    features[index] = value
                data.append((label, features))

    # 处理缺少的值，置零
    X, y = [], []
    for i, item in enumerate(data):
        label, features = item
        y.append(label)
        feat = np.zeros(8)
        for index in range(1, 9):
            if index not in features.keys():
                print('{} row has no {} col'.format(i+1, index))
                continue
            feat[index - 1] = np.array(features[index], dtype=np.float64)
        X.append(feat)
    X = np.vstack(X)
    y = np.array(y)
    # 将标签为-1的值设为0
    y[y==-1] = 0
    print(X.shape, y.shape)
    return X, y
         

# 定义感知机模型
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perceptron Training')
    parser.add_argument('--data', type=str, default='diabetes_scale.txt', help='input data')
    parser.add_argument('--model', type=str, default='MLP', choices=['Perceptron', 'MLP'], help='model')
    parser.add_argument('--input_size', type=int, default=8, help='input size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--output_size', type=int, default=1, help='output size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    args = parser.parse_args()
    
    print('Start read data')
    X, y = read_data(args.data)
    
    if args.model == 'Perceptron':
        # 创建感知机模型实例
        model = Perceptron(args.input_size)
    elif args.model == 'MLP':
        # 创建MLP模型实例
        model = MLP(args.input_size, args.hidden_size, args.output_size, args.num_layers)
    else:
        print('no such model')
        exit(-1)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  # 优化器
    
    # 设置学习率调度器
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=1000, verbose=True)
    scheduler = ExponentialLR(optimizer, gamma=1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换数据和标签为PyTorch的Tensor
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).unsqueeze(1).float()
    y_test = torch.from_numpy(y_test).unsqueeze(1).float()


    # 记录训练过程中的loss和准确率
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # 跟踪最高准确率和保存模型
    best_acc = 0.0
    best_model_path = 'best_model.pt'

    # 开始训练循环
    for epoch in tqdm(range(args.num_epochs)):
        for batch_start in range(0, X_train.size(0), args.batch_size):
            # 获取当前批次的数据和标签
            batch_inputs = X_train[batch_start:batch_start + args.batch_size]
            batch_targets = y_train[batch_start:batch_start + args.batch_size]

            # 前向传播
            outputs = model(batch_inputs)

            # 计算损失
            loss = criterion(outputs, batch_targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 调整学习率
        scheduler.step()
        
        # 在每个epoch结束后计算训练集和测试集上的loss和准确率
        with torch.no_grad():
            train_outputs = model(X_train)
            train_loss = criterion(train_outputs, y_train)
            train_pred = torch.round(train_outputs)
            train_acc = accuracy_score(y_train, train_pred)

            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_pred = torch.round(test_outputs)
            test_acc = accuracy_score(y_test, test_pred)

            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            # 如果当前测试准确率是历史最高准确率，则保存模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), best_model_path)
        # 打印每个epoch的loss和准确率
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss.item()}, Train Acc: {train_acc}')
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {test_loss.item()}, Test Acc: {test_acc}')

    print(f'Best Test Acc: {best_acc}')
    
    # 保存模型
    torch.save(model.state_dict(), 'final_model.pt')
    
    # 绘制loss曲线
    plt.figure()
    plt.plot(range(1, args.num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(range(1, args.num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, args.num_epochs+1), test_accs, label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
