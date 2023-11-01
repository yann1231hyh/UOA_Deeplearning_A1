import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
matplotlib.use('Agg')

# Define a CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Adjusted kernel size and padding
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Define some hyperparameters
batch_size = 64
learning_rate = 0.02
num_epochs = 5

# Data preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the CNN model
model = ConvNet()
if torch.cuda.is_available():
    model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Visualize random samples
sample_loader = DataLoader(test_dataset, batch_size=9, shuffle=True)
sample_batch = next(iter(sample_loader))
sample_images, sample_labels = sample_batch

sample_outputs = model(sample_images.cuda() if torch.cuda.is_available() else sample_images)
sample_predictions = torch.argmax(sample_outputs, dim=1)

plt.figure(figsize=(10, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f'Predicted: {sample_predictions[i]}, Actual: {sample_labels[i]}')
    plt.imshow(sample_images[i][0], cmap='gray')
    plt.axis('off')

plt.savefig("img.png", dpi=300)

# plt.show()

# Parameter sensitivity analysis
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
sensitivity_results = []

for lr in learning_rates:
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for data in train_loader:
            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            outputs = model(img)
            loss = criterion(outputs, label)
            eval_loss += loss.item() * label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            eval_acc += (predicted == label).sum().item()

    accuracy = eval_acc / len(test_dataset)

    print(lr, accuracy)
    sensitivity_results.append(accuracy)

# Plot parameter sensitivity results
plt.figure()
plt.plot(learning_rates, sensitivity_results, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Parameter Sensitivity Analysis')
plt.savefig("Parameter.png", dpi=300)
# plt.show()

best_lr = learning_rates[sensitivity_results.index(max(sensitivity_results))]
final_model = ConvNet()
if torch.cuda.is_available():
    final_model = final_model.cuda()
optimizer = optim.SGD(final_model.parameters(), lr=best_lr)

train_epochs = 20
# Training the model
loss_history = []  # To store the loss values for plotting
for epoch in range(train_epochs):
    total_loss = 0.0
    for data in train_loader:
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        outputs = final_model(img)
        loss = criterion(outputs, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # total_loss /= len(train_dataset)
    loss_history.append(total_loss)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, train_epochs, total_loss))

# Plot the loss history
plt.figure()
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig("Loss.png", dpi=300)
# plt.show()

# Model evaluation
final_model.eval()
eval_loss = 0
eval_acc = 0
predictions = []
with torch.no_grad():
    for data in test_loader:
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        outputs = final_model(img)
        loss = criterion(outputs, label)
        eval_loss += loss.item() * label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc += (predicted == label).sum().item()
        predictions.extend(predicted.cpu().numpy())

    accuracy = eval_acc / len(test_dataset)
    print('Test Loss: {:.6f}, Accuracy: {:.6f}'.format(eval_loss / len(test_dataset), accuracy))