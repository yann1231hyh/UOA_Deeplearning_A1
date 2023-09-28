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

# Define data & reading function
def read_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove newlines and spaces at the end of lines
            if line:
                parts = line.split(' ')  # Use spaces to separate sections in a line
                label = int(parts[0])  # The first part is the tag value
                features = {}
                for feature in parts[1:]:
                    index, value = feature.split(':')  # Use a colon to separate the index and value of a feature
                    index = int(index)
                    value = float(value)
                    features[index] = value
                data.append((label, features))

    # Handle missing values, set to zero
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
    # Set the value labeled -1 to 0
    y[y==-1] = 0
    print(X.shape, y.shape)
    return X, y
         

# Define the perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
# Define multilayer perceptron model
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
        # Create a perceptron model instance
        model = Perceptron(args.input_size)
    elif args.model == 'MLP':
        # 创Build an MLP model example
        model = MLP(args.input_size, args.hidden_size, args.output_size, args.num_layers)
    else:
        print('no such model')
        exit(-1)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary classification cross entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  # optimizer
    
    # Set learning rate scheduler
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=1000, verbose=True)
    scheduler = ExponentialLR(optimizer, gamma=1)
    
    # Divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data and labels to PyTorch Tensor
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).unsqueeze(1).float()
    y_test = torch.from_numpy(y_test).unsqueeze(1).float()


    # Record loss and accuracy during training
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # Track top accuracy and save models
    best_acc = 0.0
    best_model_path = 'best_model.pt'

    # Start training cycle
    for epoch in tqdm(range(args.num_epochs)):
        for batch_start in range(0, X_train.size(0), args.batch_size):
            # Get the data and labels of the current batch
            batch_inputs = X_train[batch_start:batch_start + args.batch_size]
            batch_targets = y_train[batch_start:batch_start + args.batch_size]

            # forward propagation
            outputs = model(batch_inputs)

            # Calculate losses
            loss = criterion(outputs, batch_targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Adjust learning rate
        scheduler.step()
        
        # Calculate the loss and accuracy on the training set and test set after each epoch.
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

            # If the current test accuracy is the highest historical accuracy, save the model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), best_model_path)
        # Print the loss and accuracy of each epoch
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss.item()}, Train Acc: {train_acc}')
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {test_loss.item()}, Test Acc: {test_acc}')

    print(f'Best Test Acc: {best_acc}')
    
    # Save model
    torch.save(model.state_dict(), 'final_model.pt')
    
    # Draw loss curve
    plt.figure()
    plt.plot(range(1, args.num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Draw accuracy curve
    plt.figure()
    plt.plot(range(1, args.num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, args.num_epochs+1), test_accs, label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
