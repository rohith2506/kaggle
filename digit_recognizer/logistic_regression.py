'''
Solve MNIST problem using logistic regression
@Author: Rohith Uppala
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pdb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


'''
A simple logistic regression model
'''
class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

'''
Train the model
'''
def train_model():
    train = pd.read_csv("input/train.csv", dtype=np.float32)
    targets_numpy = train.label.values
    features_numpy = train.loc[:, train.columns != "label"].values / 255
    features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)

    # Change it to torch tensors
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

    batch_size, n_iter = 100, 10000
    num_epochs = n_iter / (len(features_train) / batch_size)
    num_epochs = int(num_epochs)

    # Create a tensor dataset
    train = TensorDataset(featuresTrain, targetsTrain)
    test = TensorDataset(featuresTest, targetsTest)

    # Create a data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    input_dim, output_dim = 28*28, 10
    model = LogisticRegressionModule(input_dim, output_dim)
    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            train = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(train)
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()
            count = count + 1
            if count % 500 == 0:
                correct, total = 0, 0
                for images, labels in test_loader:
                    test = Variable(images.view(-1, 28*28))
                    outputs = model(test)
                    predicted = torch.max(outputs.data, 1)[1]
                    total += len(labels)
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / float(total)
                print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))

train_model()
