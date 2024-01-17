import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

from layer import MLPLayer, FeatureSelector

'''
Bottom Model
'''
# Passive Party
class MNIST_BottomModel(MLPLayer):
    def __init__(self, input_dim, nr_classes, hidden_dims, mu, batch_norm=None, dropout=None, activation='relu',sigma=0.01, lam=0.1,is_top=False):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation,is_top=is_top)
        self.FeatureSelector = FeatureSelector(sigma, mu)
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, x):
        regx = None
        x = self.FeatureSelector(x)
        logits = super().forward(x)
        if self.training:
            regx = self.lam * torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        return logits, regx

class EMNIST_BottomModel(nn.Module):
    def __init__(self, mu, sigma=1.0, lam=0.1):
        super(EMNIST_BottomModel, self).__init__()
        self.FeatureSelector = FeatureSelector(sigma, mu)
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        self.fc1 = nn.Linear(392, 50)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(50)

    def forward(self, x):
        regx = None
        x = self.FeatureSelector(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn1(x)
        if self.training:
            regx = self.lam * torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        return x, regx

class CCFD_BottomModel(MLPLayer):
    def __init__(self, input_dim, nr_classes, hidden_dims, mu, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1, is_top=False):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation, is_top=is_top)
        self.FeatureSelector = FeatureSelector(sigma, mu)
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, x):
        regx = None
        x = self.FeatureSelector(x)
        logits = super().forward(x)
        if self.training:
            regx = self.lam * torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        return logits, regx

# Active Party
class Active_EMNIST_BottomModel(nn.Module):
    # emnist
    def __init__(self):
        super(Active_EMNIST_BottomModel, self).__init__()
        self.fc1 = nn.Linear(392, 50)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn1(x)
        return x


class Active_CCFD_BottomModel(MLPLayer):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                  is_top=False):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation, is_top=is_top)

    def forward(self, x):
        logits = super().forward(x)
        return logits


class Active_MNIST_BottomModel(MLPLayer):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 is_top=False):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation, is_top=is_top)

    def forward(self, x):
        logits = super().forward(x)
        return logits


'''
Top Model
'''


class CCFD_TopModel(MLPLayer):
    def __init__(self, input_dim, output_dims, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 is_top=True):
        super().__init__(input_dim, output_dims, hidden_dims, batch_norm=batch_norm, dropout=dropout,
                         activation=activation, is_top=is_top)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        logits = super().forward(x)
        l = self.sft(logits)
        return l


class EMNIST_TopModel(nn.Module):
    # emnist
    def __init__(self, ):
        super(EMNIST_TopModel, self).__init__()
        self.lin1 = torch.nn.Linear(100, 100)
        self.relu1 = nn.ReLU()
        self.lin3 = torch.nn.Linear(100, 26)
        self.bn1 = nn.BatchNorm1d(100)
        self.sft = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        l = self.sft(x)
        return l


class Mnist_TopModel(nn.Module):
    # For MNIST and FashionMNIST
    def __init__(self, input_dim, output_dims, hidden_dims):
        super(Mnist_TopModel, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dims)
        self.lin2 = torch.nn.Linear(hidden_dims, output_dims)
        self.bn1 = nn.BatchNorm1d(hidden_dims)
        self.relu = torch.nn.ReLU()
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        l = self.sft(x)
        return l
