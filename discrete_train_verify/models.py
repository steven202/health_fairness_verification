""" a model file """
from torch import nn
import torch


def mlp_4_layer(num_classes):
    """a 4 layer mlp"""
    model = nn.Sequential(
        nn.Linear(29, 128),
        nn. BatchNorm1d(128),nn.ReLU(),
        nn.Linear(128, 256),
        nn. BatchNorm1d(256),nn.ReLU(),
        nn.Linear(256, 128),
        nn. BatchNorm1d(128),nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    return model


def mlp_3_layer(num_classes):
    """a 4 layer mlp"""
    model = nn.Sequential(
        nn.Linear(29, 32),
        nn. BatchNorm1d(32),nn.ReLU(),
        nn.Linear(32, 32),
        nn. BatchNorm1d(32),nn.ReLU(),
        nn.Linear(32, num_classes),
        # nn.Dropout(0.5),
    )
    return model


def mlp_6_layer(num_classes):
    """a 4 layer mlp"""
    model = nn.Sequential(
        nn.Linear(29, 64),
        nn. BatchNorm1d(64),nn.ReLU(),
        nn.Linear(64, 128),
        nn. BatchNorm1d(128),nn.ReLU(),
        nn.Linear(128, 256),
        nn. BatchNorm1d(256),nn.ReLU(),
        nn.Linear(256, 128),
        nn. BatchNorm1d(128),nn.ReLU(),
        nn.Linear(128, 64),
        nn. BatchNorm1d(64),nn.ReLU(),
        nn.Linear(64, num_classes),
        # nn.Dropout(0.5),
    )
    return model


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = torch.nn.Linear(29, output_dim)
        # torch.nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        # torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        z = self.fc(x)
        # p = torch.sigmoid(z)
        p = 1 / (1 + torch.exp(-(z)))
        return p.squeeze()

    def predict(self, X):
        Z = self.forward(X)
        Y = torch.where(Z > 0.5, 1, 0)
        return Y.to(torch.long)


class RNNModel_1_layer(nn.Module):
    """a one layer rnn model"""

    def __init__(self, input_dim=32, hidden_dim=512, layer_dim=1, output_dim=3, dropout_prob=0.1):
        super(RNNModel_1_layer, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # RNN layers
        self.rnn = nn.RNNCell(input_dim, hidden_dim, nonlinearity="relu")
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.length = -1

    def set_length(self, length):
        """set length each time before forward"""
        self.length = length

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(x.shape[0], self.hidden_dim).requires_grad_().cuda()

        # Forward propagation by passing in the input and hidden state into the model
        for _ in range(self.length):
            h0 = self.rnn.cuda()(x.cuda(), h0.detach().cuda())

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = nn.functional.softmax(self.fc.cuda()(h0.cuda()).cuda())
        return out


class RNNModel_2_layer(nn.Module):
    """a two layer rnn model"""

    def __init__(self, input_dim=32, hidden_dim=512, layer_dim=1, output_dim=3, dropout_prob=0.1):
        super(RNNModel_1_layer, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # RNN layers
        self.rnn1 = nn.RNNCell(input_dim, hidden_dim, nonlinearity="relu")
        self.rnn2 = nn.RNNCell(hidden_dim, hidden_dim, nonlinearity="relu")
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.length = -1

    def set_length(self, length):
        """set length each time before forward"""
        self.length = length

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(x.shape[0], self.hidden_dim).requires_grad_().cuda()

        # Forward propagation by passing in the input and hidden state into the model
        for _ in range(self.length):
            h0 = self.rnn1.cuda()(x.cuda(), h0.cuda())
            h0 = self.rnn1.cuda()(h0.cuda(), h0.cuda())
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = nn.functional.softmax(self.fc.cuda()(h0.cuda()).cuda())
        return out
