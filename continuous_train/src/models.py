from torch import nn
class MLP3Layer(nn.Module):
    def __init__(self, num_classes, input_dim=29, hidden_size=128, dropout_rate=0.2):
        super(MLP3Layer, self).__init__()
        self.layer1 = self._make_layer(input_dim, hidden_size, dropout_rate)
        self.layer2 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.layer3 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.output = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, in_dim, out_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2 + out1)
        out4 = self.output(out3 + out2)
        return out4

class MLP6Layer(nn.Module):
    def __init__(self, num_classes, input_dim=29, hidden_size=128, dropout_rate=0.2):
        super(MLP6Layer, self).__init__()
        self.layer1 = self._make_layer(input_dim, hidden_size, dropout_rate)
        self.layer2 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.layer3 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.layer4 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.layer5 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.layer6 = self._make_layer(hidden_size, hidden_size, dropout_rate)
        self.output = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, in_dim, out_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2 + out1)
        out4 = self.layer4(out3 + out2)
        out5 = self.layer5(out4 + out3)
        out6 = self.layer6(out5 + out4)
        out7 = self.output(out6 + out5)
        return out7