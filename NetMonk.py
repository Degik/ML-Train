import torch.nn as nn
import torch.nn.functional as F

class NetMonkClassifier(nn.Module):
    def __init__(self, hidden_size:int):
        super(NetMonkClassifier, self).__init__()

        #Layer 1 Input: 17 Output: hidden_size
        self.layer1 = nn.Linear(17, hidden_size)
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.layer1.bias)
        # Layer 2 Input: hidden_size Output: 2
        # Output: P(0) and P(1)
        self.layer2 = nn.Linear(hidden_size, 2)
        nn.init.xavier_uniform_(self.layer2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class NetMonkRegressor(nn.Module):
    def __init__(self, interval:float, hidden_size:int):
        super(NetMonkRegressor, self).__init__()

        #Layer 1 Input: 17 Output: hidden_size
        self.layer1 = nn.Linear(17, hidden_size)
        nn.init.uniform_(self.layer1.weight, -interval, interval)
        nn.init.constant_(self.layer1.bias, 0.01)
        #Layer 2 Input: hidden_size Output: 1
        self.layer2 = nn.Linear(hidden_size, 1)
        nn.init.uniform_(self.layer2.weight, -interval, interval)
        nn.init.constant_(self.layer2.bias, 0.01)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x