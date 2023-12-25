import torch.nn as nn
import torch.nn.functional as F

class NetCupRegressor(nn.Module):
    def __init__(self):
        super(NetCupRegressor, self).__init__()

        #Layer 1 Input: 10 Output: 128
        self.layer1 = nn.Linear(10, 128)
        #nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer1.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer1.bias, 0.01)
        #Layer 2 Input: 50 Output: 100
        self.layer2 = nn.Linear(128, 64)
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer2.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer2.bias, 0.01)
        #Layer 3 Input: 100 Output: 25
        self.layer3 = nn.Linear(64, 3)
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer3.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer3.bias, 0.01)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x
    
    
class NetCupCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(10, 16, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 5, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        return x