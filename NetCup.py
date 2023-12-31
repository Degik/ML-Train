import torch.nn as nn
import torch.nn.functional as F

class NetCupRegressor(nn.Module):
    def __init__(self, hd_layer1, hd_layer2, hd_layer3, hd_layer4, hd_layer5, activation:str = "relu"):
        super(NetCupRegressor, self).__init__()
        self.activation = activation
        
        #Layer 1 Input: 10 Output: 128
        self.layer1 = nn.Linear(10, hd_layer1)
        #nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer1.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer1.bias, 0.01)
        #Layer 2 Input: 50 Output: 100
        self.layer2 = nn.Linear(hd_layer1, hd_layer2)
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer2.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer2.bias, 0.01)
        #Layer 3 Input: 100 Output: 25
        self.layer3 = nn.Linear(hd_layer2, hd_layer3)
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer2.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer2.bias, 0.01)
        #Layer 3 Input: 100 Output: 25
        self.layer4 = nn.Linear(hd_layer3, hd_layer4)
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.uniform_(self.layer3.weight, -0.7, 0.7)
        #nn.init.constant_(self.layer3.bias, 0.01)
        self.layer5 = nn.Linear(hd_layer4, hd_layer5)
        #
        self.layer6 = nn.Linear(hd_layer5, 3)

    def forward(self, x):
        if self.activation == "relu":
            activationFunction = F.relu
        elif self.activation == "tanh":
            activationFunction = F.tanh
        elif self.activation == "softmax":
            activationFunction = F.softmax
        else:
            print("NESUNNA FUNZIONE DI ATTIVAZIONE!")
            exit(1)
        x = self.layer1(x)
        x = activationFunction(x)
        x = self.layer2(x)
        x = activationFunction(x)
        x = self.layer3(x)
        x = activationFunction(x)
        x = self.layer4(x)
        x = activationFunction(x)
        x = self.layer5(x)
        x = activationFunction(x)
        x = self.layer6(x)
        return x