import torch.nn as nn
import torch.nn.functional as F
    
class NetCupRegressor(nn.Module):
    def __init__(self, hd_layers:list, structure:list, activation:str = "relu",):
        super(NetCupRegressor, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(hd_layers) - 1):
            self.layers.append(nn.Linear(hd_layers[i], hd_layers[i+1]))
            res = f'Layer[{i+1}]: {hd_layers[i]} - {hd_layers[i+1]}'
            structure.append(res)

    def forward(self, x):
        if self.activation == "relu":
            activationFunction = F.relu
        elif self.activation == "tanh":
            activationFunction = F.tanh
        elif self.activation == "softmax":
            activationFunction = F.softmax
        elif self.activation == "LeakyReLU":
            activationFunction = F.leaky_relu(0.1)
        else:
            print("NESUNNA FUNZIONE DI ATTIVAZIONE!")
            exit(1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = activationFunction(x)

        return x