import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import utils
import torch.nn.functional as F


# NN model
class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2)
        self.linear3 = torch.nn.Linear(2, D_out)

    def forward(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = F.sigmoid(x)
        #x = torch.cat(torch.sin(x[:nb_relu_dim]), x[nb_relu_dim:])
        return self.linear3(self.linear2(x))

    def forward2(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = F.sigmoid(x)
        #x = torch.cat(torch.sin(x[:nb_relu_dim]), x[nb_relu_dim:])
        return self.linear2(x)


path = './model/model.pth'
model = torch.load(path)
#torch.manual_seed(231)

f = plt.figure()
f.add_axes([0, 0, 1.005, 1.005])
utils.plot_decision(model)
#utils.plot_decision(model)
#utils.plot_data(X, y)
plt.savefig(f'decision.png')
plt.show()