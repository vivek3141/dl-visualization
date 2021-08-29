import torch
from torch import nn
import torch.nn.functional as F


class DecisionModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DecisionModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2)
        self.linear3 = torch.nn.Linear(2, D_out)

    def forward(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear3(self.linear2(x))

    def forward2(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)


class ModelRELU(DecisionModel):
    def set_activation_func(self):
        self.activation_func = F.relu


class ModelSin(DecisionModel):
    def __init__(self, *args):
        print("hello, inside constructor")
        DecisionModel.__init__(*args)
        

    def set_activation_func(self):
        print("setting activation func")
        self.activation_func = torch.sin


class ModelSigmoid(DecisionModel):
    def set_activation_func(self):
        self.activation_func = F.sigmoid
