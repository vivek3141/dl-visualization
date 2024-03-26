import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from manimlib.imports import *
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.H = H
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x, nb_relu_dim=-1):
        z = self.linear1(x)
        if nb_relu_dim == self.H or nb_relu_dim == -1:
            h = F.relu(z)
        elif nb_relu_dim == 0:
            h = z
        else:
            h = torch.cat(F.relu(z[:nb_relu_dim]), z[nb_relu_dim:])
        return self.linear2(h), h, z


torch.manual_seed(6)
model = Model(2, 3, 2)
model.linear1.weight.data = 3*torch.tensor(
    [[-np.sqrt(2)/2, -np.sqrt(3)/3],
     [np.sqrt(2)/2, -np.sqrt(3)/3],
     [0, np.sqrt(3)/3]], dtype=torch.float32)
model.linear1.bias.data.fill_(1)

zieger = plt.imread('ziegler.png')

X = torch.randn(1000, 2)
H = torch.tanh(X)

x_min = -1
x_max = +1
colors = (X - x_min) / (x_max - x_min)
colors = (colors * 511).short().numpy()
colors = np.clip(colors, 0, 511)

colors = zieger[colors[:, 0], colors[:, 1]]



def rgb2hex(r, g, b):
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


class Part1(Scene):
    def construct(self):
        points = VGroup(*[
            Dot(2*np.array([point[0], point[1], 0]), color=rgb2hex(*colors[index]), stroke_color=BLACK,
                stroke_opacity=1, stroke_width=1,

                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)
        ])
        self.add(points)

    def func(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y = y.detach().numpy()
        return 1 * np.array([x, y, 0])


class Part2(Part1):
    def construct(self):
        points = VGroup(*[
            Dot(self.func(*point), color=rgb2hex(*colors[index]), stroke_color=BLACK,
                stroke_opacity=1, stroke_width=1,
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)
        ])
        self.add(points)


class Part3(Scene):
    def construct(self):
        self.add(NumberPlane())
