import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from manimlib.imports import *

torch.manual_seed(8348)
# 8348


def rgb2hex(r, g, b):
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


zieger = plt.imread('ziegler.png')

X = torch.randn(1000, 2)
H = torch.tanh(X)

x_min = -1
x_max = +1
colors = (X - x_min) / (x_max - x_min)
colors = (colors * 511).short().numpy()
colors = np.clip(colors, 0, 511)

colors = zieger[colors[:, 0], colors[:, 1]]
model = nn.Sequential(nn.Linear(2, 100), nn.Linear(100, 2))


class RandomTransform(LinearTransformationScene):
    CONFIG = {
        "show_basis_vectors": False,
        "foreground_plane_kwargs": {
            "x_line_frequency": 1,
            "y_line_frequency": 1,
        }
    }

    def construct(self):
        points = VGroup(*[
            Dot(2*np.array([point[0], point[1], 0]), color=rgb2hex(*colors[index]),
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)
        ])
        self.setup()
        self.play(Write(points))
        self.wait()

        self.moving_mobjects += [*points]

        self.apply_nonlinear_transformation(self.func)
        self.wait()

    def func(self, point):
        x, y, z = point
        inp = torch.tensor(point[:2], dtype=torch.float32)
        x, y = model.forward(inp).detach().numpy()
        return 5*(x * RIGHT + y * UP)


torch.manual_seed(8348)
model2 = nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2))


class ReluTransform(RandomTransform):
    def func(self, point):
        x, y, z = point
        inp = torch.tensor(point[:2], dtype=torch.float32)
        x, y = model2.forward(inp).detach().numpy()
        return 5*(x * RIGHT + y * UP)
