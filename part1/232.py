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


torch.manual_seed(8)
model = Model(2, 3, 2)


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


class TestTransform(ThreeDScene):
    def construct(self):
        points = VGroup(*[
            Dot(2*np.array([point[0], point[1], 0]), color=rgb2hex(*colors[index]),
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)
        ])

        plane = NumberPlane()

        self.play(Write(plane))
        self.play(Write(points))
        self.wait()

        self.move_camera(0.8 * np.pi / 2, -0.45 * np.pi)

        points2 = VGroup(
            *[Dot(self.func(*point), color=rgb2hex(*colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points2))
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(10)

    def func(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y, z = z.detach().numpy()
        return 3 * np.array([x, y, 2*z])


class RandomTransform(LinearTransformationScene):
    CONFIG = {
        "show_basis_vectors": False,
        "foreground_plane_kwargs": {
            "x_line_frequency": 1,
            "y_line_frequency": 1,
        },
        "relu": False,
        "camera_class": ThreeDCamera
    }

    def construct(self):
        points = VGroup(*[
            Dot(2*np.array([point[0], point[1], 0]), color=rgb2hex(*colors[index]),
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)
        ])

        lines = VGroup()
        colors2 = [RED, BLUE]

        for i in range(2):
            w1, w2 = model.linear1.weight.data[i]
            b = model.linear1.bias.data[i]
            xin = np.linspace(-1.5, 1.5)
            lines.add(FunctionGraph(
                lambda x: (-1/w2) * (w1 * x + b),
                stroke_width=1.5 * DEFAULT_STROKE_WIDTH,
                color=colors2[i]
            ))

        self.setup()
        self.play(Write(points))
        self.play(Write(lines))
        self.wait()

        self.moving_mobjects += [*points]

        self.apply_nonlinear_transformation(self.func, run_time=6)
        self.wait()

        if self.relu:

            self.apply_nonlinear_transformation(lambda point: list(
                map(lambda x: max(x, 0), point)), run_time=6)
            self.wait()

            self.apply_nonlinear_transformation(self.func2, run_time=6)
            self.wait()


class FoldTransform(RandomTransform):
    CONFIG = {"relu": False}

    def func(self, point):
        x, y, z = point
        inp = torch.tensor(point[:2], dtype=torch.float32)
        y, h, z = model(inp)
        x, y = z.detach().numpy()
        return (x * RIGHT + y * UP)
