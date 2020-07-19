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

zieger = plt.imread('ziegler.png')

X = torch.randn(1000, 2)
H = torch.tanh(X)

x_min = -1
x_max = +1
colors = (X - x_min) / (x_max - x_min)
colors = (colors * 511).short().numpy()
colors = np.clip(colors, 0, 511)

colors = zieger[colors[:, 0], colors[:, 1]]

def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*map(lambda x: int(x * 255), rgb))


class TestTransform(Scene):
    def construct(self):
        frame = self.camera.frame
        points = VGroup()

        for index, point in enumerate(H):
            d = Dot(list(2*point) + [0], color=rgb2hex(c),
                radius=0.75*DEFAULT_DOT_RADIUS)
            points.add(d)

        plane = NumberPlane()

        self.play(Write(plane))
        self.play(Write(points))
        self.wait()
        """
        self.play(
            frame.set_theta, 0,
            frame.set_phi, 0.35 * PI
        )

        
        rotate = True
        frame.add_updater(
            lambda m, dt: m.become(m.rotate(-0.2 * dt)) if rotate else None
        )

        points2 = VGroup(
            *[Dot(self.func(*point), color=rgb2hex(colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points2), run_time=5)
        self.wait(10)

        points3 = VGroup(
            *[Dot(self.func2(*point), color=rgb2hex(colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points3), run_time=5)
        self.wait(10)

        points4 = VGroup(
            *[Dot(self.func3(*point), color=rgb2hex(colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points4), run_time=5)
        self.wait(2)
        
        rotate = False
        self.play(
            frame.set_phi, 0,
            frame.set_theta, 0
        )

        #self.begin_ambient_camera_rotation(rate=0.1)
        #self.wait(10)
        """

    def func(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y, z = z.detach().numpy()
        return 3 * np.array([x, y, 2*z])
    
    def func2(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y, z = h.detach().numpy()
        return 3 * np.array([x, y, 2*z])

    def func3(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y = y.detach().numpy()
        return 6 * np.array([x, y, 0])



class Test(Scene):
    def construct(self):
        d = Dot(color="#dd01dd")

        self.play(DrawBorderThenFill(d))
        self.wait()


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