import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from manimlib import *
import torch.nn.functional as F

"""
Hi!
For those familiar with manim, this code won't work ona the master branch. Instead, you're gonna have to
use the shaders branch. It has some extra dependencies, so it could be a bit tricky to setup, but if
you're on mac or linux, using a package manager (brew, apt, pacman, etc.) and pip to install them worked for me.
-Vivek
"""


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

X = torch.randn(500, 2)
H = torch.tanh(X)

x_min = -1
x_max = +1
colors = (X - x_min) / (x_max - x_min)
colors = (colors * 511).short().numpy()
colors = np.clip(colors, 0, 511)

colors = zieger[colors[:, 0], colors[:, 1]]


def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*map(lambda x: int(x * 255), rgb))


def get_sphere(radius=0.12, shift=ORIGIN, color=RED, resolution=(21, 21)):
    sphere = Sphere(resolution=resolution)
    sphere.set_height(radius)
    sphere.set_color(color)
    sphere.shift(shift)
    return sphere


class FoldTransform(Scene):
    CONFIG = {
        "wait_duration": 10
    }

    def construct(self):
        frame = self.camera.frame
        points = SGroup()

        for index, point in enumerate(H):
            d = get_sphere(shift=list(2*point) +
                           [0], color=rgb2hex(colors[index]))
            # d = Sphere(color=rgb2hex(colors[index]),
            #        radius=0.75*DEFAULT_DOT_RADIUS).shift(list(2*point) + [0])
            points.add(d)

        plane = NumberPlane()

        self.play(Write(plane))
        self.play(FadeIn(points))
        self.wait()

        self.play(
            frame.set_theta, 0,
            frame.set_phi, 0.35 * PI
        )

        axes = ThreeDAxes()

        self.play(Write(axes))
        self.wait()

        rotate = True
        frame.add_updater(
            lambda m, dt: m.increment_theta(-0.2 * dt) if rotate else None
        )

        points2 = SGroup(
            *[get_sphere(shift=self.func(*point), color=rgb2hex(colors[index])) for index, point in enumerate(H)]
        )
        # /Users/vivek/manim/manimlib/camera/camera.py:71: RuntimeWarning: invalid value encountered in arccos
        # phi = np.arccos(Fz[2])
        self.play(Transform(points, points2), run_time=5)
        self.wait(self.wait_duration)

        points3 = SGroup(
            *[get_sphere(shift=self.func2(*point), color=rgb2hex(colors[index])) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points3), run_time=5)
        p = Polygon([3.85, 0, 0], [0, 3.85, 0], [0, 0, 1.9],
                    fill_opacity=0.5, color=WHITE)
        p2 = Polygon([3.85, 0, 0], [4.15, 0, 0], [5, 0.5, 0], [0.5, 5.25, 0], [
                     0, 4.15, 0], [0, 3.85, 0], fill_opacity=0.5, color=WHITE)
        p3 = Polygon([3.85, 0, 0], [0, 0, 1.9], [0, 0, 2.75], [
                     1.5, 0, 2.75], [4.15, 0, 0], fill_opacity=0.5, color=WHITE)
        p4 = Polygon([0, 0, 1.9], [0, 0, 2.75], [0, 1.5, 2.75], [
                     0, 4.15, 0], [0, 3.85, 0], fill_opacity=0.5, color=WHITE)
        ps = VGroup(p, p2, p3, p4)
        self.play(Write(ps))
        self.wait(self.wait_duration)

        points4 = SGroup(
            *[get_sphere(shift=self.func3(*point), color=rgb2hex(colors[index])) for index, point in enumerate(H)]
        )
        self.play(Uncreate(ps))
        self.play(Transform(points, points4), run_time=5)
        self.wait(2)

        rotate = False
        self.play(
            frame.set_phi, 0,
            frame.set_theta, 0
        )
        self.wait(2)

        # self.begin_ambient_camera_rotation(rate=0.1)
        # self.wait(10)

    def func(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y, z = z.detach().numpy()
        return 1 * np.array([x, y, 1*z])

    def func2(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y, z = h.detach().numpy()
        return 1 * np.array([x, y, 1*z])

    def func3(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y = y.detach().numpy()
        return 1 * np.array([x, y, 0])


class FoldTransform2(Scene):
    def construct(self):
        points = VGroup()

        for index, point in enumerate(H):
            d = Dot(list(2*point) + [0], color=rgb2hex(colors[index]),
                    radius=0.75*DEFAULT_DOT_RADIUS)
            points.add(d)

        plane = NumberPlane()

        self.play(Write(plane))
        self.play(Write(points))
        self.wait()

        points2 = VGroup(
            *[Dot(self.func(*point), color=rgb2hex(colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points2), run_time=6)
        self.wait()

    def func(self, x, y):
        inp = torch.tensor([x, y], dtype=torch.float32)
        y, h, z = model(inp)
        x, y = y.detach().numpy()
        return 1 * np.array([x, y, 0])


class TestTransform(Scene):
    CONFIG = {
        "m1": np.array([
            [-np.sqrt(2)/2, -np.sqrt(3)/3],
            [np.sqrt(2)/2, -np.sqrt(3)/3],
            [0, np.sqrt(3)/3]
        ])
    }

    def construct(self):
        frame = self.camera.frame
        points = VGroup()

        for index, point in enumerate(H):
            d = Dot(list(2*point) + [0], color=rgb2hex(colors[index]),
                    radius=0.75*DEFAULT_DOT_RADIUS)
            points.add(d)

        plane = NumberPlane()

        self.play(Write(plane))
        self.play(FadeIn(points))
        self.wait()

        self.play(
            frame.set_theta, 0,
            frame.set_phi, 0.35 * PI
        )

        axes = ThreeDAxes()
        self.play(Write(axes))
        self.wait()

        points2 = VGroup(
            *[Dot(self.func(point), color=rgb2hex(colors[index]),
                  radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(H)]
        )

        self.play(Transform(points, points2))
        self.wait()

    def func(self, point):
        x, y = point
        inp = np.array([[x], [y]])
        z = 3 * self.m1 @ inp + 1/3 * \
            np.array([[np.sqrt(3)], [np.sqrt(3)], [np.sqrt(3)]])
        # print(z)
        # from pdb import set_trace; set_trace()
        return z.T

    def relu(self, point):
        pass
