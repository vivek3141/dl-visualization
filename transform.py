import matplotlib.pyplot as plt
from manimlib.imports import *
import torch


path = './K-5_2-Linear_H-100/model.pth'
model = torch.load(path)


def get_data(n=100, d=2, c=3, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    for i in range(c):
        index = 0
        r = torch.linspace(0.2, 1, n)
        t = torch.linspace(
            i * 2 * math.pi / c,
            (i + 2) * 2 * math.pi / c,
            n
        ) + torch.randn(n) * std

        for ix in range(n * i, n * (i + 1)):
            X[ix] = r[index] * torch.FloatTensor((
                math.sin(t[index]), math.cos(t[index])
            ))
            y[ix] = i
            index += 1
    return X, y


X, Y = get_data(c=5)
X = X.numpy()
Y = Y.numpy()


colors = [RED, YELLOW, GREEN, BLUE, PURPLE]


class NNTransform(LinearTransformationScene):
    CONFIG = {
        "show_basis_vectors": False,
    }

    def construct(self):
        NumberPlane
        init_dots = VGroup(
            *[
                Dot([point[0], point[1], 0], color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )
        final_dots = VGroup(
            *[
                Dot(self.function([point[0], point[1], 0]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )
        self.setup()
        self.add(init_dots)
        self.plane.prepare_for_nonlinear_transform()
        self.wait()
        self.apply_nonlinear_transformation(self.function, added_anims=[
                                            Transform(init_dots, final_dots)], run_time=6)
        self.wait()

    def function(self, point):
        x, y, z = point
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)
