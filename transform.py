from manimlib.imports import *
import torch

path = './model/model.pth'
model = torch.load(path)
torch.manual_seed(231)


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
    return X.numpy(), y.numpy()


X, Y = get_data(c=5)

colors = [RED, YELLOW, GREEN, BLUE, PURPLE]


class DecisionQuad(VGroup):
    def __init__(self, func1, point, func2, color, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.add(Polygon(ORIGIN, ))

class Decisions(VGroup):
    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        M1 = [8, 0.15, -1.65]
        M2 = [1.75, 0.1]
        poly_args={
            "fill_opacity": 0.5,
            "stroke_width": 8,
            "color": RED
        }
        self.add(
            Polygon(ORIGIN, [FRAME_HEIGHT/8, FRAME_HEIGHT-0.1, 0], [FRAME_WIDTH/2, 0, 0], **poly_args)
        )
        self.add(*[
            FunctionGraph(lambda x: i * x, x_min=0) for i in M1
        ],
            *[
            FunctionGraph(lambda x: i * x, x_max=0) for i in M2
        ])
class NNTest(Scene):
    def construct(self):
        final_dots = VGroup(
            *[
                Dot(self.function([point[0], point[1], 0]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )
        d = Decisions()
        self.add(final_dots, d)
    def function(self, point):
        x, y, z = point
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)

class NNTransform(LinearTransformationScene):
    CONFIG = {
        "show_basis_vectors": False,
        "foreground_plane_kwargs": {
            "x_line_frequency": 0.5,
            "y_line_frequency": 0.5,
        }
    }

    def construct(self):
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
        d = Decisions()

        self.setup()
        self.add(init_dots)
        self.wait()
        self.apply_nonlinear_transformation(
            self.function, added_anims=[Transform(init_dots, final_dots)], run_time=6)
        self.wait()
        self.play(Write(d))
        self.wait()

    def function(self, point):
        x, y, z = point
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)
