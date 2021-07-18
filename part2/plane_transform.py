from manimlib import *
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


class ContourGroup(VGroup):
    CONFIG = {
        "u_min": -FRAME_WIDTH/2,
        "u_max": FRAME_WIDTH/2,
        "v_min": -FRAME_HEIGHT/2,
        "v_max": FRAME_HEIGHT/2,
        "resolution": 256,
        "fill_opacity": 0.45,
    }

    def __init__(self, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.setup()

    def get_u_values_and_v_values(self):
        res = tuplify(self.resolution)
        if len(res) == 1:
            u_res = v_res = res[0]
        else:
            u_res, v_res = res
        u_min = self.u_min
        u_max = self.u_max
        v_min = self.v_min
        v_max = self.v_max

        u_values = np.linspace(u_min, u_max, u_res + 1)
        v_values = np.linspace(v_min, v_max, v_res + 1)

        return u_values, v_values

    def setup(self):
        u_values, v_values = self.get_u_values_and_v_values()
        faces = VGroup()
        for i in range(len(u_values) - 1):
            for j in range(len(v_values) - 1):
                u1, u2 = u_values[i:i + 2]
                v1, v2 = v_values[j:j + 2]
                face = VMobject()
                face.set_points_as_corners([
                    [u1, v1, 0],
                    [u2, v1, 0],
                    [u2, v2, 0],
                    [u1, v2, 0],
                    [u1, v1, 0],
                ])
                inp = torch.tensor([(u1+u2)/2, (v1+v2)/2], dtype=torch.float32)
                c = self.get_color(inp)
                face.set_color(c)
                faces.add(face)
        faces.set_fill(
            opacity=self.fill_opacity
        )
        faces.set_stroke(
            width=0,
        )
        self.add(*faces)

    def get_color(self, inp):
        return NotImplementedError


class DecisionContour(ContourGroup):
    def get_color(self, inp):
        return colors[np.argmax(model[3:].forward(inp).detach().numpy())]


class InputContour(ContourGroup):
    def get_color(self, inp):
        return colors[np.argmax(model.forward(inp).detach().numpy())]


class Test(Scene):
    def construct(self):
        n = NumberPlane()
        n.prepare_for_nonlinear_transform()
        n.apply_complex_function(lambda z: z**2)
        self.add(n)


class NNTransformPlane(Scene):
    def construct(self):
        n = NumberPlane()
        n.prepare_for_nonlinear_transform()
        n.apply_complex_function(self.function)
        self.add(n)

    def function(self, z):
        x, y = z.real, z.imag
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return x + y*1j


class NNTransformPlane(Scene):
    CONFIG = {
        "show_basis_vectors": False,
        "foreground_plane_kwargs": {
            "x_line_frequency": 0.5,
            "y_line_frequency": 0.5,
        }
    }

    def construct(self):
        dots = VGroup(
            *[
                Dot(self.function([point[0], point[1], 0]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )
        i = InputContour()
        d = DecisionContour()

        self.setup()

        self.play(FadeIn(i))
        self.wait()
        self.play(FadeOut(i))

        self.apply_nonlinear_transformation(
            self.function, added_anims=[Transform(init_dots, final_dots)], run_time=8)
        self.wait()

        self.play(FadeIn(d))
        self.wait()

    def function(self, point):
        x, y, z = point
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)
