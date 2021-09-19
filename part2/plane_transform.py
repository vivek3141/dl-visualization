from manimlib import *
import torch

path = './model/model.pth'
model = torch.load(path)
torch.manual_seed(231)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


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


def tuplify(obj):
    if isinstance(obj, str):
        return (obj,)
    try:
        return tuple(obj)
    except TypeError:
        return (obj,)


class InteractiveScene(Scene):
    def interact(self):
        self.quit_interaction = False
        self.lock_static_mobject_data()
        try:
            while True:
                self.update_frame()
        except KeyboardInterrupt:
            self.unlock_mobject_data()


# class ContourGroup(VGroup):
#     CONFIG = {
#         "u_min": -FRAME_WIDTH/2,
#         "u_max": FRAME_WIDTH/2,
#         "v_min": -FRAME_HEIGHT/2,
#         "v_max": FRAME_HEIGHT/2,
#         "resolution": 10,
#         "fill_opacity": 0.45,
#     }

#     def __init__(self, **kwargs):
#         VGroup.__init__(self, **kwargs)
#         self.setup()

#     def get_u_values_and_v_values(self):
#         res = tuplify(self.resolution)
#         if len(res) == 1:
#             u_res = v_res = res[0]
#         else:
#             u_res, v_res = res
#         u_min = self.u_min
#         u_max = self.u_max
#         v_min = self.v_min
#         v_max = self.v_max

#         u_values = np.linspace(u_min, u_max, u_res + 1)
#         v_values = np.linspace(v_min, v_max, v_res + 1)

#         return u_values, v_values

#     def setup(self):
#         u_values, v_values = self.get_u_values_and_v_values()
#         faces = VGroup()
#         for i in range(len(u_values) - 1):
#             for j in range(len(v_values) - 1):
#                 u1, u2 = u_values[i:i + 2]
#                 v1, v2 = v_values[j:j + 2]
#                 face = VMobject()
#                 face.set_points_as_corners([
#                     [u1, v1, 0],
#                     [u2, v1, 0],
#                     [u2, v2, 0],
#                     [u1, v2, 0],
#                     [u1, v1, 0],
#                 ])
#                 inp = torch.tensor([(u1+u2)/2, (v1+v2)/2], dtype=torch.float32)
#                 c = self.get_color(inp)
#                 face.set_color(c)
#                 faces.add(face)
#         faces.set_fill(
#             opacity=self.fill_opacity
#         )
#         faces.set_stroke(
#             width=0,
#         )
#         self.add(*faces)

#     def get_color(self, inp):
#         return NotImplementedError


class ContourGroup(VGroup):
    CONFIG = {
        "count": 5,
        "fill_opacity": 0.5,
        "x_min": -FRAME_WIDTH/2,
        "x_max": FRAME_WIDTH/2,
        "y_min": -FRAME_HEIGHT/2,
        "y_max": FRAME_HEIGHT/2,
    }

    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.setup()

    def setup(self):
        x_values = np.linspace(self.x_min, self.x_max, self.count + 1)
        y_values = np.linspace(self.y_min, self.y_max, self.count + 1)

        for i in range(len(x_values) - 1):
            for j in range(len(y_values) - 1):
                x1, x2 = x_values[i:i + 2]
                y1, y2 = y_values[j:j + 2]

                inp = torch.tensor(
                    [(x1 + x2)/2, (y1 + y2)/2], dtype=torch.float32)
                c = self.get_color(inp)

                face = Polygon(
                    [x1, y1, 0],
                    [x2, y1, 0],
                    [x2, y2, 0],
                    [x1, y2, 0],
                    color=c,
                    stroke_width=0,
                    fill_opacity=self.fill_opacity,
                )

                # face = VMobject(
                #     color=c,
                #     stroke_width=0,
                #     fill_opacity=self.fill_opacity,
                # )
                # face.set_points_as_corners([
                #     [x1, y1, 0],
                #     [x2, y1, 0],
                #     [x2, y2, 0],
                #     [x1, y2, 0],
                #     [x1, y1, 0],
                # ])

                self.add(face)

    def get_color(self, inp):
        return NotImplementedError


class DecisionTest(InteractiveScene):
    def construct(self):
        d = ImageMobject("output_decisions.png", height=FRAME_HEIGHT)
        d.set_opacity(0.65)
        self.add(d)
        self.embed()


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


class NNTransform(Scene):
    CONFIG = {
        "foreground_plane_kwargs": {
            "faded_line_ratio": 1,
        },
        "background_plane_kwargs": {
            "color": GREY,
            "axis_config": {
                "stroke_color": GREY_B,
            },
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
            "faded_line_ratio": 1
        }
    }

    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        init_dots = VGroup(
            *[
                Dot([point[0], point[1], 0], color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )

        final_dots = VGroup(
            *[
                Dot(self.func_real([point[0], point[1]]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )

        i = ImageMobject("relu_inp_decisions.png", height=FRAME_HEIGHT)
        i.set_opacity(0.65)

        d = ImageMobject("output_decisions.png", height=FRAME_HEIGHT)
        d.set_opacity(0.65)

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(init_dots))
        self.wait()

        self.play(FadeIn(i))
        self.wait()

        self.play(FadeOut(i))
        self.wait()

        #self.add(b_plane, f_plane, init_dots)

        f_plane.prepare_for_nonlinear_transform()
        # f_plane.apply_complex_function(self.func_complex)
        # self.add(b_plane, f_plane, final_dots)

        self.play(
            ApplyMethod(f_plane.apply_complex_function, self.func_complex),
            Transform(init_dots, final_dots),
            run_time=8
        )

        self.wait()

        self.play(FadeIn(d))
        self.wait()

    def func_complex(self, z):
        inp = torch.tensor([z.real, z.imag], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x + y*1j)

    def func_real(self, point):
        inp = torch.tensor(point, dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)


class NNTransformPlane(Scene):
    CONFIG = {
        "foreground_plane_kwargs": {
            "faded_line_ratio": 1,
        },
        "background_plane_kwargs": {
            "color": GREY,
            "axis_config": {
                "stroke_color": GREY_B,
            },
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
            "faded_line_ratio": 1
        }
    }

    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        init_dots = VGroup(
            *[
                Dot([point[0], point[1], 0], color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )

        final_dots = VGroup(
            *[
                Dot(self.func_real([point[0], point[1]]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )

        frame = self.camera.frame

        d = ImageMobject("output_decisions.png", height=FRAME_HEIGHT)
        d.set_opacity(0.65)

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(init_dots))
        self.wait()

        #self.add(b_plane, f_plane, init_dots)

        f_plane.prepare_for_nonlinear_transform()
        # f_plane.apply_complex_function(self.func_complex)
        # self.add(b_plane, f_plane, final_dots)

        self.play(
            ApplyMethod(f_plane.apply_complex_function, self.func_complex),
            Transform(init_dots, final_dots),
            run_time=8
        )

        self.wait()

        self.play(FadeIn(d))
        self.wait()

        self.play(frame.set_phi, 0.35*PI)

        rotate = True
        frame.add_updater(
            lambda m, dt: m.increment_theta(-0.2 * dt)
            if rotate else None
        )

        w = model[3].weight.detach().numpy()
        b = model[3].bias.detach().numpy()

        self.w = w
        self.b = b

        # planes = self.get_planes_cut(z_max=1, y_max=3, stroke_color=WHITE,
        #                              fill_color=BLACK, fill_opacity=0.5)

        # lines = self.get_lines(
        #     t_min=-4, t_max=4, stroke_width=6, stroke_color=PINK)

        s = SGroup()

        for i in range(5):
            surf = self.surface_func_softmax(
                i=i, u_range=(-4, 4), v_range=(-4, 4), color=colors[i], opacity=0.5)
            s.add(surf)

        self.play(ShowCreation(s[0]))

        for i in range(5):
            self.wait(5)
            self.play(Transform(s[0], s[i]))

        self.wait(5)

        # self.embed()

    def surface_func_softmax(self, i=0, scale=3, **kwargs):
        return ParametricSurface(lambda u, v: [u, v, scale * softmax(self.w.dot(np.array([[u], [v]]) + self.b[0]))[i]], **kwargs)

    def surface_func(self, i=0, scale=3, activation=sigmoid, **kwargs):
        return ParametricSurface(lambda u, v: [u, v, scale * activation(self.w[0][0] * u + self.w[0][1] * v + self.b[0])], **kwargs)

    @staticmethod
    def get_plane_func(w0, w1, b):
        return lambda u, v: [u, v, w0*u+w1*v+b]

    def get_lines(self, **kwargs):
        lines = VGroup()

        for i in range(5):
            lines.add(
                ParametricCurve(
                    lambda t: [t, -(self.w[i][0] * t + self.b[i])/self.w[i][1], 0], **kwargs)
            )

        return lines

    def get_planes(self, **kwargs):
        planes = SGroup()

        for i in range(5):
            p1 = self.get_plane_func(self.w[i][0], self.w[i][1], self.b[i])
            p = self.get_plane(p1, **kwargs)
            planes.add(p)

        return planes

    def get_planes_cut(self, z_max=1, y_max=3, **kwargs):
        planes = SGroup()

        for i in range(5):
            def func(y, z):
                return [(z - (self.w[i][1] * y + self.b[i])) / self.w[i][0], y, z]

            vertices = list()

            for y in range(-1, 2, 2):
                for z in range(-1, 2, 2):
                    if y > 0:
                        z *= -1

                    vertices.append(func(y * y_max, z * z_max))

            planes.add(Polygon(*vertices, **kwargs))

        return planes

    def get_plane(self, func, u_max=3, v_max=3, **kwargs):
        vertices = []

        for x in range(-1, 2, 2):
            for y in range(-1, 2, 2):
                if x > 0:
                    y *= -1
                vertices.append(func(u_max*x, v_max*y))

        return Polygon(*vertices, **kwargs)

    def func_complex(self, z):
        inp = torch.tensor([z.real, z.imag], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x + y*1j)

    def func_real(self, point):
        inp = torch.tensor(point, dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)

    def interact(self):
        self.quit_interaction = False
        self.lock_static_mobject_data()
        try:
            while True:
                self.update_frame()
        except KeyboardInterrupt:
            self.unlock_mobject_data()


class NNTransformPlane2(Scene):
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
