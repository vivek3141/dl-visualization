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


def get_dots(func, color_func=lambda idx: colors[Y[idx]]):
    return VGroup(
        *[
            Dot(func([point[0], point[1], 0]), color=color_func(index),
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
        ]
    )


X, Y = get_data(c=5)

colors = [RED, YELLOW, GREEN, BLUE, PURPLE]


class InteractiveScene(Scene):
    def interact(self):
        self.quit_interaction = False
        self.lock_static_mobject_data()
        try:
            while True:
                self.update_frame()
        except KeyboardInterrupt:
            self.unlock_mobject_data()


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

        init_dots = get_dots(lambda point: point)
        final_dots = get_dots(lambda point: self.func(point[:2]))

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

        f_plane.prepare_for_nonlinear_transform(num_inserted_curves=200)

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

    def func(self, point):
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

        final_dots = get_dots(lambda point: self.func(point[:2]))

        frame = self.camera.frame

        f_plane.prepare_for_nonlinear_transform(num_inserted_curves=200)
        f_plane.apply_complex_function(self.func_complex)

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(final_dots))
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

        s = SGroup()

        for i in range(5):
            surf = self.surface_func_softmax(
                i=i, u_range=(-4, 4), v_range=(-4, 4), color=colors[i], opacity=0.5)
            s.add(surf)

        p = SGroup()

        # def get_plane_points(i=0, scale=3/5, u_range=(-4, 4), v_range=(-4, 4)):
        #     def get_x(y):
        #         return -(self.w[i][1] * y + self.b[i])/self.w[i][0]

        #     def get_y(x):
        #         return -(self.w[i][0] * x + self.b[i])/self.w[i][1]

        #     def p_func(u, v):
        #         return [u, v, self.w[i][0] * u + self.w[i][1] * v + self.b[i]]

        #     points = []
        #     end = []

        #     x = get_x(v_range[0])
        #     if u_range[0] <= x <= u_range[1]:
        #         points.append([x, v_range[0], 0])
        #         end.append(max(
        #             p_func(u_range[0], v_range[0]),
        #             p_func(u_range[1], v_range[0]),
        #             key=lambda x: x[-1]))
        #     else:
        #         points.append([u_range[0], get_y(u_range[0]), 0])
        #         end.append(max(
        #             p_func(u_range[0], v_range[0]),
        #             p_func(u_range[0], v_range[1]),
        #             key=lambda x: x[-1]))

        #     x = get_x(v_range[1])
        #     if u_range[0] <= x <= u_range[1]:
        #         points.append([x, v_range[1], 0])
        #         end.append(max(
        #             p_func(u_range[0], v_range[1]),
        #             p_func(u_range[1], v_range[1]),
        #             key=lambda x: x[-1]))
        #     else:
        #         points.append([u_range[1], get_y(u_range[1]), 0])
        #         end.append(max(
        #             p_func(u_range[1], v_range[0]),
        #             p_func(u_range[1], v_range[1]),
        #             key=lambda x: x[-1]))

        #     points += end[::-1]
        #     return [[*points[i][:2], scale*points[i][2]] for i in range(4)]

        #     self.w[0][0] * u + self.w[0][1] * v + self.b[0]

        for i in range(5):
            plane = self.surface_func(i=i, scale=3/5, func=lambda x: max(
                x, 0), u_range=(-8, 8), v_range=(-4, 4), opacity=0.5, color=colors[i])
            p.add(plane)

        plane = ParametricSurface(
            self.surface_func_max(),
            u_range=(-FRAME_WIDTH/2, FRAME_WIDTH/2),
            v_range=(-FRAME_HEIGHT/2, FRAME_HEIGHT/2),
            resolution=(512, 512)
        )
        surf = TexturedSurface(
            plane,
            "/Users/vivek/python/nn-visualization/part2/output_decisions.png"
        )

        # for i in range(5):
        #     plane = Polygon(*get_plane_points(i=i), fill_opacity=0.5,
        #                     stroke_opacity=1, stroke_color=WHITE, fill_color=colors[i])
        #     p.add(plane)

        cp = s[0].copy()

        self.play(ShowCreation(cp))
        # self.wait()

        # self.embed()

        for i in range(5):
            self.wait(5)
            self.play(Transform(cp, s[i]))

        self.wait(5)

        self.play(ShowCreation(s[:-1]))
        self.wait(5)

        rotate = False
        self.play(
            frame.set_phi, 0,
            frame.set_theta, 0
        )
        self.wait()

        self.embed()

    def surface_func_max(self, i=6):
        return lambda u, v: [u, v, 0.5 * max(*(np.array([[u, v]]).dot(self.w.T) + self.b)[0][:i], 0)]

    def surface_func_softmax(self, i=0, scale=3, **kwargs):
        return ParametricSurface(lambda u, v: [u, v, scale * softmax((np.array([[u, v]]).dot(self.w.T) + self.b)[0])[i]], **kwargs)

    def surface_func(self, i=0, scale=3, func=sigmoid, **kwargs):
        return ParametricSurface(lambda u, v: [u, v, scale * func((np.array([[u, v]]).dot(self.w.T) + self.b)[0][i])], **kwargs)

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

    def func(self, point):
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
