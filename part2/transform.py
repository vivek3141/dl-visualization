from manimlib import *
import torch

path = './model/model.pth'
model = torch.load(path)
torch.manual_seed(231)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def relu(x):
    return max(x, 0)


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

        # p = SGroup()

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

        # for i in range(5):
        #     plane = self.surface_func(i=i, scale=3/5, func=lambda x: max(
        #         x, 0), u_range=(-8, 8), v_range=(-4, 4), opacity=0.5, color=colors[i])
        #     p.add(plane)

        SCALE = 0.4

        plane_kwargs = {
            "scale": SCALE,
            "u_range": (-FRAME_WIDTH/2, FRAME_WIDTH/2),
            "v_range": (-FRAME_HEIGHT/2, FRAME_HEIGHT/2),
            "opacity": 0.65,
        }

        red_plane1 = self.surface_func(
            i=0, color=colors[0], func=lambda x: x, **plane_kwargs
        )
        red_plane2 = self.surface_func(
            i=0, color=colors[0], func=relu, **plane_kwargs
        )

        yellow_plane = self.surface_func(
            i=1, color=colors[1], func=relu, **plane_kwargs
        )

        p = SGroup()

        for i in range(1, 6):
            plane = ParametricSurface(
                self.surface_func_max(i=i, scale=SCALE),
                resolution=(128, 128),
                **plane_kwargs
            )
            surf = TexturedSurface(
                plane,
                f"./img/plane{i-1}.png"
            )
            p.add(surf)

        def plane_intersect(a, b):
            a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

            aXb_vec = np.cross(a_vec, b_vec)

            A = np.array([a_vec, b_vec, aXb_vec])
            d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

            p_inter = np.linalg.solve(A, d).T

            return p_inter[0], (p_inter + aXb_vec)[0]

        def intersection(a, b):
            da = a[1] - a[0]
            db = b[1] - b[0]
            dc = b[0] - a[0]
            s = np.dot(np.cross(dc, db), np.cross(da, db)) / \
                np.linalg.norm(np.cross(da, db))**2
            return a[0] + s * da

        def get_plane_intersect(i1, i2):
            return plane_intersect(SCALE * np.array([*w[i1], -1/SCALE, b[i1]]), SCALE * np.array([*w[i2], -1/SCALE, b[i2]]))

        def get_bound(line, coeff, i):
            v = line[1] - line[0]
            const = FRAME_HEIGHT if i else FRAME_WIDTH
            t = (coeff * const/2 - line[0][i])/v[i]
            return line[0] + t * v

        lines = VGroup()
        lines_c = []

        for i in range(4):
            i_points = get_plane_intersect(i, i+1)
            lines_c.append([i_points[0], i_points[1]])
            lines.add(Line(i_points[0], i_points[1]))

        """
        Following this comment, is by far, the worst code I've ever written in my life. Please clean your eyes before and after viewing this. Thanks!
        """

        decision_planes = VGroup()

        red_purple_line = get_plane_intersect(0, -1)
        red_blue_line = get_plane_intersect(0, 3)

        purple_p = [
            intersection(red_purple_line, lines_c[3]),
            get_bound(red_purple_line, -1, 1),
            get_bound(lines_c[3], -1, 1)
        ]
        purple_fplane = Polygon(
            *purple_p, color=colors[4], stroke_opacity=0, fill_opacity=plane_kwargs["opacity"])

        red_p = [
            purple_p[0],
            purple_p[1],
            self.surface_func_max(scale=SCALE)(FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            get_bound(lines_c[0], 1, 0),
            intersection(lines_c[0], red_blue_line)
        ]
        red_fplane = Polygon(
            *red_p, color=colors[0], stroke_opacity=0, fill_opacity=plane_kwargs["opacity"])

        yellow_p = [
            red_p[4],
            red_p[3],
            self.surface_func_max(scale=SCALE)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            get_bound(lines_c[1], 1, 1),
            get_bound(lines_c[1], 1, 1),
        ]
        yellow_fplane = Polygon(
            *yellow_p, color=colors[1], stroke_opacity=0, fill_opacity=plane_kwargs["opacity"])

        green_p = [
            yellow_p[-1],
            yellow_p[-1],
            yellow_p[-2],
            self.surface_func_max(scale=SCALE)(-FRAME_WIDTH/2, FRAME_HEIGHT/2),
            get_bound(lines_c[2], -1, 0)
        ]
        green_fplane = Polygon(
            *green_p, color=colors[2], stroke_opacity=0, fill_opacity=plane_kwargs["opacity"])

        blue_p = [
            green_p[0],
            green_p[3],
            self.surface_func_max(
                scale=SCALE)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            purple_p[2],
            purple_p[0],
            red_p[4]
        ]
        blue_fplane = Polygon(
            *blue_p, color=colors[3], stroke_opacity=0, fill_opacity=plane_kwargs["opacity"])

        final_decision_plane = VGroup(
            purple_fplane, red_fplane, yellow_fplane, green_fplane, blue_fplane)

        self.embed()

        self.play(ShowCreation(red_plane1))
        self.wait()

        self.play(Transform(red_plane1, red_plane2))
        self.wait()

        self.play(ShowCreation(yellow_plane))
        self.wait()

        self.play(ReplacementTransform(SGroup(red_plane1, yellow_plane), p[1]))
        self.wait()

        for i in range(1, 4):
            self.play(ReplacementTransform(p[i], p[i+1]))
            self.wait()

        self.play(Uncreate(p[4]))
        self.wait()

        # surf = TexturedSurface(
        #     plane,
        #     "/Users/vivek/python/nn-visualization/part2/output_decisions.png"
        # )

        # for i in range(5):
        #     plane = Polygon(*get_plane_points(i=i), fill_opacity=0.5,
        #                     stroke_opacity=1, stroke_color=WHITE, fill_color=colors[i])
        #     p.add(plane)

        cp = s[0].copy()

        self.play(ShowCreation(cp))

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

    def surface_func_max(self, i=6, scale=0.4, func=lambda *x: max(*x, 0)):
        return lambda u, v: [u, v, scale * func(*(np.array([[u, v]]).dot(self.w.T) + self.b)[0][:i])]

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


class Test(NNTransformPlane):
    def construct(self):
        frame = self.camera.frame
        frame.set_phi(0.35*PI)
        plane_kwargs = {
            "scale": 0.5,
            "u_range": (-FRAME_WIDTH/2, FRAME_WIDTH/2),
            "v_range": (-FRAME_HEIGHT/2, FRAME_HEIGHT/2),
            "opacity": 0.65,
        }

        w = model[3].weight.detach().numpy()
        b = model[3].bias.detach().numpy()
        Camera

        self.w = w
        self.b = b

        red_plane1 = self.surface_func(
            i=0, color=colors[0], func=lambda x: x, **plane_kwargs
        )
        red_plane2 = self.surface_func(
            i=0, color=colors[0], func=relu, **plane_kwargs
        )

        yellow_plane1 = self.surface_func(
            i=1, color=colors[1], func=lambda x: x, **plane_kwargs
        )
        yellow_plane2 = self.surface_func(
            i=1, color=colors[1], func=relu, **plane_kwargs
        )

        p = SGroup()

        for i in range(1, 6):
            plane = ParametricSurface(
                self.surface_func_max(i=i),
                resolution=(128, 128),
                **plane_kwargs
            )
            surf = TexturedSurface(
                plane,
                f"./img/plane{i-1}.png"
            )
            p.add(surf)

        self.play(ShowCreation(p[1]))
        self.play(FadeTransform(p[1], p[2]))
        self.wait()
