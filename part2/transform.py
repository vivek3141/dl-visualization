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

RED_rgb = (252, 98, 85)
YELLOW_rgb = (255, 255, 0)
GREEN_rgb = (131, 193, 103)
BLUE_rgb = (88, 196, 221)
PURPLE_rgb = (154, 114, 172)

colors_rgb = [RED_rgb, YELLOW_rgb, GREEN_rgb, BLUE_rgb, PURPLE_rgb]
colors_rgb = [[i/256 for i in color] for color in colors_rgb]


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
        },
        "camera_config": {
            "samples": 1,
            "anti_alias_width": 1.5
        },
        "always_update_mobjects": True,
    }

    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        final_dots = get_dots(lambda point: self.func(point[:2]))

        frame = self.camera.frame

        f_plane.prepare_for_nonlinear_transform(num_inserted_curves=300)
        f_plane.apply_complex_function(self.func_complex)

        self.play(Write(b_plane))  # , Write(f_plane))
        self.play(Write(final_dots))
        self.wait()

        self.play(frame.set_phi, 0.25*PI)

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

        SCALE = 0.4

        plane_kwargs = {
            "scale": SCALE,
            "u_range": (-FRAME_WIDTH/2, FRAME_WIDTH/2),
            "v_range": (-FRAME_HEIGHT/2, FRAME_HEIGHT/2),
            "opacity": 0.65,
            "resolution": (50, 50)
        }

        vector_plane_kwargs = {
            "stroke_width": 0,
            "fill_opacity": plane_kwargs["opacity"]
        }

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

        def check(obj):
            self.remove(obj)
            self.wait(0.5)
            self.add(obj)

        """
        Following this comment, is by far, the worst code I've ever written in my life. Please clean your eyes before and after viewing this. Thanks!
        """

        # Initial Red Plane

        red_line = [
            np.array([1, -(self.w[0][0] * 1 + self.b[0])/self.w[0][1], 0]),
            np.array([-1, -(self.w[0][0] * -1 + self.b[0])/self.w[0][1], 0])
        ]

        y_minus = get_bound(red_line, -1, 1)
        y_plus = get_bound(red_line, 1, 1)

        red_points0 = [
            self.surface_func_max(i=1)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            self.surface_func_max(i=1)(-FRAME_WIDTH/2, FRAME_HEIGHT/2),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, -FRAME_HEIGHT/2)
        ]

        red_plane0 = Polygon(
            *red_points0, **vector_plane_kwargs, color=colors[0]
        )

        red_points = [
            y_minus,
            y_plus,
            self.surface_func_max(i=1)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, -FRAME_HEIGHT/2)
        ]

        red_plane = Polygon(
            *red_points, **vector_plane_kwargs, color=colors[0]
        )

        red_line0 = Line(
            get_bound(red_line, -1, 1),
            get_bound(red_line, 1, 1),
            stroke_width=4,
            color=colors[0]
        )

        self.play(
            Write(red_plane0),
            Write(red_line0)
        )
        self.wait(5)

        # Red + Yellow intersection

        red_yellow_line = get_plane_intersect(0, 1)
        z_plane = np.array([0, 0, 1, 0])

        red_plane_eq = SCALE * np.array([*w[0], -1/SCALE, b[0]])
        red_z = plane_intersect(red_plane_eq, z_plane)

        yellow_plane_eq = SCALE * np.array([*w[1], -1/SCALE, b[1]])
        yellow_z = plane_intersect(yellow_plane_eq, z_plane)

        yellow_points = [
            intersection(red_z, red_yellow_line),
            get_bound(red_yellow_line, 1, 0),
            self.surface_func_max(i=2)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            get_bound(yellow_z, 1, 1)
        ]

        yellow_plane = Polygon(
            *yellow_points, **vector_plane_kwargs, color=colors[1]
        )

        red_points2 = [
            intersection(red_z, red_yellow_line),
            get_bound(red_yellow_line, 1, 0),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            get_bound(red_line, 1, 1)
        ]
        red_plane2 = Polygon(
            *red_points2, **vector_plane_kwargs, color=colors[0]
        )

        red_points3 = [
            intersection(red_z, red_yellow_line),
            get_bound(red_yellow_line, 1, 0),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            get_bound(red_z, -1, 1),
        ]
        red_plane3 = Polygon(
            *red_points3, **vector_plane_kwargs, color=colors[0]
        )

        red_line1 = Line(
            get_bound(red_line, -1, 1),
            intersection(red_z, red_yellow_line),
            stroke_width=4,
            color=colors[0]
        )

        yellow_line1 = Line(
            intersection(red_z, red_yellow_line),
            get_bound(yellow_z, 1, 1),
            stroke_width=4,
            color=colors[1]
        )

        red_line01 = Line(
            intersection(red_z, red_yellow_line),
            get_bound(red_line, 1, 1),
            stroke_width=4,
            color=colors[0]
        )

        self.play(ReplacementTransform(red_plane0, red_plane))
        self.wait(5)

        self.remove(red_plane, red_line0)
        self.add(red_plane2, red_plane3, red_line01, red_line1)

        self.play(
            ReplacementTransform(red_plane2, yellow_plane),
            ReplacementTransform(red_line01, yellow_line1),
        )
        self.wait(10)

        # Red + Yellow + Green Plane

        yellow_green_line = get_plane_intersect(1, 2)
        green_plane_eq = SCALE * np.array([*w[2], -1/SCALE, b[2]])
        green_z = plane_intersect(green_plane_eq, z_plane)

        green_points = [
            intersection(yellow_z, yellow_green_line),
            get_bound(yellow_green_line, 1, 1),
            self.surface_func_max(i=3)(-FRAME_WIDTH/2, FRAME_HEIGHT/2),
            self.surface_func_max(i=3)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            get_bound(green_z, -1, 1)
        ]
        green_plane = Polygon(
            *green_points, **vector_plane_kwargs, color=colors[2])

        yellow_points2 = [
            green_points[0],
            *yellow_points[:3],
            green_points[1]
        ]
        yellow_plane2 = Polygon(
            *yellow_points2, **vector_plane_kwargs, color=colors[1])

        z_green_points = [[*i[:2], 0] for i in green_points]
        z_green_plane = Polygon(
            *z_green_points, fill_opacity=0, stroke_width=0)

        yellow_line2 = Line(
            intersection(red_z, red_yellow_line),
            intersection(yellow_z, yellow_green_line),
            stroke_width=4,
            color=colors[1]
        )

        yellow_line02 = Line(
            get_bound(yellow_z, 1, 1),
            intersection(yellow_z, yellow_green_line),
            stroke_width=4,
            color=colors[1]
        )

        green_line = Line(
            intersection(yellow_z, yellow_green_line),
            get_bound(green_z, -1, 1),
            stroke_width=4,
            color=colors[2]
        )

        self.remove(yellow_line1)
        self.add(yellow_line2, yellow_line02)

        self.play(
            ReplacementTransform(yellow_plane, yellow_plane2),
            ReplacementTransform(z_green_plane, green_plane),
            ReplacementTransform(yellow_line02, green_line)
        )
        self.wait(5)

        # Red + Yellow + Green + Blue Plane

        blue_green_line = get_plane_intersect(2, 3)
        blue_yellow_line = get_plane_intersect(1, 3)
        blue_red_line = get_plane_intersect(0, 3)

        blue_points = [
            get_bound(blue_red_line, -1, 1),
            intersection(blue_red_line, blue_yellow_line),
            intersection(blue_yellow_line, blue_green_line),
            get_bound(blue_green_line, -1, 0),
            self.surface_func_max(i=4)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2)
        ]
        blue_plane = Polygon(
            *blue_points, **vector_plane_kwargs, color=colors[3])

        green_points2 = [
            blue_points[2],
            *green_points[1:3],
            *blue_points[3:1:-1]
        ]
        green_plane2 = Polygon(
            *green_points2, **vector_plane_kwargs, color=colors[2])

        red_points4 = [
            blue_points[1],
            *red_points3[1:3],
            blue_points[0]
        ]
        red_plane4 = Polygon(
            *red_points4, **vector_plane_kwargs, color=colors[0])

        yellow_points3 = [
            blue_points[2],
            blue_points[1],
            *yellow_points2[2:]
        ]
        yellow_plane3 = Polygon(
            *yellow_points3, **vector_plane_kwargs, color=colors[1])

        z_blue_points = [[*i[:2], 0] for i in blue_points]
        z_blue_plane = Polygon(
            *z_blue_points, fill_opacity=0, stroke_width=0)

        self.play(
            Uncreate(VGroup(red_line1, yellow_line2, green_line)),
            ReplacementTransform(red_plane3, red_plane4),
            ReplacementTransform(green_plane, green_plane2),
            ReplacementTransform(z_blue_plane, blue_plane),
            ReplacementTransform(yellow_plane2, yellow_plane3)
        )
        self.wait(5)

        # Final Decision Plane

        red_purple_line = get_plane_intersect(0, 4)
        blue_purple_line = get_plane_intersect(3, 4)

        purple_points = [
            intersection(red_purple_line, blue_purple_line),
            get_bound(red_purple_line, -1, 1),
            get_bound(blue_purple_line, -1, 1),
        ]
        purple_plane = Polygon(
            *purple_points, **vector_plane_kwargs, color=colors[4])

        red_points5 = [
            *red_points4[:-1],
            *purple_points[0:2][::-1]
        ]
        red_plane5 = Polygon(
            *red_points5, **vector_plane_kwargs, color=colors[0])

        blue_points2 = [
            purple_points[2],
            purple_points[0],
            *blue_points[1:]
        ]
        blue_plane2 = Polygon(
            *blue_points2, **vector_plane_kwargs, color=colors[3])

        z_purple_points = [[*i[:2], 0] for i in purple_points]
        z_purple_plane = Polygon(
            *z_purple_points, fill_opacity=0, stroke_width=0)

        self.play(
            ReplacementTransform(blue_plane, blue_plane2),
            ReplacementTransform(red_plane4, red_plane5),
            ReplacementTransform(z_purple_plane, purple_plane)
        )
        self.wait(10)

        rotate = False
        self.play(
            frame.set_phi, 0,
            frame.set_theta, -1 * TAU
        )
        self.wait()

        rotate = True
        self.play(frame.set_phi, 0.35*PI)
        self.play(Uncreate(VGroup(red_plane5, yellow_plane3,
                  green_plane2, blue_plane2, purple_plane)))
        self.wait()

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
            frame.set_theta, -2 * TAU
        )
        self.wait()

        self.embed()

    def surface_func_max(self, i=6, scale=0.4, func=lambda x: max(x)):
        return lambda u, v: [u, v, scale * func((np.array([[u, v]]).dot(self.w.T) + self.b)[0][:i])]

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


class Tweet(Scene):
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
        },
        "camera_config": {
            "samples": 1,
            "anti_alias_width": 1.5
        },
        "always_update_mobjects": True,
    }

    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        final_dots = get_dots(lambda point: self.func(point[:2]))
        init_dots = get_dots(lambda point: point)

        frame = self.camera.frame
        f_plane.prepare_for_nonlinear_transform(num_inserted_curves=200)
        f_plane.apply_complex_function(self.func_complex)

        self.play(Write(b_plane), Write(f_plane), Write(final_dots))

        rotate = False
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

        SCALE = 0.4

        plane_kwargs = {
            "scale": SCALE,
            "u_range": (-FRAME_WIDTH/2, FRAME_WIDTH/2),
            "v_range": (-FRAME_HEIGHT/2, FRAME_HEIGHT/2),
            "opacity": 0.65,
            "resolution": (50, 50)
        }

        vector_plane_kwargs = {
            "stroke_width": 0,
            "fill_opacity": plane_kwargs["opacity"]
        }

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

        def check(obj):
            self.remove(obj)
            self.wait(0.5)
            self.add(obj)

        """
        Following this comment, is by far, the worst code I've ever written in my life. Please clean your eyes before and after viewing this. Thanks!
        """

        # Initial Red Plane

        red_line = [
            np.array([1, -(self.w[0][0] * 1 + self.b[0])/self.w[0][1], 0]),
            np.array([-1, -(self.w[0][0] * -1 + self.b[0])/self.w[0][1], 0])
        ]

        # Red + Yellow intersection

        red_yellow_line = get_plane_intersect(0, 1)
        z_plane = np.array([0, 0, 1, 0])

        red_plane_eq = SCALE * np.array([*w[0], -1/SCALE, b[0]])
        red_z = plane_intersect(red_plane_eq, z_plane)

        yellow_plane_eq = SCALE * np.array([*w[1], -1/SCALE, b[1]])
        yellow_z = plane_intersect(yellow_plane_eq, z_plane)

        yellow_points = [
            intersection(red_z, red_yellow_line),
            get_bound(red_yellow_line, 1, 0),
            self.surface_func_max(i=2)(FRAME_WIDTH/2, FRAME_HEIGHT/2),
            get_bound(yellow_z, 1, 1)
        ]

        red_points3 = [
            intersection(red_z, red_yellow_line),
            get_bound(red_yellow_line, 1, 0),
            self.surface_func_max(i=1)(FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            get_bound(red_z, -1, 1),
        ]
        # Red + Yellow + Green Plane

        yellow_green_line = get_plane_intersect(1, 2)
        green_plane_eq = SCALE * np.array([*w[2], -1/SCALE, b[2]])
        green_z = plane_intersect(green_plane_eq, z_plane)

        green_points = [
            intersection(yellow_z, yellow_green_line),
            get_bound(yellow_green_line, 1, 1),
            self.surface_func_max(i=3)(-FRAME_WIDTH/2, FRAME_HEIGHT/2),
            self.surface_func_max(i=3)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2),
            get_bound(green_z, -1, 1)
        ]

        yellow_points2 = [
            green_points[0],
            *yellow_points[:3],
            green_points[1]
        ]

        # Red + Yellow + Green + Blue Plane

        blue_green_line = get_plane_intersect(2, 3)
        blue_yellow_line = get_plane_intersect(1, 3)
        blue_red_line = get_plane_intersect(0, 3)

        blue_points = [
            get_bound(blue_red_line, -1, 1),
            intersection(blue_red_line, blue_yellow_line),
            intersection(blue_yellow_line, blue_green_line),
            get_bound(blue_green_line, -1, 0),
            self.surface_func_max(i=4)(-FRAME_WIDTH/2, -FRAME_HEIGHT/2)
        ]

        green_points2 = [
            blue_points[2],
            *green_points[1:3],
            *blue_points[3:1:-1]
        ]
        green_plane2 = Polygon(
            *green_points2, **vector_plane_kwargs, color=colors[2])

        red_points4 = [
            blue_points[1],
            *red_points3[1:3],
            blue_points[0]
        ]

        yellow_points3 = [
            blue_points[2],
            blue_points[1],
            *yellow_points2[2:]
        ]
        yellow_plane3 = Polygon(
            *yellow_points3, **vector_plane_kwargs, color=colors[1])

        # Final Decision Plane

        red_purple_line = get_plane_intersect(0, 4)
        blue_purple_line = get_plane_intersect(3, 4)

        purple_points = [
            intersection(red_purple_line, blue_purple_line),
            get_bound(red_purple_line, -1, 1),
            get_bound(blue_purple_line, -1, 1),
        ]
        purple_plane = Polygon(
            *purple_points, **vector_plane_kwargs, color=colors[4])

        red_points5 = [
            *red_points4[:-1],
            *purple_points[0:2][::-1]
        ]
        red_plane5 = Polygon(
            *red_points5, **vector_plane_kwargs, color=colors[0])

        blue_points2 = [
            purple_points[2],
            purple_points[0],
            *blue_points[1:]
        ]
        blue_plane2 = Polygon(
            *blue_points2, **vector_plane_kwargs, color=colors[3])

        self.play(Write(VGroup(red_plane5, yellow_plane3,
                  green_plane2, blue_plane2, purple_plane)))
        self.wait(1)
        self.play(frame.set_phi, 0.35*PI)
        rotate = True
        self.wait(5)
        self.play(Uncreate(VGroup(red_plane5, yellow_plane3,
                  green_plane2, blue_plane2, purple_plane)))

        cp = s[0].copy()

        self.play(ShowCreation(cp))

        for i in range(1, 5):
            self.wait()
            self.play(Transform(cp, s[i]))

        self.wait()

        self.play(ShowCreation(s[:-1]))
        self.wait()

        rotate = False
        self.play(
            frame.set_phi, 0,
            frame.set_theta, -1 * TAU
        )
        self.wait()

        self.embed()

    def surface_func_max(self, i=6, scale=0.4, func=lambda x: max(x)):
        return lambda u, v: [u, v, scale * func((np.array([[u, v]]).dot(self.w.T) + self.b)[0][:i])]

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
