from manimlib import *

import torch
import pickle
import gzip

colors = [RED, YELLOW, GREEN, BLUE, PURPLE]

AQUA = "#8dd3c7"
YELLOW = "#ffffb3"
LAVENDER = "#bebada"
RED = "#fb8072"
BLUE = "#80b1d3"
ORANGE = "#fdb462"
GREEN = "#b3de69"
PINK = "#fccde5"
GREY = "#d9d9d9"
VIOLET = "#bc80bd"
UNKA = "#ccebc5"
UNKB = "#ffed6f"

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


def load_data():
    f = gzip.open('../mnist/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def heaviside(x):
    return int(x >= 0)


def get_dots(func, color_func=lambda idx: colors[Y[idx]]):
    return VGroup(
        *[
            Dot(func([point[0], point[1], 0]), color=color_func(index),
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
        ]
    )


def get_isolate_rect(vertices, **kwargs):
    rect = VGroup(
        Polygon(
            [-FRAME_WIDTH/2, -FRAME_HEIGHT/2, 0],
            [FRAME_WIDTH/2, -FRAME_HEIGHT/2, 0],
            [FRAME_WIDTH/2, vertices[2][1], 0],
            [-FRAME_WIDTH/2, vertices[2][1], 0],
            **kwargs),
        Polygon(
            [FRAME_WIDTH/2, vertices[2][1], 0],
            vertices[3],
            vertices[0],
            [FRAME_WIDTH/2, vertices[0][1], 0],
            **kwargs),
        Polygon(
            [-FRAME_WIDTH/2, vertices[2][1], 0],
            vertices[2],
            vertices[1],
            [-FRAME_WIDTH/2, vertices[0][1], 0],
            **kwargs),
        Polygon(
            [-FRAME_WIDTH/2, FRAME_HEIGHT/2, 0],
            [FRAME_WIDTH/2, FRAME_HEIGHT/2, 0],
            [FRAME_WIDTH/2, vertices[0][1], 0],
            [-FRAME_WIDTH/2, vertices[0][1], 0],
            **kwargs),
    )

    return rect


"""
This is terrible practice, but the reason why I've redefined Scene is because Scene itself has a interact() method that breaks self.embed() after hitting Ctrl + C (KeyboardInterrupt). The better way would be to find a better fix than this and to make a pull request (or I could be using touch() incorrectly!), which I will probably do in the future :)
"""


class Scene(Scene):
    def interact(self):
        self.quit_interaction = False
        self.lock_static_mobject_data()
        try:
            while True:
                self.update_frame()
        except KeyboardInterrupt:
            self.unlock_mobject_data()


"""
Again, not the best practice but I didn't feel the need to have the same CONFIG in every class I create the plane
"""


class DotsScene(Scene):
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

# BG Color - #5b6190


class Intro(Scene):
    def construct(self):
        w_width = FRAME_WIDTH/4
        h_height = FRAME_HEIGHT/4
        buff = 0.5

        s = VGroup()
        t = VGroup()

        for i, coors in enumerate([[-i * (w_width + buff), i * (h_height - 0.25)] for i in range(-1, 2)]):
            s.add(
                ScreenRectangle(height=2).shift([*coors, 0])
            )
            t.add(
                TexText(
                    f"Part {3-i}", stroke_width=0.1).shift([coors[0], coors[1]+1.5, 0]).scale(1.5)
            )

        for i in range(3):
            self.play(GrowFromCenter(s[2-i]))
            self.play(FadeIn(t[2-i], shift=DOWN))
            FadeIn

        self.wait()

        s1 = ScreenRectangle(height=2, color=RED).shift([0, 0, 0])
        t1 = TexText(f"Part 2", color=RED).shift([0, 1.5, 0]).scale(1.5)

        self.play(Transform(t[1], t1), Transform(s[1], s1))
        self.wait()


# BG Color - #200f21

class LastVideo(Scene):
    def construct(self):
        title = TexText("Part 1")
        title.scale(1.5)
        title.to_edge(UP)
        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)
        self.play(
            FadeIn(title, shift=DOWN),
            Write(rect)
        )
        self.wait(2)


class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.25,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "neuron_stroke_color": BLUE,
        "neuron_stroke_width": 6,
        "neuron_fill_color": GREEN,
        "edge_color": GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 10,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }

    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        # self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()

    def get_nn_fill_color(self, index):
        if index == -1:
            return self.neuron_stroke_color
        if index == 0:
            return PINK
        elif index == len(self.layer_sizes) - 1:
            return BLUE
        else:
            return GREEN

    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=0.25,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexText("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    def add_input_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = TexText(f"x_{n + 1}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def get_x(self, layer=-2):
        self.x_lbl = VGroup()
        for n, neuron in enumerate(self.layers[layer].neurons):
            label = Tex(r"x_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.x_lbl.add(label)
        return self.x_lbl

    def get_y(self):
        self.y_lbl = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(r"\hat{y}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.y_lbl.add(label)
        return self.y_lbl

    def add_weight_labels(self):
        weight_group = VGroup()

        for n, i in enumerate(self.layers[0].neurons):
            edge = self.get_edge(i, self.layers[-1][0])
            text = Tex(f"w_{n + 1}", color=RED)
            text.move_to(edge)
            weight_group.add(text)
        self.add(weight_group)

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(str(n+1))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_middle_a(self):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = Tex(f"h_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)


class IntroPoints(DotsScene):
    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        points = get_dots(lambda point: point)

        eq = Tex(
            r"""X_{k}(t)=t\left(\begin{array}{c}
                \sin \left[\frac{2 \pi}{K}(2 t+k-1+\mathcal{N}(0, \sigma^2) )\right] \\
                \\ 
                \cos \left[\frac{2 \pi}{K}(2 t+k-1+\mathcal{N}(0, \sigma^2) )\right]
                \end{array}\right)"""
        )

        eq[0][16].set_color(AQUA)
        eq[0][41].set_color(AQUA)

        eq[0][21].set_color(LAVENDER)
        eq[0][46].set_color(LAVENDER)

        bg = BackgroundRectangle(eq, buff=0.5)
        eq = VGroup(bg, eq)
        eq.shift(2.5 * DOWN)

        def check(obj):
            self.remove(obj)
            self.wait(0.25)
            self.add(obj)

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(points))
        self.wait()

        self.play(Write(eq), run_time=4)
        self.wait()

        self.play(
            Indicate(eq[1][0][16], scale_factor=2),
            Indicate(eq[1][0][41], scale_factor=2)
        )
        self.wait()

        self.play(
            Indicate(eq[1][0][21], scale_factor=2),
            Indicate(eq[1][0][46], scale_factor=2)
        )
        self.wait()

        self.play(Uncreate(eq))
        self.wait()

        self.embed()


class ShowTrainingPoint(DotsScene):
    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)

        points = get_dots(lambda point: point)

        def check(obj):
            self.remove(obj)
            self.wait(0.25)
            self.add(obj)

        self.add(b_plane, f_plane, points)

        grp = VGroup(b_plane, f_plane, points)

        self.play(grp.scale, 4)
        self.play(grp.shift, 2 * LEFT)
        self.wait()

        b = BackgroundRectangle(
            points[37], stroke_opacity=1, fill_opacity=0, stroke_width=4, buff=0.05, color=WHITE)

        r1 = Rectangle(height=7, width=4, fill_opacity=0.75,
                       stroke_opacity=0, color=BLACK)
        r2 = Rectangle(height=7, width=4, fill_opacity=0,
                       stroke_opacity=1, color=WHITE)

        r = VGroup(r1, r2)
        r.shift(4.5 * RIGHT)

        eq1 = Tex(r"""
            X = \begin{bmatrix}
                0.374\\
                0.329
            \end{bmatrix}""",
                  tex_to_color_map={"X": PINK})
        eq1.scale(1.25)
        eq1.shift(4.5 * RIGHT + 1.5 * UP)

        eq2 = Tex(r"""y = 0""",
                  tex_to_color_map={"y": BLUE})
        eq2.scale(1.25)
        eq2.shift(4 * RIGHT + 1.5 * DOWN)

        red_r = Rectangle(
            height=0.5, width=0.5, fill_color=colors[0], stroke_color=WHITE, fill_opacity=1)
        red_r.shift(5.5 * RIGHT + 1.5 * DOWN)

        v1 = b.get_vertices()
        v2 = r1.get_vertices()

        l1 = Line(v1[0], v2[1])
        l2 = Line(v1[3], v2[2])

        sheet = ImageMobject("highlight_point.png", height=FRAME_HEIGHT)
        sheet.set_opacity(0.85)

        self.play(
            FadeIn(sheet),
            Write(b),
            Write(l1),
            Write(l2),
            Write(r),
            run_time=3
        )
        self.wait()

        self.play(Write(eq1))
        self.wait()

        self.play(Write(eq2), Write(red_r))
        self.wait()

        self.play(Uncreate(VGroup(b, l1, l2, r, eq1, eq2, red_r)), FadeOut(sheet))
        self.play(grp.shift, 2 * RIGHT)
        self.play(grp.scale, 0.25)
        self.wait()

        self.embed()


class IntroNNDiagram(DotsScene):
    CONFIG = {
        "nn_config": {
            "neuron_radius": 0.15,
            "neuron_to_neuron_buff": SMALL_BUFF,
            "layer_to_layer_buff": 1.5,
            "neuron_stroke_color": RED,
            "edge_stroke_width": 1,
            "include_output_labels": True,
            "neuron_stroke_width": 3
        }
    }

    def construct(self):
        def check(obj):
            self.remove(obj)
            self.wait(0.25)
            self.add(obj)

        nn = NeuralNetworkMobject(
            [2, 100, 5],
            **self.nn_config
        )

        nn2 = NeuralNetworkMobject(
            [2, 100, 2, 5],
            **self.nn_config
        )
        nn2.shift(RIGHT)

        for i in range(3):
            nn2.edge_groups[i].stretch(2/3, 0)

        nn2.edge_groups[0].move_to(-1.5 * RIGHT)
        nn2.layers[1].shift(LEFT)

        nn2.edge_groups[1].shift(1.25 * LEFT)
        nn2.layers[2].shift(1.5 * LEFT)

        nn2.edge_groups[2].shift(1.75 * LEFT)
        nn2.layers[3].shift(2 * LEFT)
        nn2.output_labels.shift(2 * LEFT)

        f_plane = NumberPlane((-4, 4), (-4, 4), **self.foreground_plane_kwargs)
        b_plane = NumberPlane((-4, 4), (-4, 4), **self.background_plane_kwargs)

        points = get_dots(lambda point: 4*np.array(point),
                          color_func=lambda _: GREY)

        inp_points = VGroup(b_plane, f_plane, points)
        inp_points.scale(0.4)
        inp_points.shift(5 * LEFT)

        f_plane = NumberPlane((-4, 4), (-4, 4), **self.foreground_plane_kwargs)
        b_plane = NumberPlane((-4, 4), (-4, 4), **self.background_plane_kwargs)

        points = get_dots(lambda point: 4*np.array(point))

        out_points = VGroup(b_plane, f_plane, points)
        out_points.scale(0.4)
        out_points.shift(5.2 * RIGHT)

        self.play(
            Write(inp_points),
            run_time=2
        )
        self.wait()

        self.play(
            TransformFromCopy(inp_points, nn.layers[0]),
            run_time=2
        )
        self.wait()

        self.play(Write(nn.layers[2]), Write(nn.output_labels))
        self.play(
            TransformFromCopy(nn.layers[2], out_points)
        )
        self.wait()

        self.bring_to_back(nn.edge_groups[0])
        self.play(Write(nn.edge_groups[0]))
        self.play(Write(nn.layers[1]))
        self.bring_to_back(nn.edge_groups[1])
        self.play(Write(nn.edge_groups[1]))
        self.wait()

        self.play(
            Transform(nn.edge_groups[0], nn2.edge_groups[0]),
            Transform(nn.layers[1], nn2.layers[1]),
            Transform(nn.edge_groups[1], nn2.edge_groups[2]),
            Write(nn2.edge_groups[1],
                  run_time=2)
        )
        self.wait()

        self.play(Write(nn2.layers[2]))
        self.wait()

        rect = Rectangle(height=FRAME_HEIGHT, width=FRAME_WIDTH,
                         color=BLACK, fill_opacity=0.75, stroke_width=0)

        grp = VGroup(inp_points, out_points, nn, nn2)

        self.play(ApplyMethod(grp.scale, 2.5))
        self.play(ApplyMethod(grp.move_to, [-4.5, 0, 0]))
        self.play(FadeIn(rect))
        self.play(FadeIn(VGroup(
            nn2.layers[2].copy(),
            nn2.edge_groups[2][0],
            nn2.edge_groups[2][5],
            nn2.output_labels[0],
            nn2.layers[-1][0][0]
        )))
        self.wait()

        eq = Tex(r"\hat{y}_1 = \sigma ( w_1 x + b_1 )",
                 tex_to_color_map={r"\sigma": AQUA, r"x": PINK, r"b_1": BLUE, r"\hat{y}_1": BLUE})
        eq.scale(2.5)
        eq = VGroup(BackgroundRectangle(eq, color=BLACK, buff=0.25), eq)
        eq.shift(2.5 * DOWN)

        self.play(Write(eq), Write(nn2.get_x(layer=-2)), Write(nn2.get_y()[0]))
        self.wait()
        self.embed()


class IntroDecisionScene(DotsScene):
    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)
        points = get_dots(lambda point: point)

        grp = VGroup(b_plane, f_plane, points)
        grp.scale(3)

        coors = np.array([0.75, -2.5, 0])

        d = Dot(coors)
        d_y = Dot(coors, color=colors[1])
        brect = BackgroundRectangle(
            d_y, buff=0.1, fill_opacity=0, color=WHITE,
            stroke_opacity=1, stroke_width=DEFAULT_STROKE_WIDTH
        )

        rect = Rectangle(
            height=FRAME_HEIGHT, width=FRAME_WIDTH, color=BLACK,
            fill_opacity=0.6, stroke_opacity=0, stroke_width=0
        )

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(points))
        self.wait()

        self.play(FadeIn(rect))
        self.wait(0.5)

        self.play(Write(d), Write(brect))
        self.wait()

        self.play(Transform(d, d_y), run_time=2)
        self.wait()

        image = ImageMobject("relu_inp_decisions.png", height=FRAME_HEIGHT)
        image.scale(3)
        image.set_opacity(0.65)

        self.play(Uncreate(d), Uncreate(brect))
        self.play(FadeOut(rect))
        self.wait()

        self.play(FadeIn(image), run_time=2)
        self.wait()

        self.embed()


class DemoActivation(DotsScene):
    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)
        points = get_dots(lambda point: point)

        grp = VGroup(b_plane, f_plane, points)
        grp.scale(3)

        image = ImageMobject("relu_inp_decisions.png", height=FRAME_HEIGHT)
        image.scale(3)
        image.set_opacity(0.65)

        l_points = [
            [[1, 1.5], [3, 3.5], [1, 3]],
            [[1, 0], [7, -3.5], [1.5, 6]],
            [[-0.4, -2], [-1, -3.5], [-0.3, -1]],
            [[-4, -2], [-3, -1.5], [-1.5, -6]],
            [[-0.625, 1], [-0.875, 3.5], [-0.65, -0.92]]
        ]

        lines = VGroup()

        for p1, p2, t in l_points:
            lines.add(
                self.get_line(p1, p2, t, color=ORANGE, stroke_width=5)
            )

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(points))
        self.play(FadeIn(image))
        self.wait()

        self.play(ApplyMethod(grp.scale, 1/3), ApplyMethod(image.scale, 1/3))
        self.bring_to_back(grp)
        self.wait()

        rect = Rectangle(width=FRAME_WIDTH, height=FRAME_HEIGHT,
                         color=BLACK, fill_opacity=0.5, stroke_width=0)

        self.play(FadeIn(rect))
        self.play(Write(lines))
        self.wait()

        self.play(Uncreate(lines), FadeOut(rect))
        self.wait()

        ext = ImageMobject("relu_ext_decisions.png", height=FRAME_HEIGHT)
        ext.set_opacity(0.65)

        self.play(FadeTransform(image, ext))
        self.wait()

        self.embed()

    def get_line(self, p1, p2, t, **kwargs):
        def func(t):
            slope = (p2[1]-p1[1])/(p2[0]-p1[0])
            return np.array([t, slope*(t-p1[0])+p1[1], 0])
        return ParametricCurve(func, t_range=t, **kwargs)


class DemoSin(DotsScene):
    def construct(self):
        f_plane = NumberPlane(**self.foreground_plane_kwargs)
        b_plane = NumberPlane(**self.background_plane_kwargs)
        points = get_dots(lambda point: point)

        grp = VGroup(b_plane, f_plane, points)
        grp.scale(3)

        image = ImageMobject("sin_inp_decisions.png", height=FRAME_HEIGHT)
        image.scale(3)
        image.set_opacity(0.65)

        self.play(Write(b_plane), Write(f_plane))
        self.play(Write(points))
        self.play(FadeIn(image))
        self.wait()

        self.play(ApplyMethod(grp.scale, 1/3), ApplyMethod(image.scale, 1/3))
        self.bring_to_back(grp)
        self.wait()
