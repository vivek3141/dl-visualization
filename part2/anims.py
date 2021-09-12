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


def get_dots(func):
    return VGroup(
        *[
            Dot(func([point[0], point[1], 0]), color=colors[Y[index]],
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
        ]
    )


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
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "neuron_stroke_color": BLUE,
        "neuron_stroke_width": 6,
        "neuron_fill_color": GREEN,
        "edge_color": GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 16,
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
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
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
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
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
            label = TexMobject(f"x_{n + 1}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_y(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(r"\hat{y}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_weight_labels(self):
        weight_group = VGroup()

        for n, i in enumerate(self.layers[0].neurons):
            edge = self.get_edge(i, self.layers[-1][0])
            text = TexMobject(f"w_{n + 1}", color=RED)
            text.move_to(edge)
            weight_group.add(text)
        self.add(weight_group)

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(str(n))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_middle_a(self):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = TexMobject(f"h_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)


class NNDiagram(Scene):
    def construct(self):
        nn = NeuralNetworkMobject(
            [2, 100, 5],
            neuron_radius=0.15,
            neuron_to_neuron_buff=SMALL_BUFF,
            layer_to_layer_buff=1.5,
            neuron_stroke_color=RED,
            edge_stroke_width=1,
            include_output_labels=True,
            neuron_stroke_width=3
        )

        for i in range(2):
            self.play(Write(nn.layers[i]))
            self.bring_to_back(nn.edge_groups[i])
            self.play(Write(nn.edge_groups[i]))

        self.play(Write(nn.layers[-1]), Write(nn.output_labels))
        self.wait()

        nn2 = NeuralNetworkMobject(
            [2, 100, 2, 5],
            neuron_radius=0.15,
            neuron_to_neuron_buff=SMALL_BUFF,
            layer_to_layer_buff=1.5,
            neuron_stroke_color=RED,
            edge_stroke_width=1,
            include_output_labels=True,
            neuron_stroke_width=3
        )

        self.play(Transform(nn, nn2))
        self.wait()


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

        self.add(r, eq1, eq2, red_r, b)

        self.embed()
