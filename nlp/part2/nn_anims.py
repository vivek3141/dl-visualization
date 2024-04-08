from manimlib import *

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

# Old Manim Code Mappings
TexMobject = Tex
TextMobject = Text
LIGHT_GRAY = GREY
FadeInFromDown = lambda x: FadeIn(x, DOWN)


def heaviside(x):
    return 1 if x >= 0 else 0


# fmt: off
mnist_example = np.array(
    [0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0.01171875, 0.0703125, 0.0703125,
     0.0703125, 0.4921875, 0.53125, 0.68359375, 0.1015625,
     0.6484375, 0.99609375, 0.96484375, 0.49609375, 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.1171875, 0.140625, 0.3671875, 0.6015625,
     0.6640625, 0.98828125, 0.98828125, 0.98828125, 0.98828125,
     0.98828125, 0.87890625, 0.671875, 0.98828125, 0.9453125,
     0.76171875, 0.25, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.19140625, 0.9296875,
     0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125,
     0.98828125, 0.98828125, 0.98828125, 0.98046875, 0.36328125,
     0.3203125, 0.3203125, 0.21875, 0.15234375, 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.0703125, 0.85546875, 0.98828125, 0.98828125,
     0.98828125, 0.98828125, 0.98828125, 0.7734375, 0.7109375,
     0.96484375, 0.94140625, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0.3125, 0.609375, 0.41796875, 0.98828125, 0.98828125,
     0.80078125, 0.04296875, 0., 0.16796875, 0.6015625,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.0546875,
     0.00390625, 0.6015625, 0.98828125, 0.3515625, 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.54296875,
     0.98828125, 0.7421875, 0.0078125, 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0.04296875, 0.7421875, 0.98828125,
     0.2734375, 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.13671875, 0.94140625, 0.87890625, 0.625,
     0.421875, 0.00390625, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0.31640625, 0.9375, 0.98828125, 0.98828125, 0.46484375,
     0.09765625, 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.17578125,
     0.7265625, 0.98828125, 0.98828125, 0.5859375, 0.10546875,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.0625, 0.36328125,
     0.984375, 0.98828125, 0.73046875, 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.97265625, 0.98828125,
     0.97265625, 0.25, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.1796875, 0.5078125,
     0.71484375, 0.98828125, 0.98828125, 0.80859375, 0.0078125,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.15234375,
     0.578125, 0.89453125, 0.98828125, 0.98828125, 0.98828125,
     0.9765625, 0.7109375, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0.09375, 0.4453125, 0.86328125, 0.98828125, 0.98828125,
     0.98828125, 0.98828125, 0.78515625, 0.3046875, 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.08984375, 0.2578125, 0.83203125, 0.98828125,
     0.98828125, 0.98828125, 0.98828125, 0.7734375, 0.31640625,
     0.0078125, 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0.0703125, 0.66796875, 0.85546875,
     0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.76171875,
     0.3125, 0.03515625, 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.21484375, 0.671875,
     0.8828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125,
     0.953125, 0.51953125, 0.04296875, 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.53125, 0.98828125, 0.98828125, 0.98828125,
     0.828125, 0.52734375, 0.515625, 0.0625, 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0., 0., 0.], dtype=np.float32)
# fmt: on


class MNISTImage(VMobject):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for x, xpos in enumerate(np.arange(-3, 3, 6 / 28)):
            for y, ypos in enumerate(np.arange(-3, 3, 6 / 28)):
                self.add(
                    Rectangle(
                        height=6 / 28,
                        width=6 / 28,
                        stroke_width=1,
                        stroke_opacity=0.25,
                        fill_opacity=data[abs(y - 27) * 28 + x],
                    ).shift([xpos, ypos, 0])
                )

    def set_opacity(self, opacity):
        for rect in self:
            rect.set_fill(opacity=opacity * rect.get_fill_opacity())
            rect.set_stroke(opacity=opacity * rect.get_stroke_opacity())


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
        "neuron_fill_opacity": 1,
    }

    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    def add_neurons(self):
        layers = VGroup(
            *[
                self.get_layer(size, index)
                for index, size in enumerate(self.layer_sizes)
            ]
        )
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
        neurons = VGroup(
            *[
                Circle(
                    radius=self.neuron_radius,
                    stroke_color=self.get_nn_fill_color(index),
                    stroke_width=self.neuron_stroke_width,
                    fill_color=BLACK,
                    fill_opacity=self.neuron_fill_opacity,
                )
                for x in range(n_neurons)
            ]
        )
        neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[: len(neurons) // 2]).next_to(dots, UP, MED_SMALL_BUFF)
            VGroup(*neurons[len(neurons) // 2 :]).next_to(dots, DOWN, MED_SMALL_BUFF)
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
                tip_length=self.arrow_tip_size,
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
            label = TexMobject(r"\hat{y}_" + "{" + f"{n + 1}" + "}")
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
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width() * RIGHT)
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


class PerceptronMobject(NeuralNetworkMobject):
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[self.get_layer(size) for size in self.layer_sizes])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers[1])
        if self.include_output_labels:
            self.add_output_labels()


class PieceWiseTwo(VGroup):
    def __init__(self, cond1, cond2, cond3, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        eq1 = TexMobject(r"1").shift(1 * UP)
        eq2 = TexMobject(r"0").shift(0 * UP)

        eq1.align_to(eq2, LEFT)

        t1 = TexMobject(r"\text{if }" + cond1, tex_to_color_map={"{x}": PINK}).shift(
            1 * UP + 2 * RIGHT
        )
        t2 = TexMobject(r"\text{if }" + cond2, tex_to_color_map={"{x}": PINK}).shift(
            0 * UP + 2 * RIGHT
        )

        t1.align_to(t2, LEFT)

        e = VGroup(eq1, eq2)
        t = VGroup(t1, t2)
        et = VGroup(e, t)

        b = Brace(et, LEFT)
        bt = b.get_tex(r"H(" + cond3 + ") = ", tex_to_color_map={"H": AQUA, "x": PINK})
        eq = VGroup(et, b, bt)

        eq.center()
        self.add(eq)


class MNISTNN(Scene):
    def construct(self):
        mnist = np.load("../part1/data/mnist_data.npy")
        mnist_example = mnist[0].flatten()

        img = MNISTImage(mnist_example)
        img.scale(0.5)
        img.shift(4 * LEFT)

        nn = NeuralNetworkMobject(
            [784, 16, 16, 10],
            neuron_radius=0.15,
            neuron_to_neuron_buff=SMALL_BUFF,
            layer_to_layer_buff=1.5,
            neuron_stroke_color=RED,
            edge_stroke_width=1,
            include_output_labels=True,
            neuron_stroke_width=3,
        )

        nn.shift(2 * RIGHT)

        lbl1 = Text(r"28").shift(4 * LEFT + 2 * UP)
        lbl2 = Text(r"28").shift(6.25 * LEFT)

        self.play(Write(img), Write(lbl1), Write(lbl2))
        self.wait()

        self.play(TransformFromCopy(img, nn.layers[0]))

        for i in range(3):
            self.bring_to_back(nn.edge_groups[i])
            self.play(Write(nn.edge_groups[i]))
            if i + 1 < 4:
                self.play(Write(nn.layers[i + 1]))

        self.wait()
        self.embed()


class Heaviside(Scene):
    def construct(self):
        axes = Axes(
            x_range=(-3, 3),
            y_range=(0, 2),
            x_min=-3,
            x_max=3,
            y_min=0,
            y_max=2,
            axis_config={
                "include_tip": False,
                # "include_ticks": False,
            },
        )
        self.embed()
        x_axis, y_axis = axes.get_axes()
        x_axis.add_tick_marks_without_end()
        y_axis.add_tick_marks_without_end()

        tip_scale_factor = 0.15
        tip = VGroup(
            *[
                Line(
                    [3, 0, 0],
                    np.array([3, 0, 0])
                    + tip_scale_factor
                    * np.array([-np.sqrt(2) / 2, i * np.sqrt(2) / 2, 0]),
                    color=LIGHT_GRAY,
                )
                for i in [-1, 1]
            ],
            *[
                Line(
                    [0, 2, 0],
                    np.array([0, 2, 0])
                    + tip_scale_factor
                    * np.array([i * np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]),
                    color=LIGHT_GRAY,
                )
                for i in [-1, 1]
            ],
        )

        f = VGroup(
            FunctionGraph(lambda x: 0, x_min=-3, x_max=0),
            FunctionGraph(lambda x: 1, x_min=0, x_max=3),
        )

        func = VGroup(axes, f)
        func.center()
        func.scale(1.5)
        func.shift(DOWN)

        eq = PieceWiseTwo(r"{x} \geq 0", r"{x} < 0", "{x}")
        eq.shift(2.5 * UP)

        self.play(Write(func), Write(eq))
        self.wait()

        self.play(Uncreate(func), ApplyMethod(eq.shift, 2.5 * DOWN))

        eq2 = PieceWiseTwo(r"{x} -20 \geq 0", "{x} -20 < 0", "{x} -20")
        eq2.scale(1.5)

        self.play(Transform(eq, eq2))
        self.wait()


# activation function aqua, y hat should be blue x should be pink b should be blue
# fix trnasform


class PerceptronTwo(Scene):
    CONFIG = {"n_color": GREEN}

    def construct(self):
        q_width = FRAME_WIDTH / 4

        x = ValueTracker(0)

        perc = PerceptronMobject(
            [2, 1, 1],
            arrow=True,
            arrow_tip_size=0.1,
            neuron_radius=0.25,
            neuron_stroke_color=self.n_color,
        )
        perc.scale(1.5)
        perc.shift(q_width * LEFT)

        circ = Circle(
            fill_opacity=0.5, color=self.n_color, radius=0.25, stroke_opacity=0
        )
        circ_on = False

        def circ_updater(circle):
            new_circle = Circle(
                fill_opacity=(
                    0.5 * heaviside(x.get_value() - 20) if not circ_on else 0.5
                ),
                color=self.n_color,
                radius=0.25,
                stroke_opacity=0,
            )
            new_circle.scale(1.5)
            new_circle.shift(q_width * LEFT)
            circle.become(new_circle)

        circ.add_updater(circ_updater)

        y_disp = TexMobject("0")
        y_disp.shift(1 * LEFT)

        x_disp1 = TextMobject("Temp").scale(0.75)
        x_disp1.shift(6.25 * LEFT + 0.65 * UP)

        x_disp2 = TextMobject(r"Humidity").scale(0.75)
        x_disp2.shift(6.25 * LEFT + 0.65 * DOWN)

        xlbl = TextMobject(r"Temperature (°C)")
        xlbl.shift(3 * DOWN + q_width * RIGHT)

        ylbl = TextMobject(r"Humidity (\%)")
        ylbl.rotate(PI / 2)
        ylbl.shift(3.5 * LEFT + 0.5 * UP + q_width * RIGHT)

        n = 100

        points = []
        colors = []
        c1 = WHITE
        c2 = RED

        for _ in range(n):
            point = np.random.random(2) * 5.5 + 0.25
            points.append(point)
            colors.append(1 if -point[0] + 6 > point[1] else 0)

        axes = Axes(
            x_range=(0, 6),
            y_range=(0, 6),
            height=FRAME_HEIGHT - 2,
            width=FRAME_HEIGHT - 2,
            axis_config={"include_tip": False},
        )

        pointg = VGroup(
            *[
                Dot(axes.c2p(*points[i]), color=c1 if colors[i] else c2)
                for i in range(n)
            ]
        )

        line = axes.get_graph(lambda x: -x + 6, x_min=0, x_max=6, color=YELLOW)

        grp = VGroup(axes, pointg, line)
        grp.center()
        grp.shift(0.5 * UP + q_width * RIGHT)

        inp_title = TextMobject(r"Input Space")
        inp_title.scale(1.5)
        inp_title.shift(q_width * RIGHT + 3 * UP)

        self.play(
            Write(circ), Write(perc), Write(x_disp2), Write(x_disp1), Write(y_disp)
        )
        self.wait()

        self.play(Write(axes))
        self.play(Write(pointg))
        self.play(FadeInFromDown(xlbl), FadeInFromDown(ylbl))
        self.wait()

        self.play(ApplyMethod(pointg.set_opacity, 0.5), Write(line))
        self.wait()

        self.play(Uncreate(Group(grp, xlbl, ylbl)))

        temp_grp = VGroup(*[i for i in self.mobjects if isinstance(i, VMobject)])

        self.play(ApplyMethod(temp_grp.shift, -temp_grp.get_center() + 1 * UP))
        self.wait()

        eq = TexMobject(
            r"\hat{y} = H(",
            r"w_1 x_1 + w_2 x_2 + b",
            r")",
            tex_to_color_map={
                r"\hat{y}": BLUE,
                "x_1": PINK,
                "x_2": PINK,
                "b": BLUE,
                "H": AQUA,
            },
        )
        eq.scale(1.5)
        eq.shift(2 * DOWN)

        brect = BackgroundRectangle(
            eq[4:-1],
            buff=0.1,
            fill_opacity=0,
            stroke_opacity=1,
            color=PURPLE,
            stroke_width=4,
        )
        brect_label = TextMobject("Plane", color=PURPLE)
        brect_label.shift(1 * DOWN + 1 * RIGHT)

        self.play(Write(eq))
        self.wait()

        self.play(Write(brect))
        self.play(Write(brect_label))
        self.wait()

        self.play(Uncreate(temp_grp), Uncreate(Group(brect, brect_label)))
        self.play(eq.shift, 3.5 * UP)
        self.wait()

        eq2 = TexMobject(
            r"\hat{y} = H \left( ",
            r"\begin{bmatrix} w_1 \ w_2 \end{bmatrix} ",
            r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}",
            r" + {b} \right)",
            tex_to_color_map={
                r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}": PINK,
                "H": AQUA,
                "{b}": BLUE,
                r"\hat{y}": BLUE,
            },
        )

        eq2.scale(1.5)
        eq2.shift(1.5 * DOWN)

        m = TexMobject(r"\bm{W}")
        m.scale(1.5)
        m.shift(1.5 * DOWN + 0.5 * LEFT)

        xtex = TexMobject(r"\bm{x}", color=PINK)
        xtex.scale(1.5)
        xtex.shift(1.5 * DOWN + 1.5 * RIGHT)

        self.play(Write(eq2))
        self.wait()

        self.play(Transform(eq2[4], m))
        self.wait()

        self.play(Transform(eq2[5], xtex))
        self.wait()

        eq4 = TexMobject(
            r" \hat{y} = H( ",
            r"\bm{W}  \bm{x}",
            r" + {b} )",
            tex_to_color_map={
                r"\bm{x}": PINK,
                r"{b}": BLUE,
                r"H": AQUA,
                r"\hat{y}": BLUE,
            },
        )
        eq4.scale(1.3)
        eq4.shift(1.5 * DOWN)

        w_lbl = TextMobject("Weights", color=WHITE)
        w_lbl.shift(0.5 * DOWN + 0 * RIGHT)

        b_lbl = TextMobject("Bias", color=BLUE)
        b_lbl.shift(2.5 * DOWN + 2.5 * RIGHT)

        anims = []
        move_anims = [4, 5]

        for i in range(len(eq2)):
            if i in move_anims:
                self.remove(eq2[i][1:])
                anims.append(Transform(eq2[i][0], eq4[i]))
            else:
                anims.append(Transform(eq2[i], eq4[i]))

        self.embed()
        self.play(*anims)
        self.wait()

        self.play(Write(w_lbl))
        self.play(Write(b_lbl))
        self.wait()
