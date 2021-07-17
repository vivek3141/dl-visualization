from manimlib.imports import *
import pickle
import gzip

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


def load_data():
    f = gzip.open('../mnist/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def heaviside(x):
    return int(x >= 0)

# NeuralNetworkMobject is not my code, from 3b1b/manim
# number of layers change


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
                TextMobject(
                    f"Part {3-i}").shift([coors[0], coors[1]+1.5, 0]).scale(1.5)
            )

        for i in range(3):
            self.play(GrowFromCenter(s[2-i]))
            self.play(FadeInFromDown(t[2-i]))

        self.wait()

        s1 = ScreenRectangle(height=2, color=RED).shift([0, 0, 0])
        t1 = TextMobject(f"Part 2", color=RED).shift([0, 1.5, 0]).scale(1.5)

        self.play(Transform(t[1], t1), Transform(s[1], s1))
        self.wait()


# BG Color - #200f21

class LastVideo(Scene):
    def construct(self):
        title = TextMobject("Part 1")
        title.scale(1.5)
        title.to_edge(UP)
        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)
        self.play(
            FadeInFromDown(title),
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
        "edge_color": LIGHT_GREY,
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
            label = TexMobject(str(n ))
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