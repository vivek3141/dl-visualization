from manimlib import *
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
    f = gzip.open('mnist/mnist.pkl.gz', 'rb')
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

        tbc = TextMobject("To be continued...")
        tbc.scale(2)
        tbc.shift(3 * DOWN + 2.5 * LEFT)

        self.play(Write(tbc))
        self.wait()


# BG Color - #200f21


class Helpers(Scene):
    def construct(self):
        alf = ImageMobject("./img/alfredo.jpg")
        alf.scale(2)
        alf.shift(3 * LEFT + UP)

        yann = ImageMobject("./img/yann.jpg")
        yann.scale(2)
        yann.shift(3 * RIGHT + UP)

        t1 = TextMobject("Alfredo Canziani")
        t1.scale(1.5)
        t1.shift(3 * LEFT + 2.5 * DOWN)

        t2 = TextMobject("Yann LeCun")
        t2.scale(1.5)
        t2.shift(3 * RIGHT + 2.5 * DOWN)

        self.play(FadeInFromDown(alf))
        self.play(FadeInFromDown(t1))
        self.play(FadeInFromDown(yann))
        self.play(FadeInFromDown(t2))
        self.wait()


class MNISTImageMobject(VGroup):
    def __init__(self, data, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        for x, xpos in enumerate(np.arange(-3, 3, 6/28)):
            for y, ypos in enumerate(np.arange(-3, 3, 6/28)):
                self.add(
                    Rectangle(
                        height=6/28,
                        width=6/28,
                        stroke_width=1,
                        stroke_opacity=0.5,
                        fill_opacity=data[abs(y - 27) * 28 + x]
                    ).shift([xpos, ypos, 0])
                )


class MNISTIntro(Scene):
    def construct(self):
        data = load_data()[0]
        imgs = VGroup()

        ptr = 0
        for x in range(-4, 5, 4):
            for y in [-FRAME_HEIGHT/4, FRAME_HEIGHT/4]:
                imgs.add(
                    MNISTImageMobject(data[0][ptr]).shift([x, y, 0]).scale(0.5)
                )
                ptr += 1
        #obj = MNISTImageMobject(data[0][92])
        self.play(Write(imgs), run_time=6)
        self.wait()


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

# pink, green, blue


class MNISTNN(Scene):
    def construct(self):
        img = MNISTImageMobject(mnist_example)
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
            neuron_stroke_width=3
        )

        nn.shift(2 * RIGHT)

        lbl1 = TextMobject(r"28").shift(4 * LEFT + 2 * UP)
        lbl2 = TextMobject(r"28").shift(6.25 * LEFT)

        self.play(Write(img), Write(lbl1), Write(lbl2))
        self.wait()

        self.play(TransformFromCopy(img, nn.layers[0]))

        for i in range(3):
            self.bring_to_back(nn.edge_groups[i])
            self.play(Write(nn.edge_groups[i]))
            if i + 1 < 4:
                self.play(Write(nn.layers[i+1]))

        self.wait()


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


class PerceptronMobject(NeuralNetworkMobject):
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers[1])
        if self.include_output_labels:
            self.add_output_labels()


class PerceptronOne(Scene):
    CONFIG = {
        "n_color": GREEN
    }

    def construct(self):
        q_width = FRAME_WIDTH/4

        x = ValueTracker(0)

        perc = PerceptronMobject(
            [1, 1, 1], arrow=True, arrow_tip_size=0.1, 
            neuron_radius=0.25, neuron_stroke_color=self.n_color,
            neuron_fill_opacity=0)
        perc.scale(1.5)
        perc.shift(q_width * LEFT)

        circ = Circle(fill_opacity=0.5, color=self.n_color,
                      radius=0.25, stroke_opacity=0)
        circ_on = False

        def circ_updater(circle):
            new_circle = Circle(
                fill_opacity=0.5 *
                heaviside(x.get_value() - 20) if not circ_on else 0.5,
                color=self.n_color,
                radius=0.25,
                stroke_opacity=0
            )
            new_circle.scale(1.5)
            new_circle.shift(q_width * LEFT)
            circle.become(new_circle)

        circ.add_updater(circ_updater)

        l = NumberLine(x_min=0, x_max=30, numbers_with_elongated_ticks=[], unit_size=0.2, tick_frequency=5,
                       include_numbers=True, numbers_to_show=list(range(0, 31, 5)))
        l.shift((0.2 * -15 + q_width) * RIGHT)

        y_disp = TexMobject("0")

        def y_disp_updater(y_disp):
            new_disp = TexMobject(
                str(heaviside(x.get_value() - 20)) if not circ_on else "1"
            )
            new_disp.shift(1 * LEFT)
            y_disp.become(new_disp)

        y_disp.add_updater(y_disp_updater)

        x_disp = TexMobject("0")

        def x_disp_updater(x_disp):
            new_disp = TexMobject(
                str(int(x.get_value())) + r"^{\circ} \text{C}"
            )
            new_disp.shift(6.25 * LEFT)
            x_disp.become(new_disp)

        x_disp.add_updater(x_disp_updater)

        ptr = Triangle(fill_opacity=1)

        def ptr_updater(ptr):
            new_ptr = Triangle(fill_opacity=1)
            new_ptr.rotate(180 * DEGREES)
            new_ptr.scale(0.15)
            new_ptr.shift(
                [(x.get_value()) * 0.2 + (0.2 * -15 + q_width), -0.1, 0])
            ptr.become(new_ptr)

        ptr.add_updater(ptr_updater)

        inp_title = TextMobject(r"Input Space")
        inp_title.scale(1.5)
        inp_title.shift(q_width * RIGHT + 3 * UP)

        self.play(Write(circ), Write(perc), Write(x_disp), Write(y_disp))
        self.wait()

        self.play(Write(l), Write(ptr), Write(inp_title))
        self.wait()

        self.play(x.increment_value, 30, rate_func=linear, run_time=4)
        self.play(x.increment_value, -30, rate_func=linear, run_time=4)
        self.wait()

        line = DashedLine(2 * UP, 2 * DOWN, stroke_width=4)
        line.shift((20 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        active_lbl = TextMobject("Active", color=RED)
        active_lbl.add_background_rectangle()
        active_lbl.shift((25 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        inactive_lbl = TextMobject("Inactive")
        inactive_lbl.add_background_rectangle()
        inactive_lbl.shift((10 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        rect = Rectangle(height=4, width=2, color=RED, fill_opacity=0.3)
        rect.shift((25 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        self.play(Write(rect), Write(active_lbl), Write(inactive_lbl))
        self.play(x.increment_value, 30, rate_func=linear, run_time=4)
        self.play(x.increment_value, -30, rate_func=linear, run_time=4)
        self.wait()

        eq = TexMobject(r"\hat{y} = H(", r"{x}", r" -", r" 20 ", r")",
                        tex_to_color_map={r"H": AQUA, r"\hat{y}": BLUE, r"{x}": PINK})
        eq.scale(1.75)
        eq.shift(2.75 * DOWN)

        self.play(Write(eq))
        self.wait()

        rect2 = Rectangle(height=4, width=4, color=RED, fill_opacity=0.3)
        rect2.shift((10 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        eq2 = TexMobject(r"-", color=RED)
        eq2.scale(1.75)
        eq2.shift(2.75 * DOWN + 0.4 * LEFT)

        eq22 = TexMobject(r"+", color=RED)
        eq22.scale(1.75)
        eq22.shift(2.75 * DOWN + 1 * RIGHT)

        circ_on = True

        self.play(
            Transform(rect, rect2),
            ApplyMethod(active_lbl.shift, 0.2 * 15 * LEFT),
            ApplyMethod(inactive_lbl.shift, 0.2 * 15 * RIGHT),
            ApplyMethod(circ.set_opacity, 1)
        )
        self.play(ApplyMethod(eq[:4].shift, 0.75 * LEFT), FadeInFromDown(eq2))
        self.play(FadeOut(eq[-3]), FadeInFromDown(eq22))
        self.wait()

        list_of_ok = [perc, eq2, eq[:3], eq[3:], eq22]

        grp = Group(rect, rect2, inactive_lbl, active_lbl, line,
                    l, circ, x_disp, y_disp, inp_title, ptr)

        self.play(Uncreate(grp))

        to_move = False

        eq3 = TexMobject(r"\hat{y} = H( w ", r"x", r" + b )",
                         tex_to_color_map={r"H": AQUA, r"x": PINK, r"\hat{y}": BLUE, r"b": BLUE})
        eq3.scale(1.75)
        eq3.shift(2.75 * DOWN + 0.7 * LEFT)

        x_inp = TexMobject("x", color=PINK)
        x_inp.scale(1.5)
        x_inp.shift(2.5 * LEFT)

        y_out = TexMobject(r"\hat{y}", color=BLUE)
        y_out.scale(1.5)
        y_out.shift(2.5 * RIGHT)

        self.play(perc.shift, q_width * RIGHT)
        self.play(Write(x_inp), Write(y_out))
        self.play(FadeOut(eq2), FadeOut(eq22), FadeOut(eq[-3:]))
        self.play(FadeInFromDown(eq3[3]), FadeInFromDown(eq3[-3:]))
        self.wait()


class PieceWiseTwo(VGroup):
    def __init__(self, cond1, cond2, cond3, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        eq1 = TexMobject(r"1").shift(1 * UP)
        eq2 = TexMobject(r"0").shift(0 * UP)

        eq1.align_to(eq2, LEFT)

        t1 = TexMobject(
            r"\text{if }" + cond1, tex_to_color_map={"{x}": PINK}).shift(1 * UP + 2 * RIGHT)
        t2 = TexMobject(
            r"\text{if }" + cond2,  tex_to_color_map={"{x}": PINK}).shift(0 * UP + 2 * RIGHT)

        t1.align_to(t2, LEFT)

        e = VGroup(eq1, eq2)
        t = VGroup(t1, t2)
        et = VGroup(e, t)

        b = Brace(et, LEFT)
        bt = b.get_tex(r"H("+cond3+") = ",
                       tex_to_color_map={"H": AQUA, "x": PINK})
        eq = VGroup(et, b, bt)

        eq.center()
        self.add(eq)


class Heaviside(Scene):
    def construct(self):
        axes = Axes(
            x_min=-3,
            x_max=3,
            y_min=0,
            y_max=2,
            axis_config={
                "include_tip": False,
                "include_ticks": False,
            }
        )
        x_axis, y_axis = axes.get_axes()
        x_axis.add_tick_marks_without_end()
        y_axis.add_tick_marks_without_end()

        tip_scale_factor = 0.15
        tip = VGroup(
            *[Line(
                [3, 0, 0],
                np.array([3, 0, 0]) + tip_scale_factor *
                np.array([- np.sqrt(2)/2, i * np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
            *[Line(
                [0, 2, 0],
                np.array([0, 2, 0]) + tip_scale_factor *
                np.array([i * np.sqrt(2)/2,  - np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
        )
        axes
        f = VGroup(FunctionGraph(lambda x: 0, x_min=-3, x_max=0),
                   FunctionGraph(lambda x: 1, x_min=0, x_max=3))

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
    CONFIG = {
        "n_color": GREEN
    }

    def construct(self):
        q_width = FRAME_WIDTH/4

        x = ValueTracker(0)

        perc = PerceptronMobject(
            [2, 1, 1], arrow=True, arrow_tip_size=0.1, neuron_radius=0.25, neuron_stroke_color=self.n_color)
        perc.scale(1.5)
        perc.shift(q_width * LEFT)

        circ = Circle(fill_opacity=0.5, color=self.n_color,
                      radius=0.25, stroke_opacity=0)
        circ_on = False

        def circ_updater(circle):
            new_circle = Circle(
                fill_opacity=0.5 *
                heaviside(x.get_value() - 20) if not circ_on else 0.5,
                color=self.n_color,
                radius=0.25,
                stroke_opacity=0
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
        ylbl.rotate(PI/2)
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

        pointg = VGroup(
            *[Dot([points[i][0], points[i][1], 0], color=c1 if colors[i] else c2) for i in range(n)]
        )
        axes = Axes(
            x_min=0,
            x_max=6,
            y_min=0,
            y_max=6,
            axis_config={
                "include_tip": False
            }
        )

        line = FunctionGraph(lambda x: -x+6, x_min=0, x_max=6)
        grp = VGroup(axes, pointg, line)
        grp.center()
        grp.shift(0.5 * UP + q_width * RIGHT)

        inp_title = TextMobject(r"Input Space")
        inp_title.scale(1.5)
        inp_title.shift(q_width * RIGHT + 3 * UP)

        self.play(Write(circ), Write(perc), Write(
            x_disp2), Write(x_disp1), Write(y_disp))
        self.wait()

        self.play(Write(axes))
        self.play(Write(pointg))
        self.play(FadeInFromDown(xlbl), FadeInFromDown(ylbl))
        self.wait()

        self.play(ApplyMethod(pointg.set_opacity, 0.5), Write(line))
        self.wait()

        self.play(Uncreate(Group(grp, xlbl, ylbl)))

        temp_grp = VGroup(*self.mobjects)

        self.play(ApplyMethod(temp_grp.shift, -temp_grp.get_center() + 1 * UP))
        self.wait()

        eq = TexMobject(r"\hat{y} = H(", r"w_1 x_1 + w_2 x_2 + b", r")",
                        tex_to_color_map={r"\hat{y}": BLUE, "x_1": PINK, "x_2": PINK, "b": BLUE, "H": AQUA})
        eq.scale(1.5)
        eq.shift(2 * DOWN)

        brect = BackgroundRectangle(
            eq[4:-1], buff=0.1, fill_opacity=0, stroke_opacity=1, color=PURPLE, stroke_width=4)
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
            r"\hat{y} = H \left( ", r"\begin{bmatrix} w_1 \ w_2 \end{bmatrix} ",
            r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}", r" + {b} \right)",
            tex_to_color_map={r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}": PINK, "H": AQUA, "{b}": BLUE, r"\hat{y}": BLUE})
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
            r" \hat{y} = H( ", r"\bm{W}  \bm{x}", r" + {b} )",
            tex_to_color_map={r"\bm{x}": PINK, r"{b}": BLUE, r"H": AQUA, r"\hat{y}": BLUE})
        eq4.scale(1.3)
        eq4.shift(1.5 * DOWN)

        w_lbl = TextMobject("Weights", color=WHITE)
        w_lbl.shift(0.5 * DOWN + 0 * RIGHT)

        b_lbl = TextMobject("Bias", color=BLUE)
        b_lbl.shift(2.5 * DOWN + 2.5 * RIGHT)

        temp_grp2 = VGroup(eq2, m, xtex)

        self.play(*[Transform(eq2[i], eq4[i]) for i in range(0, len(eq2))])
        self.wait()

        self.play(Write(w_lbl))
        self.play(Write(b_lbl))
        self.wait()


class PerceptronThree(Scene):
    def construct(self):
        perc = PerceptronMobject(
            [3, 1, 1], neuron_stroke_color=GREEN, arrow=True)
        perc.scale(2.5)
        perc.shift(1.5 * UP)

        eq = TexMobject(
            r"\hat{y} = H \Bigg( ", r"\begin{bmatrix} w_1 \ w_2 \ w_3 \end{bmatrix} ",
            r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}", r"+ {b} \Bigg)",
            tex_to_color_map={
                r"\hat{y}": BLUE,
                r"H": AQUA,
                r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}": PINK,
                r"{b}": BLUE}
        )

        eq.scale(1.25)
        eq.shift(2 * DOWN)

        m = TexMobject(r"\bm{W}")
        m.scale(1.5)
        m.shift(2 * DOWN + 0.5 * LEFT)

        xtex = TexMobject(r"\bm{x}", color=PINK)
        xtex.scale(1.5)
        xtex.shift(2 * DOWN + 1.5 * RIGHT)

        x_disp1 = TextMobject("Temp").scale(0.75)
        x_disp1.shift(4 * LEFT + 2.75 * UP)

        x_disp2 = TextMobject(r"Humidity").scale(0.75)
        x_disp2.shift(4 * LEFT + 1.5 * UP)

        x_disp3 = TextMobject("Wind Speed").scale(0.75)
        x_disp3.shift(4 * LEFT + 0.25 * UP)

        eq2 = TexMobject(
            r"\bm{\hat{y}} = H( ", r"\bm{W}  \bm{x}", r" + \bm{b} )",
            tex_to_color_map={r"\bm{x}": PINK, r"\bm{b}": BLUE, r"H": AQUA, r"\bm{\hat{y}}": BLUE})
        eq2.scale(2)
        eq2.shift(2 * DOWN)

        self.play(Write(perc))
        self.play(Write(x_disp1), Write(x_disp2), Write(x_disp3))
        self.wait()

        self.play(Write(eq))
        self.wait()

        self.play(Transform(eq[4], m))
        self.wait()

        self.play(Transform(eq[5], xtex))
        self.wait()

        self.play(*[Transform(eq[i], eq2[i]) for i in range(0, len(eq2))])
        self.wait()


class SigmoidIntro(Scene):
    def construct(self):
        NumberLine
        axes = Axes(
            x_min=-3.5,
            x_max=3.5,
            y_min=0,
            y_max=2.5,
            axis_config={
                "include_tip": False,
                "include_ticks": True,
            },
            x_axis_config={
                "tick_frequency": 1.5,
                "decimal_number_config": {
                    "num_decimal_places": 1,
                }
            }
        )

        x_axis, y_axis = axes.get_axes()
        x_axis.add_tick_marks_without_end()
        y_axis.add_tick_marks_without_end()

        tip_scale_factor = 0.15
        tip = VGroup(
            *[Line(
                [3.5, 0, 0],
                np.array([3.5, 0, 0]) + tip_scale_factor *
                np.array([- np.sqrt(2)/2, i * np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
            *[Line(
                [0, 2.5, 0],
                np.array([0, 2.5, 0]) + tip_scale_factor *
                np.array([i * np.sqrt(2)/2,  - np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
        )

        f = VGroup(FunctionGraph(lambda x: 0, x_min=-3, x_max=0),
                   FunctionGraph(lambda x: 2, x_min=0, x_max=3))
        scale = 2
        f2 = VGroup(FunctionGraph(
            lambda x: scale/(1+np.exp(-scale*x)),
            x_min=-2.5, x_max=0, color=GOLD),
            FunctionGraph(
            lambda x: scale/(1+np.exp(-scale*x)),
            x_min=0, x_max=2.5, color=GOLD))
        axes.add(tip)

        func = VGroup(axes, f, f2)
        func.center()

        f2.stretch_to_fit_width(6)
        grp = VGroup()
        grp.add(
            axes.x_axis.get_number_mobject(-2.5).move_to([-1.65, -3, 0]),
            axes.x_axis.get_number_mobject(2.5).move_to([1.5, -3, 0]),
            axes.x_axis.get_number_mobject(-5).move_to([-3.15, -3, 0]),
            axes.x_axis.get_number_mobject(5).move_to([3, -3, 0]),
        )
        grp.scale(1.5)

        xy = VGroup(
            TexMobject("x", color=PINK).shift(2.7 * DOWN + 5.75 * RIGHT),
            TexMobject(r"\hat{y}", color=BLUE).shift(0.55 * RIGHT + 1.3 * UP)
        )

        func.scale(1.5)
        func.shift(0.7 * DOWN)

        eq = PieceWiseTwo(r"{x} \geq 0", r"{x} < 0", "{x}")
        eq.shift(2.5 * UP)

        lbls = VGroup(
            TexMobject("0.5").shift(DOWN),
            TexMobject("1.0").shift(0.5 * UP)
        ).shift(0.75 * LEFT + 0.1 * UP)

        self.play(Write(axes), Write(lbls), Write(grp), Write(xy))
        self.play(Write(f))
        self.play(Write(eq))
        self.wait()

        eq2 = TexMobject(
            r"\sigma ({x}) = {{1} \over {1 + \text{exp} (-{x})}",
            tex_to_color_map={r"\sigma": AQUA, r"{x}": PINK})
        eq2.scale(1.5)
        eq2.shift(2.75 * UP)

        self.play(Transform(f[0], f2[0]), Transform(f[1], f2[1]))
        self.wait()

        self.play(Transform(eq, eq2))
        self.wait()


class ReluIntro(Scene):
    def construct(self):
        axes = Axes(
            x_min=-3,
            x_max=3,
            y_min=-1,
            y_max=3,
            axis_config={
                "include_tip": False,
                "include_ticks": False
            }
        )

        x_axis, y_axis = axes.get_axes()
        x_axis.add_tick_marks_without_end()
        y_axis.add_tick_marks_without_end()

        tip_scale_factor = 0.15
        tip = VGroup(
            *[Line(
                [3, 0, 0],
                np.array([3, 0, 0]) + tip_scale_factor *
                np.array([- np.sqrt(2)/2, i * np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
            *[Line(
                [0, 3, 0],
                np.array([0, 3, 0]) + tip_scale_factor *
                np.array([i * np.sqrt(2)/2,  - np.sqrt(2)/2, 0]),
                color=LIGHT_GRAY) for i in [-1, 1]],
        )
        axes.add(tip)

        f = FunctionGraph(lambda x: max(0, x), x_min=-3, x_max=3)

        grp = VGroup(axes, f)
        grp.shift(1.5 * DOWN)

        eq = TexMobject(
            r"\texttt{ReLU}(x) = \max(0, {x}) = ({x})^+",
            tex_to_color_map={r"\texttt{ReLU}": AQUA, r"{x}": PINK}
        )
        eq.scale(1.25)
        eq.shift(3 * UP)

        t1 = TexMobject(r"y=0", color=YELLOW)
        t1.shift(1.5 * LEFT + 1 * DOWN)

        t2 = TexMobject(r"y=x", color=YELLOW)
        t2.rotate(45 * DEGREES)
        t2.shift((np.sqrt(3) - 1.5) * UP + 1 * RIGHT)

        self.play(Write(axes))
        self.play(Write(f))
        self.wait()

        self.play(Write(t1))
        self.play(Write(t2))
        self.wait()

        self.play(Write(eq))
        self.wait()


class LinearlyS(Scene):
    def construct(self):
        n = 100
        points = []
        colors = []
        c1 = "#99EDCC"
        c2 = "#B85C8C"

        for _ in range(n):
            point = np.random.random(2) * 5.5 + 0.25
            points.append(point)
            colors.append(1 if point[0] > point[1] else 0)

        pointg = VGroup(
            *[Dot([points[i][0], points[i][1], 0], color=c1 if colors[i] else c2) for i in range(len(colors))]
        )
        axes = Axes(
            x_min=0,
            x_max=6,
            y_min=0,
            y_max=6,
            axis_config={
                "include_tip": False
            }
        )
        line = FunctionGraph(lambda x: x, x_min=0, x_max=6)
        grp = VGroup(axes, pointg, line)
        grp.center()

        self.play(Write(axes))
        self.play(Write(pointg))
        self.wait()

        self.play(ApplyMethod(pointg.set_opacity, 0.5), Write(line))
        self.wait()

        title = TextMobject("Linearly Separable")
        title.scale(2.5)
        title.add_background_rectangle()

        self.play(Write(title))
        self.wait()


class HardDataset(Scene):
    def construct(self):
        n = 300
        points = []
        colors = []
        c1 = "#99EDCC"
        c2 = "#B85C8C"

        def func(x, y):
            return 0.75 * x ** 2 + 1.25 * y ** 2 < 9

        for _ in range(n):
            x = np.random.random() * FRAME_WIDTH - FRAME_WIDTH/2
            y = np.random.random() * FRAME_HEIGHT - FRAME_HEIGHT/2
            points.append([x, y])
            colors.append(1 if func(x, y) else 0)

        pointg = VGroup(
            *[Dot([points[i][0], points[i][1], 0], color=c1 if colors[i] else c2) for i in range(len(colors))]
        )
        pointg.center()

        self.play(Write(pointg))
        self.wait()


class NeuralNetwork(Scene):
    def construct(self):
        n = NeuralNetworkMobject([3, 4, 3])
        n.scale(3)

        self.play(Write(n))
        self.wait()


class NN22(Scene):
    def construct(self):
        n = NeuralNetworkMobject([2, 2])
        n.scale(3)
        n.shift(1.5 * UP)

        n.add_input_labels()
        n.add_y()

        eq1 = TexMobject(
            r"\hat{y}_1 = ", r"\sigma (", r"w_{11} x_1 + w_{12} x_2 + b_1", r")",
            tex_to_color_map={
                r"\sigma": AQUA,
                r"x_1": PINK, r"x_2": PINK, r"b_1": BLUE, r"\hat{y}_1": BLUE}
        )
        eq1.scale(1.5)
        eq1.shift(1 * DOWN)

        eq2 = TexMobject(
            r"\hat{y}_2 = ", r"\sigma (", r"w_{21} x_1 + w_{22} x_2 + b_2", r")",
            tex_to_color_map={
                r"\sigma": AQUA,
                r"x_1": PINK, r"x_2": PINK, r"b_2": BLUE, r"\hat{y}_2": BLUE}
        )
        eq2.scale(1.5)
        eq2.shift(2.5 * DOWN)

        eq_c = TexMobject(
            r"\begin{bmatrix} w_{11} \ w_{12} \\ w_{21} \ w_{22} \end{bmatrix}",
            r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} {b_1} \\ {b_2} \end{bmatrix}= ",
            r"\begin{bmatrix} {w_{11}} x_1 + {w_{12}} x_2 + b_1 \\ {w_{21}} x_1 + {w_{22}} x_2 + b_2 \end{bmatrix}",

        )
        eq_c.scale(1.25)
        eq_c.shift(2 * DOWN)

        eq = TexMobject(
            r"\begin{bmatrix} w_{11} \ w_{12} \\ w_{21} \ w_{22} \end{bmatrix}",
            r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}", r" + ",
            r"\begin{bmatrix} {b_1} \\ {b_2} \end{bmatrix}",
            r"= ",
            r"\Bigg[ " + "\ " * 26 + r" \Bigg]",

        )
        eq[1].set_color(PINK)
        eq[3].set_color(BLUE)
        eq.scale(1.25)
        eq.shift(2 * DOWN)

        #rect = Rectangle(width=5.5, height=1.35, fill_opacity=1, color=BLACK)
        #rect.shift(2 * DOWN + 3.5 * RIGHT)
        # eq.add(rect)

        eqw1 = TexMobject(
            r"w_{11} x_1 + w_{12} x_2 + b_1",
            tex_to_color_map={
                r"\sigma": AQUA,
                r"x_1": PINK, r"x_2": PINK, r"b_1": BLUE})
        eqw1.scale(1.25)
        eqw1.move_to(eq_c[-1])
        eqw1.shift(0.39 * UP)

        eqw2 = TexMobject(
            r"w_{21} x_1 + w_{22} x_2 + b_2",
            tex_to_color_map={
                r"\sigma": AQUA,
                r"x_1": PINK, r"x_2": PINK, r"b_2": BLUE})
        eqw2.scale(1.25)
        eqw2.move_to(eq_c[-1])
        eqw2.shift(0.36 * DOWN)

        self.play(Write(n))
        self.wait()

        # self.add(eq) #, eqw1, eqw2

        self.play(Write(eq1))
        self.wait()

        self.play(Write(eq2))
        self.wait()

        self.play(FadeOut(eq1[:4]), FadeOut(eq1[-1:]),
                  FadeOut(eq2[:4]), FadeOut(eq2[-1:]))
        self.play(Transform(eq1[4:-1], eqw1), Transform(eq2[4:-1], eqw2))
        self.wait()

        self.play(Write(eq[-1]))
        self.wait()

        self.play(Write(eq[2:5]))
        self.play(Write(eq[1]))
        self.wait()

        self.play(Write(eq[0]))
        self.wait()

        n2 = NeuralNetworkMobject([2, 2, 2])
        n2.scale(3)
        n2.shift(1.5 * UP)

        n2.add_input_labels()
        n2.add_y()
        n2.add_middle_a()

        self.play(Transform(n, n2))
        self.play(
            FadeOut(VGroup(*[i for i in self.mobjects if i not in [n, n2]])))
        self.wait()

        eq1 = TexMobject(
            r"{{\bm{h}}} = ( \bm{W}_{\bm{h}} \bm{x} + \bm{b}_{\bm{h}} )^+",
            tex_to_color_map={
                r"^+": AQUA,
                r"\bm{x}": PINK, r"\bm{b}_{\bm{h}}": GREEN, r"{{\bm{h}}}": GREEN}
        )
        eq1.scale(1.5)
        eq1.shift(1 * DOWN)

        eq2 = TexMobject(
            r"\bm{\hat{y}} = \sigma( \bm{W}_{\bm{y}} \bm{h} + \bm{b}_{\bm{y}})",
            tex_to_color_map={
                r"\sigma": AQUA,
                r"\bm{h}": GREEN, r"\bm{b}_{\bm{y}}": BLUE, r"\bm{\hat{y}}": BLUE}
        )
        eq2.scale(1.5)
        eq2.shift(2.5 * DOWN)

        self.play(Write(eq1))
        self.wait()

        self.play(Write(eq2))
        self.wait()


class LinTDemo(LinearTransformationScene):
    def construct(self):
        self.setup()

        matrix = np.array([[1, 2], [2, 1]]).transpose()
        matrix_mob = Matrix(matrix)
        matrix_mob.to_corner(UP+LEFT)
        matrix_mob.add_background_to_entries()
        matrix_mob.add_background_rectangle()

        col1 = Matrix(matrix[:, 0])
        col1.set_color(Y_COLOR)
        col1.add_background_rectangle()
        col1.shift([1, 3, 0])

        col2 = Matrix(matrix[:, 1])
        col2.set_color(X_COLOR)
        col2.add_background_rectangle()
        col2.shift([3, 1, 0])

        transform_matrix1 = np.array(matrix)
        transform_matrix1[:, 1] = [0, 1]
        transform_matrix2 = np.dot(
            matrix,
            np.linalg.inv(transform_matrix1)
        )

        self.play(Write(matrix_mob))
        self.wait()

        self.apply_matrix(
            transform_matrix1,
            added_anims=[TransformFromCopy(matrix_mob, col1)])
        self.wait()

        self.apply_matrix(
            transform_matrix2,
            added_anims=[TransformFromCopy(matrix_mob, col2)])
        self.wait()


class NN232(Scene):
    def construct(self):
        n = NeuralNetworkMobject([2, 3, 2], edge_stroke_width=8)
        n.scale(3.5)
        n.add_input_labels()
        n.add_middle_a()
        n.add_y()

        self.play(Write(n))
        self.wait()
