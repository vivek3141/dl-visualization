from manimlib.imports import *


def heaviside(x):
    return int(x >= 0)

# NeuralNetworkMobject is not my code, from 3b1b/manim


class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "neuron_stroke_color": BLUE,
        "neuron_stroke_width": 3,
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
        "arrow_tip_size": 0.1
    }

    def __init__(self, neural_network, size=0.15, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.neuron_radius = size
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()

    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.neuron_stroke_color,
                stroke_width=self.neuron_stroke_width,
                fill_color=self.neuron_fill_color,
                fill_opacity=0,
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
            label = TexMobject("y")
            label.set_height(0.3 * neuron.get_height())
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


class PerceptronMobject(NeuralNetworkMobject):
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


def heaviside(x):
    return int(x >= 0)


class PerceptronOne(Scene):
    CONFIG = {
        "n_color": RED
    }

    def construct(self):
        q_width = FRAME_WIDTH/4

        x = ValueTracker(0)

        perc = PerceptronMobject(
            [1, 1, 1], arrow=True, arrow_tip_size=0.1, size=0.25, neuron_stroke_color=self.n_color)
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
        # l.center()
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

        eq = TexMobject(r"\hat{y} = H( x - 20 )",
                        tex_to_color_map={r"H": YELLOW})
        eq.scale(1.75)
        eq.shift(2.5 * DOWN)

        self.play(Write(eq))
        self.wait()

        rect2 = Rectangle(height=4, width=4, color=RED, fill_opacity=0.3)
        rect2.shift((10 * 0.2 + (0.2 * -15 + q_width)) * RIGHT)

        eq2 = TexMobject(r"\hat{y} = H( -x + 20 )",
                         tex_to_color_map={r"H": YELLOW, r"-": RED})
        eq2.scale(1.75)
        eq2.shift(2.5 * DOWN)

        circ_on = True

        self.play(Transform(rect, rect2), ApplyMethod(active_lbl.shift, 0.2 * 15 * LEFT), ApplyMethod(inactive_lbl.shift, 0.2 * 15 * RIGHT),
                  ApplyMethod(circ.set_opacity, 1))
        self.play(Transform(eq, eq2))
        self.wait()

        list_of_ok = [perc, eq, eq2]

        grp = Group(*[i for i in self.mobjects if i not in list_of_ok])
        
        self.play(Uncreate(grp))
        
        to_move = False

        eq3 = TexMobject(r"\hat{y} = H( mx + b )",
                        tex_to_color_map={r"H": YELLOW, "m": RED, "b": TEAL})
        eq3.scale(1.75)
        eq3.shift(2.5 * DOWN)
        
        x_inp = TexMobject("x")
        x_inp.scale(1.5)
        x_inp.shift(2.5 * LEFT)

        y_out = TexMobject(r"\hat{y}")
        y_out.scale(1.5)
        y_out.shift(2.5 * RIGHT)

        self.play(perc.shift, q_width * RIGHT)
        self.play(Write(x_inp), Write(y_out))
        self.play(Transform(eq, eq3))
        self.wait()


class PieceWiseTwo(VGroup):
    def __init__(self, cond1, cond2, cond3, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        eq1 = TexMobject(r"1").shift(1 * UP)
        eq2 = TexMobject(r"0").shift(0 * UP)

        eq1.align_to(eq2, LEFT)

        t1 = TexMobject(r"\text{if }" + cond1).shift(1 * UP + 2 * RIGHT)
        t2 = TexMobject(r"\text{if }" + cond2).shift(0 * UP + 2 * RIGHT)

        t1.align_to(t2, LEFT)

        e = VGroup(eq1, eq2)
        t = VGroup(t1, t2)
        et = VGroup(e, t)

        b = Brace(et, LEFT)
        bt = b.get_tex(r"H("+cond3+") = ", tex_to_color_map={"H": YELLOW})
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
                "include_tip": False
            }
        )
        f = VGroup(FunctionGraph(lambda x: 0, x_min=-3, x_max=0),
                   FunctionGraph(lambda x: 1, x_min=0, x_max=3))

        func = VGroup(axes, f)
        func.center()
        func.scale(1.5)
        func.shift(DOWN)

        eq = PieceWiseTwo(r"x \geq 0", r"x < 0", "x")
        eq.shift(2.5 * UP)

        self.play(Write(func), Write(eq))
        self.wait()

        self.play(Uncreate(func), ApplyMethod(eq.shift, 2.5 * DOWN))

        eq2 = PieceWiseTwo(r"x-20 \geq 0", "x-20 < 0", "x-20")
        eq2.scale(1.5)

        self.play(Transform(eq, eq2))
        self.wait()
