from manimlib import *
import tiktoken

"""
Scenes In Order:

NeuralLM
Pendulum
Extrapolation
RNNIntro
RNNTraining
RNNInference
RNNBackprop
RNNBackpropLong
MachineTranslation
Attention
"""

A_AQUA = "#8dd3c7"
A_YELLOW = "#ffffb3"
A_LAVENDER = "#bebada"
A_RED = "#fb8072"
A_BLUE = "#80b1d3"
A_ORANGE = "#fdb462"
A_GREEN = "#b3de69"
A_PINK = "#fccde5"
A_GREY = "#d9d9d9"
A_VIOLET = "#bc80bd"
A_UNKA = "#ccebc5"
A_UNKB = "#ffed6f"


def softmax(x, axis=1):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class Document(VMobject):
    def __init__(self, rect_color=GREY_D, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rect = Rectangle(
            height=2.5, width=2, fill_color=rect_color, fill_opacity=1
        )
        self.lines = VGroup(
            *[
                Line(0.75 * LEFT, 0.75 * RIGHT).shift(0.25 * (i - 3) * UP)
                for i in range(7)
            ]
        )
        self.lines[-1].set_width(1)
        self.lines[-1].shift(0.25 * LEFT)

        self.add(self.rect, self.lines)


class WordDistribution(VMobject):
    def __init__(
        self,
        words,
        probs,
        bar_height=0.5,
        max_bar_width=1.5,
        word_scale=1.0,
        prob_scale=1.0,
        bar_spacing=FRAME_HEIGHT / 10,
        prob_bar_color=A_LAVENDER,
        incl_word_labels=True,
        incl_prob_labels=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prob_bars_small = VGroup()
        self.prob_bars_large = VGroup()
        self.words = VGroup()
        self.probs = VGroup()

        self.incl_word_labels = incl_word_labels
        self.incl_prob_labels = incl_prob_labels

        for i, (word, prob) in enumerate(zip(words, probs)):
            bar_small = Rectangle(
                height=bar_height,
                width=0,
                fill_color=prob_bar_color,
                fill_opacity=1,
                stroke_width=0,
            )

            bar_large = Rectangle(
                height=bar_height,
                width=prob * max_bar_width,
                fill_color=prob_bar_color,
                fill_opacity=1,
            )

            bar_small.move_to(bar_spacing * i * DOWN, LEFT)
            bar_large.move_to(bar_spacing * i * DOWN, LEFT)

            if self.incl_word_labels:
                word_text = Text(word)
                word_text.scale(word_scale)
                word_text.move_to(bar_large.get_bounding_box_point(LEFT) + 1 * LEFT)
                self.words.add(word_text)

            if self.incl_prob_labels:
                prob_text = Text(f"{prob:.4f}")
                prob_text.scale(prob_scale)
                prob_text.move_to(bar_large.get_bounding_box_point(RIGHT) + 1 * RIGHT)
                self.probs.add(prob_text)

            self.prob_bars_small.add(bar_small)
            self.prob_bars_large.add(bar_large)

        self.add(self.prob_bars_small, self.prob_bars_large, self.words, self.probs)
        self.center()

    def write(self, scene, text_run_time=1.5, prob_run_time=0.75):
        if self.incl_word_labels:
            scene.play(
                Write(self.words), Write(self.prob_bars_small), run_time=text_run_time
            )
        else:
            scene.play(Write(self.prob_bars_small), run_time=text_run_time)

        for i in range(len(self.prob_bars_small)):
            if self.incl_prob_labels:
                scene.play(
                    ApplyMethod(
                        self.prob_bars_small[i].become, self.prob_bars_large[i]
                    ),
                    FadeIn(self.probs[i], RIGHT),
                    run_time=prob_run_time,
                )
            else:
                scene.play(
                    ApplyMethod(
                        self.prob_bars_small[i].become, self.prob_bars_large[i]
                    ),
                    run_time=prob_run_time,
                )

        scene.remove(self.prob_bars_small)
        scene.add(self.prob_bars_large)


class NeuralNetwork(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "first_layer_buff": MED_SMALL_BUFF,
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
        self.add(self.layers)

    def add_neurons(self):
        layers = VGroup(
            *[
                self.get_layer(size, index)
                for index, size in enumerate(self.layer_sizes)
            ]
        )
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.add_output_labels()

    def get_nn_fill_color(self, index):
        if index == -1:
            return self.neuron_stroke_color
        if index == 0:
            return A_PINK
        elif index == len(self.layer_sizes) - 1:
            return A_BLUE
        else:
            return A_GREEN

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
        neurons.arrange(
            DOWN,
            buff=self.first_layer_buff if index == 0 else self.neuron_to_neuron_buff,
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex("\\vdots")
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
            label = Tex(f"x_{n + 1}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_y(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(r"\hat{y}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def add_weight_labels(self):
        weight_group = VGroup()

        for n, i in enumerate(self.layers[0].neurons):
            edge = self.get_edge(i, self.layers[-1][0])
            text = Tex(f"w_{n + 1}", color=RED)
            text.move_to(edge)
            weight_group.add(text)
        self.add(weight_group)

    def add_output_labels(self, labels=None):
        if labels is None:
            labels = list(map(str, range(len(self.layers[-1].neurons) + 1)))

        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(labels[n])
            label.scale(0.85)
            label.move_to(neuron)
            label.shift(1 * RIGHT)
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


class RNNCell(VMobject):
    CONFIG = {
        "fill_color": A_RED,
        "left_most": False,
        "right_most": False,
        "include_input_arrow": True,
        "include_output_arrow": True,
        "arrow_length": 1.5,
        "arrow_buff": 0.25,
        "arrow_color": A_GREY,
        "arrow_width": 10,
        "add_labels": True,
        "label_buff": 0.25,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sq = Square(fill_opacity=0.75, fill_color=self.fill_color)
        self.add(self.sq)

        self.arrows = VGroup()
        if not self.left_most:
            self.add_arrow(
                self.sq.get_left() + self.arrow_length * LEFT,
                self.sq.get_left(),
            )
            self.left_arrow = self.arrows[-1]

        if not self.right_most:
            self.add_arrow(
                self.sq.get_right(),
                self.sq.get_right() + self.arrow_length * RIGHT,
            )
            self.right_arrow = self.arrows[-1]

        if self.include_output_arrow:
            self.add_arrow(
                self.sq.get_top(),
                self.sq.get_top() + self.arrow_length * UP,
            )
            self.up_arrow = self.arrows[-1]

        if self.include_input_arrow:
            self.add_arrow(
                self.sq.get_bottom() + self.arrow_length * DOWN,
                self.sq.get_bottom(),
            )
            self.down_arrow = self.arrows[-1]

        self.add(self.arrows)
        self.get_labels(add_to_obj=self.add_labels)

        self.center()

    def add_arrow(self, start, end):
        self.arrows.add(
            Arrow(
                start,
                end,
                max_width_to_Length_ratio=float("inf"),
                stroke_width=self.arrow_width,
                stroke_color=self.arrow_color,
                buff=self.arrow_buff,
            )
        )

    def get_labels(self, english=True, add_to_obj=True):
        self.labels = VGroup()
        if not english:
            labels = ["{h}_{{t}-1}", "{h}_{t}", "{y}_{t}", "{x}_{t}"]
        else:
            labels = [
                r"\text{previous } {h}",
                r"\text{next } {h}",
                r"\text{output } {y}",
                r"\text{input } {x}",
            ]

        tex_to_color_map = {
            "{x}": A_PINK,
            "{h}": A_GREEN,
            "{y}": A_BLUE,
            "{t}": A_YELLOW,
            "1": A_UNKA,
        }

        for n, (label, arrow) in enumerate(zip(labels, self.arrows)):
            curr_arrow_vec = arrow.get_end() - arrow.get_start()
            label_tex = Tex(label, tex_to_color_map=tex_to_color_map)
            if not english:
                label_tex.scale(1.25)
            if (n & 1) ^ (n >> 1 & 1):  # odd number of bits
                label_tex.next_to(arrow.get_end(), curr_arrow_vec)
            else:
                label_tex.next_to(arrow.get_start(), -curr_arrow_vec)

            self.labels.add(label_tex)

        if add_to_obj:
            self.add(self.labels)

        return self.labels


class RNN(VMobject):
    def __init__(
        self,
        n_cells=4,
        remove_left_arrow=True,
        remove_right_arrow=True,
        rnn_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_cells = n_cells
        self.cells = VGroup()
        for i in range(n_cells):
            if i == 0 and remove_left_arrow:
                c = RNNCell(add_labels=False, left_most=True, **rnn_kwargs)
            elif i == n_cells - 1 and remove_right_arrow:
                c = RNNCell(add_labels=False, right_most=True, **rnn_kwargs)
            else:
                c = RNNCell(add_labels=False, **rnn_kwargs)

            if i != 0:
                if i == 1 and remove_left_arrow:
                    left = self.cells[i - 1].arrows[0].get_center()
                else:
                    left = self.cells[i - 1].arrows[1].get_center()
                right = c.arrows[0].get_center()
                c.shift(left - right)
            self.cells.add(c)

        self.add(self.cells)
        self.center()

    def get_inputs(self, words, **text_kwargs):
        assert len(words) >= self.n_cells

        input_words = VGroup()
        for i in range(self.n_cells):
            t = Text(words[i], **text_kwargs)
            t.move_to(self.cells[i].down_arrow.get_start() + 0.5 * DOWN)
            input_words.add(t)

        return input_words

    def get_outputs(self, words, **text_kwargs):
        assert len(words) >= self.n_cells

        output_words = VGroup()
        for i in range(self.n_cells):
            t = Text(words[i], **text_kwargs)
            t.move_to(self.cells[i].up_arrow.get_end() + 0.5 * UP)
            output_words.add(t)

        return output_words


class TitleScene(Scene):
    CONFIG = {"color": None, "text": None, "tex_to_color_map": {}}

    def construct(self):
        if self.text is None:
            raise NotImplementedError

        brect = Rectangle(
            height=FRAME_HEIGHT, width=FRAME_WIDTH, fill_opacity=1, color=self.color
        )

        title = TexText(
            self.text if isinstance(self.text, str) else self.text[0],
            tex_to_color_map=self.tex_to_color_map,
        )
        title.scale(1.5)
        title.to_edge(UP)

        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)

        self.add(brect)
        self.play(FadeIn(rect, DOWN), Write(title), run_time=2)
        self.wait()

        if isinstance(self.text, list):
            for i in range(1, len(self.text)):
                new_title = TexText(
                    self.text[i], tex_to_color_map=self.tex_to_color_map
                )
                new_title.scale(1.5)
                new_title.to_edge(UP)

                self.play(FadeOut(title, UP), FadeIn(new_title, UP))
                self.wait()

                title = new_title


class TitleL(TitleScene):
    CONFIG = {"color": "#5c608f", "text": "Last Video"}


class TitleDpub(TitleScene):
    CONFIG = {
        "color": GREY_E,
        "text": "distill.pub",
    }


class NeuralLM(Scene):
    def construct(self):
        head = Text("Neural Language Model", color=A_RED)
        head.scale(1.5)
        head.shift(3 * UP)

        yoshua_img = ImageMobject("img/yoshua.jpeg")
        yoshua_rect = SurroundingRectangle(
            yoshua_img, buff=0, color=WHITE, stroke_width=6
        )
        yoshua_text = Text("Yoshua Bengio").next_to(yoshua_img, DOWN)
        yoshua = Group(yoshua_img, yoshua_rect, yoshua_text)

        self.play(Write(head))
        self.play(GrowFromPoint(yoshua, head.get_center()))
        self.wait()

        self.play(Transform(yoshua, yoshua.copy().move_to(head).scale(0)))
        self.wait()

        words = ["the", "sky", "is"]
        word_objs, word_vecs = VGroup(), VGroup()
        arrows = VGroup()

        np.random.seed(10)
        for n, word in enumerate(words):
            t = Text(word)
            t.scale(1.25)
            t.shift(n * 2 * DOWN)
            word_objs.add(t)

            vec = Tex(
                f"""
                \\begin{{bmatrix}}
                {20 * (np.random.rand() - 0.5):.2f} \\\\
                {20 * (np.random.rand() - 0.5):.2f} \\\\
                \\vdots \\\\
                {20 * (np.random.rand() - 0.5):.2f}
                \\end{{bmatrix}}
                """
            )
            vec.scale(0.5)
            vec.shift(n * 2 * DOWN + 2 * RIGHT)
            word_vecs.add(vec)

            arrow = Arrow(
                0.75 * RIGHT,
                1.25 * RIGHT,
                max_width_to_Length_ratio=float("inf"),
                stroke_width=10,
                stroke_color=A_YELLOW,
                buff=0,
            )
            arrow.shift(n * 2 * DOWN)
            arrows.add(arrow)

        grp = VGroup(word_objs, word_vecs, arrows)
        grp.center()
        grp.shift(0.5 * DOWN)

        word_objs_center = word_objs.get_center().copy()
        word_objs.center()
        word_objs.shift(0.5 * DOWN)

        self.play(Write(word_objs))
        self.play(
            ApplyMethod(word_objs.move_to, word_objs_center),
            FadeIn(arrows, RIGHT),
            FadeIn(word_vecs, RIGHT),
        )
        self.wait()

        layer_buff = (word_objs[0].get_center() - word_objs[1].get_center())[1]
        layer_buff -= 1

        nn = NeuralNetwork(
            [3, 5, 100],
            neuron_radius=0.15,
            neuron_stroke_width=3,
            max_shown_neurons=6,
            brace_for_large_layers=False,
            first_layer_buff=layer_buff,
        )
        nn.scale(1.5)
        nn.shift(2.11 * RIGHT + 0.5 * DOWN)

        nn.add_output_labels(
            labels=[
                r"\text{blue}",
                r"\text{red}",
                r"\text{green}",
                r"\text{yellow}",
                r"\text{orange}",
                r"\text{purple}",
            ]
        )

        grp_nn_arrows = VGroup()
        for i in range(3):
            grp_nn_arrows.add(
                Arrow(
                    word_vecs[i].get_bounding_box_point(RIGHT) + 3.55 * LEFT,
                    nn.layers[0].neurons[i].get_bounding_box_point(LEFT),
                    stroke_width=5,
                    stroke_color=A_YELLOW,
                    buff=0.35,
                )
            )

        self.play(ApplyMethod(grp.shift, 3.55 * LEFT), Write(grp_nn_arrows))
        self.play(Write(nn))
        self.wait()

        self.play(FadeOut(VGroup(grp, grp_nn_arrows, nn), UP))

        first_sent = "the movie is great".split(" ")
        second_sent = "that was great !".split(" ")
        grey_black_grad = color_gradient([GREY, BLACK], 10)

        nn_left = NeuralNetwork(
            [4, 4],
            neuron_radius=0.15,
            neuron_stroke_width=3,
            max_shown_neurons=6,
            brace_for_large_layers=False,
            first_layer_buff=0.5,
        )
        nn_left.scale(1.5)
        nn_left.rotate(90 * DEGREES)
        nn_left.layers.remove(nn_left.layers[1])
        nn_left.edge_groups.set_color(grey_black_grad)

        nn_right = nn_left.deepcopy()
        first_sent_grp, second_sent_grp = VGroup(), VGroup()
        for i in range(4):
            t1 = Text(first_sent[i])
            t1.move_to(nn_left.layers[0].neurons[i])
            t1.shift(0.675 * DOWN)
            first_sent_grp.add(t1)

            t2 = Text(second_sent[i])
            t2.move_to(nn_left.layers[0].neurons[i])
            t2.shift(0.675 * DOWN)
            second_sent_grp.add(t2)

        left_grp = VGroup(nn_left, first_sent_grp)
        right_grp = VGroup(nn_right, second_sent_grp)

        left_grp.scale(1.25)
        right_grp.scale(1.25)

        left_grp.move_to(FRAME_WIDTH / 4 * LEFT)
        right_grp.move_to(FRAME_WIDTH / 4 * RIGHT)

        self.play(
            ShowCreation(nn_left.edge_groups),
            Write(nn_left.layers),
            Write(first_sent_grp),
        )
        self.play(
            ShowCreation(nn_right.edge_groups),
            Write(nn_right.layers),
            Write(second_sent_grp),
        )
        self.wait()

        self.play(Indicate(second_sent_grp[0]))
        self.wait()

        self.embed()


class Pendulum(Scene):
    def construct(self):
        origin = 2 * UP
        theta_0 = 20 * DEGREES
        L = 3

        string = Line(2 * UP, DOWN)
        bob = Sphere(radius=0.35, color=A_BLUE, fill_opacity=1)

        t = ValueTracker(0)

        def compute_bob_pos(t):
            # sin(x) = x approximation
            theta = theta_0 * np.cos(np.sqrt(9.8 / L) * t)
            x_t = L * np.sin(theta)
            y_t = -L * np.cos(theta)
            return np.array([x_t, y_t, 0]) + origin

        def bob_updater(bob):
            bob_pos = compute_bob_pos(t.get_value())
            bob.move_to(bob_pos)

        def string_updater(string):
            bob_pos = compute_bob_pos(t.get_value())
            string.put_start_and_end_on(origin, bob_pos)

        bob.add_updater(bob_updater)
        string.add_updater(string_updater)

        c = VMobject()

        def curve_updater(c):
            max_t = t.get_value()
            if max_t < 5:
                opacity = 0
            elif 5 <= max_t <= 7:
                opacity = (max_t - 5) / 2
            else:
                opacity = 1.0

            c_new = ParametricCurve(
                lambda t: [compute_bob_pos(t)[0], -1 - (max_t - t), 0],
                t_range=(5, max_t),
                stroke_opacity=opacity,
                stroke_color=A_AQUA,
                stroke_width=6,
            )
            c.become(c_new)

        c.add_updater(curve_updater)

        self.add(c)
        self.play(Write(string), FadeIn(bob))

        self.play(ApplyMethod(t.increment_value, 30), run_time=25, rate_func=linear)
        self.wait()

        self.embed()


class Extrapolation(Scene):
    def construct(self):
        func = lambda t: 1.5 * np.sin(1.5 * t)

        axes = Axes(
            x_range=(0, 10), y_range=(-3, 3), axis_config={"include_tip": False}
        )
        sin_wave = axes.get_graph(
            func,
            x_range=(0, 10),
            stroke_width=6,
            color=A_RED,
        )

        def get_dot(x):
            coords = axes.coords_to_point(x, func(x))
            return Dot([coords[0], coords[1], 0], color=A_GREY)

        x_coords = [1.5, 3.75, 6.5, 8.5]
        y_coords = [func(x) for x in x_coords]
        dots = VGroup(*[get_dot(x) for x in x_coords])

        poly_interp = np.poly1d(np.polyfit(x_coords, y_coords, 3))
        poly_curve = axes.get_graph(
            poly_interp,
            x_range=(0, 10),
            stroke_width=6,
            color=A_GREEN,
        )

        self.play(Write(axes), Write(sin_wave))
        self.wait()

        self.play(Write(dots))
        self.play(FadeOut(sin_wave))
        self.wait()

        self.bring_to_back(poly_curve)
        self.play(Write(poly_curve))
        self.wait()

        more_dots = VGroup(*[get_dot(x) for x in np.arange(0, 10, 0.5)])
        self.play(FadeOut(poly_curve))
        self.play(ShowCreation(more_dots))
        self.wait()

        many_sin_curves = VGroup()
        for i in np.linspace(0.5, 3.0, 10):
            t = (i - 0.5) / 2.5
            color = rgb_to_hex(t * hex_to_rgb(A_PINK) + (1 - t) * hex_to_rgb(A_GREEN))

            curr_sin = axes.get_graph(
                lambda t: 1.5 * np.sin(i * t),
                x_range=(0, 10),
                stroke_width=3,
                stroke_opacity=0.65 * t + 0.25,
                color=color,
            )
            many_sin_curves.add(curr_sin)

        self.play(Uncreate(more_dots), Uncreate(dots))
        for i in many_sin_curves:
            self.play(Write(i), run_time=0.5)
        self.wait()

        self.embed()


class RNNIntro(Scene):
    def construct(self):
        sent = ["the", "sky", "is", "blue"]
        words, arrows = VGroup(), VGroup()

        for n, i in enumerate(sent):
            w = Text(i)
            w.scale(1.5)
            w.shift(n * 3 * RIGHT)
            words.add(w)

            if n != 0:
                a = Arrow(
                    words[n - 1].get_right(),
                    w.get_left(),
                    buff=MED_LARGE_BUFF,
                    stroke_width=8,
                    max_tip_length_to_length_ratio=float("inf"),
                    stroke_color=A_YELLOW,
                )
                arrows.add(a)

        grp = VGroup(words, arrows)
        grp.center()

        self.play(Write(words[0]))
        for i in range(3):
            self.play(
                FadeIn(arrows[i], RIGHT), FadeIn(words[i + 1], RIGHT), run_time=0.75
            )
        self.wait()

        title = Text("Recurrence", color=A_YELLOW)
        title.scale(1.5)
        title.shift(3.25 * UP)

        self.play(Write(title))
        self.wait()

        new_title = Text("Recurrent Neural Network", color=A_YELLOW)
        new_title.scale(1.5)
        new_title.shift(3.25 * UP)

        self.play(
            Transform(title[:8], new_title[:8]),
            Uncreate(title[8:]),
            Write(new_title[8:]),
        )
        self.play(Uncreate(VGroup(words, arrows)))

        r = RNNCell()
        r.shift(0.5 * DOWN)

        self.play(Write(r.sq))
        self.wait()

        self.play(Write(r.labels[3]), FadeIn(r.arrows[3], UP))
        self.play(Write(r.labels[2]), FadeIn(r.arrows[2], UP))
        self.wait()

        self.play(Write(r.labels[0]), FadeIn(r.arrows[0], RIGHT))
        self.play(Write(r.labels[1]), FadeIn(r.arrows[1], RIGHT))
        self.wait()

        rnn = RNN(n_cells=4)
        rnn.scale(0.75)

        self.play(
            FadeOut(r.labels[0], LEFT),
            FadeOut(r.labels[1], RIGHT),
            FadeOut(r.labels[2], UP),
            FadeOut(r.labels[3], DOWN),
        )
        self.play(
            Transform(
                VGroup(r.sq, r.arrows),
                rnn.cells[1],
                replace_mobject_with_target_in_scene=True,
            )
        )
        self.play(Write(rnn.cells[0]), Write(rnn.cells[2:]))

        words = VGroup()
        for n, i in enumerate(sent):
            w = Text(i)
            w.scale(1.25)
            w.next_to(rnn.cells[n].down_arrow.get_start(), 1.5 * DOWN)
            words.add(w)

        self.play(FadeIn(words, UP))
        self.wait()

        self.play(
            FadeOut(words, DOWN),
            FadeOut(rnn.cells[0], DOWN),
            FadeOut(rnn.cells[2:], DOWN),
        )
        rnn_cell = rnn.cells[1]
        self.play(
            rnn_cell.become,
            rnn_cell.deepcopy().scale(1).move_to(0.5 * DOWN + 3.75 * LEFT),
        )
        self.wait()

        labels = rnn_cell.get_labels(english=False, add_to_obj=False)
        self.play(Write(labels[0]))
        self.play(Write(labels[1]))
        self.play(Write(labels[2]))
        self.play(Write(labels[3]))
        self.wait()

        eq1 = Tex(
            r"{h}_{t} = \sigma ( {W} {h}_{{t}-1} + {V} {x}_{t} + {b}_{h})",
            tex_to_color_map={
                "{h}": A_GREEN,
                "{t}": A_YELLOW,
                "{x}": A_PINK,
                r"\sigma": A_AQUA,
                "{W}": A_GREY,
                "{V}": A_GREY,
                "{b}": A_BLUE,
                "1": A_UNKA,
            },
        )
        eq1.scale(1.25)
        eq1.shift(3 * RIGHT + 1 * UP)

        eq2 = Tex(
            r"{y}_{t} = \sigma ( {U} {h}_{t} + {b}_{y})",
            tex_to_color_map={
                "{y}": A_BLUE,
                "{t}": A_YELLOW,
                "{h}": A_GREEN,
                r"\sigma": A_AQUA,
                "{U}": A_GREY,
                "{b}": A_BLUE,
                "1": A_UNKA,
            },
        )
        eq2.scale(1.25)
        eq2.shift(3 * RIGHT + 1 * DOWN)

        self.play(TransformFromCopy(labels[1], eq1[:2]), Write(eq1[2]))
        self.play(TransformFromCopy(labels[0], eq1[6:10]), Write(eq1[5]))
        self.play(TransformFromCopy(labels[3], eq1[12:14]), Write(eq1[10:12]))
        self.play(Write(eq1[14:-1]))
        self.play(Write(eq1[-1]), Write(eq1[3:5]))
        self.wait()

        self.play(TransformFromCopy(labels[2], eq2[:2]), Write(eq2[2]))
        self.play(TransformFromCopy(labels[1], eq2[6:8]), Write(eq2[5]))
        self.play(Write(eq2[8:11]))
        self.play(Write(eq2[11:]), Write(eq2[3:5]))
        self.wait()

        self.embed()


class RNNTraining(Scene):
    def construct(self):
        documents = VGroup()
        for i in range(5):
            d = Document()
            d.scale(0.75)
            d.shift(i * 2 * RIGHT)
            documents.add(d)
        documents.center()
        documents.shift(2 * DOWN)

        rnn = RNN(n_cells=4)
        rnn.scale(0.75)
        rnn.shift(1.5 * UP)

        self.play(Write(documents[2]))
        self.play(*[TransformFromCopy(documents[2], documents[i]) for i in range(5)])
        self.play(Write(rnn))

        self.embed()


class RNNInference(Scene):
    def construct(self):
        raw_text = "Apple is trying to limit how often\nyour iPhone apps can bug you\nto give them a rating"
        next_probs = np.load("rnn/next_probs.npy")

        def get_next_probs(idx):
            new_words, new_probs = [], []
            for w, p in next_probs[idx]:
                new_words.append(w.strip())
                new_probs.append(float(p))
            return new_words[:7], new_probs[:7]

        l = Line(10 * UP, 10 * DOWN)
        l.shift(2 * RIGHT)

        rnn_text = Text("RNN Model", color=A_YELLOW)
        rnn_text.scale(1.25)
        rnn_text.shift(4.5 * RIGHT + 3 * UP)

        prob_bars = WordDistribution(
            *get_next_probs(0), max_bar_width=1.5, word_scale=0.75, prob_scale=0.75
        )
        prob_bars.shift(4.5 * RIGHT + 0.5 * DOWN)
        prob_bars.scale(0.9)

        text = TexText(*[i.replace("\n", r"\\") for i in raw_text], alignment="")
        text.move_to(6.32553125 * LEFT + 3.21861875 * UP, UP + LEFT)

        self.play(Write(rnn_text), Write(l))
        prob_bars.write(self, text_run_time=0.5, prob_run_time=0.25)
        self.wait()

        tex_idx = 0
        for raw_idx in range(len(raw_text)):
            if raw_text[raw_idx] == "\n":
                continue

            if raw_idx > 10:
                run_time = 0.5
            else:
                run_time = 1

            next_word = raw_text[raw_idx]
            prob_word_obj = None

            if next_word != " ":
                for word_obj in prob_bars.words:
                    if word_obj.text.strip() == next_word.strip():
                        prob_word_obj = word_obj
                        break
            else:
                for i in range(len(prob_bars.words)):
                    if prob_bars.words[i].text.strip() == "":
                        prob_word_obj = prob_bars.prob_bars_large[i]
                        break

            if prob_word_obj is None:
                prob_word_obj = Text(next_word)
                prob_word_obj.scale(0.75)
                prob_word_obj = prob_word_obj.move_to(prob_bars)
                prob_word_obj.shift(FRAME_HEIGHT / 2 * 1 * DOWN)
            else:
                self.play(Indicate(prob_word_obj), run_time=0.5 * run_time)

            new_prob_word_obj = prob_word_obj.copy()
            new_prob_word_obj.scale(1 / 0.75)
            new_prob_word_obj.move_to(text[tex_idx])

            if next_word != " ":
                self.play(
                    TransformFromCopy(prob_word_obj, new_prob_word_obj),
                    run_time=run_time,
                )
                self.wait(0.5 * run_time)

            new_dist = WordDistribution(
                *get_next_probs(raw_idx + 1),
                max_bar_width=1.5,
                word_scale=0.75,
                prob_scale=0.75,
            )
            new_dist.move_to(prob_bars)

            anims = [
                FadeOut(prob_bars.words, UP),
                FadeOut(prob_bars.probs, UP),
                FadeIn(new_dist.words, UP),
                FadeIn(new_dist.probs, UP),
            ]
            for i in range(len(prob_bars.words)):
                anims += [
                    Transform(
                        prob_bars.prob_bars_large[i],
                        new_dist.prob_bars_large[i],
                    ),
                ]

            self.play(*anims, run_time=1)
            self.remove(prob_bars)
            self.add(new_dist.words, new_dist.probs, new_dist.prob_bars_large)

            prob_bars = new_dist
            if next_word != " ":
                tex_idx += 1

        self.embed()


class RNNBackprop(Scene):
    def construct(self):
        rnn = RNN(n_cells=3)
        rnn.scale(0.75)

        rnn_texts = rnn.get_inputs(["the", "sky", "is"])

        self.play(Write(rnn))
        self.play(Write(rnn_texts))
        self.wait()

        w = WordDistribution(
            ["" for _ in range(4)],
            [0.15, 0.5, 0.3, 0.05],
            bar_spacing=0.5,
            incl_prob_labels=False,
            incl_word_labels=False,
        )
        w.rotate(90 * DEGREES)
        w.move_to(rnn.cells[-1].up_arrow.get_end() + 0.75 * UP)

        w.write(self)
        self.wait()

        self.play(w.move_to, rnn.cells[-2].up_arrow.get_end() + 0.75 * UP)
        self.wait()

        self.embed()


class RNNBackpropLong(Scene):
    def construct(self):
        rnn = RNN(n_cells=10)
        rnn.scale(0.75)
        rnn.move_to((FRAME_WIDTH / 2 - 1) * RIGHT, RIGHT)

        self.play(Write(rnn))
        self.wait()

        w = WordDistribution(
            ["" for _ in range(4)],
            [0.15, 0.5, 0.3, 0.05],
            bar_spacing=0.5,
            incl_prob_labels=False,
            incl_word_labels=False,
        )
        w.rotate(90 * DEGREES)
        w.move_to(rnn.cells[-1].up_arrow.get_end() + 0.75 * UP)

        w.write(self)
        self.wait()

        anims = []
        for i in range(len(rnn.cells) - 2, -1, -1):
            anims += [
                ApplyMethod(w.move_to, rnn.cells[i].up_arrow.get_end() + 0.75 * UP)
            ]

        new_frame = self.camera.frame.deepcopy()
        new_frame.set_width(2 * FRAME_WIDTH)
        new_frame.move_to(rnn.get_center())

        self.play(
            Succession(*anims),
            ApplyMethod(self.camera.frame.become, new_frame),
            run_time=10,
        )
        self.wait()
        self.embed()


class MachineTranslation(Scene):
    def construct(self):
        np.random.seed(0)

        rnn_encoder = RNN(
            n_cells=4, remove_right_arrow=False, rnn_kwargs={"fill_color": A_RED}
        )
        rnn_decoder = RNN(
            n_cells=4, remove_left_arrow=False, rnn_kwargs={"fill_color": A_BLUE}
        )

        offset = (
            rnn_encoder.cells[-1].right_arrow.get_center()
            - rnn_decoder.cells[0].left_arrow.get_center()
        )
        rnn_decoder.shift(offset)

        rnn = VGroup(rnn_encoder, rnn_decoder)
        rnn.center()
        rnn.scale(0.5)

        anims = [Write(rnn_encoder.cells[0])]
        for i in range(1, 8):
            if i < 4:
                anims += [FadeIn(rnn_encoder.cells[i], RIGHT)]
            else:
                anims += [FadeIn(rnn_decoder.cells[i - 4], RIGHT)]

        each_run_time = 3.0 / len(anims)
        for anim in anims:
            self.play(anim, run_time=each_run_time)
        self.wait()

        b1 = Brace(rnn_encoder, UP)
        b1.add(b1.get_text("Encoder"))

        b2 = Brace(rnn_decoder, DOWN)
        b2.add(b2.get_text("Decoder"))

        self.play(Write(b1))
        self.play(Write(b2))
        self.wait()

        self.play(Uncreate(b1), Uncreate(b2))

        input_text = rnn_encoder.get_inputs(["my", "name", "is", "vivek"])
        output_text = rnn_decoder.get_outputs(["मेरा", "नाम", "विवेक", "है"])

        for i in range(4):
            self.play(FadeIn(input_text[i], UP, run_time=0.5))

        self.play(FadeIn(output_text[0], UP), run_time=0.5)
        for i in range(1, 4):
            prev_word_copy = output_text[i - 1].deepcopy()
            prev_word_copy.move_to(
                rnn_decoder.cells[i].down_arrow.get_start() + 0.5 * DOWN
            )

            self.play(
                TransformFromCopy(output_text[i - 1], prev_word_copy), run_time=0.5
            )
            self.play(FadeIn(output_text[i], UP), run_time=0.5)
        self.wait()

        vec = Tex(
            f"""
            \\begin{{bmatrix}}
            {20 * (np.random.rand() - 0.5):.2f} \\\\
            {20 * (np.random.rand() - 0.5):.2f} \\\\
            \\vdots \\\\
            {20 * (np.random.rand() - 0.5):.2f}
            \\end{{bmatrix}}
            """
        )
        vec.scale(0.65)

        meaning_text = Text("encodes meaning")
        meaning_text.next_to(vec, RIGHT)
        meaning_text.shift(RIGHT)

        arr = Arrow(
            vec,
            meaning_text,
            stroke_width=10,
            max_width_to_length_ratio=float("inf"),
            stroke_color=A_VIOLET,
        )

        meaning_grp = VGroup(vec, meaning_text, arr)
        meaning_grp.move_to(2.75 * UP)

        self.play(GrowFromPoint(vec, ORIGIN))
        self.play(Write(arr), Write(meaning_text))
        self.wait()

        self.play(Uncreate(meaning_grp))

        scale_coeffs = np.polyfit([0, 0.5, 1], [0, 0.5, 0], 2)
        scale_curve = lambda x: np.polyval(scale_coeffs, x)

        for i in range(4):
            original_vec = Tex(
                f"""
                \\begin{{bmatrix}}
                {20 * (np.random.rand() - 0.5):.2f} \\\\
                {20 * (np.random.rand() - 0.5):.2f} \\\\
                \\vdots \\\\
                {20 * (np.random.rand() - 0.5):.2f}
                \\end{{bmatrix}}
                """
            )
            original_vec.scale(0.65)

            vec = original_vec.deepcopy()

            start_point = rnn_encoder.cells[i].get_center()
            vec.move_to(start_point)

            points = np.array([start_point, start_point / 2 + 2 * UP, ORIGIN])
            pos_coeffs = np.polyfit([0, 0.5, 1], points, 2)
            pos_curve = lambda x: np.polyval(pos_coeffs, x)
            vt = ValueTracker(0)

            def vec_updater(v):
                new_vec = original_vec.deepcopy()
                new_vec.scale(scale_curve(vt.get_value()))
                new_vec.move_to(pos_curve(vt.get_value()))

                v.become(new_vec)

            vec.add_updater(vec_updater)

            self.add(vec)
            self.play(vt.increment_value, 1, run_time=0.5)

        b_rect = SurroundingRectangle(
            VGroup(
                rnn_encoder.cells[-1].sq,
                rnn_encoder.cells[-1].up_arrow,
                rnn_encoder.cells[-1].down_arrow,
            ),
            color=A_YELLOW,
            stroke_width=6,
            fill_opacity=0,
        )
        b_text = Text("Bottleneck", color=A_YELLOW)
        b_text.next_to(b_rect, UP)

        self.play(Write(b_rect), Write(b_text))
        self.wait()

        self.play(
            Uncreate(b_rect),
            Uncreate(b_text),
            Uncreate(input_text),
            Uncreate(output_text),
        )

        self.embed()


class Attention(Scene):
    def construct(self):
        rnn_encoder = RNN(
            n_cells=4, remove_right_arrow=False, rnn_kwargs={"fill_color": A_RED}
        )
        rnn_decoder = RNN(
            n_cells=4, remove_left_arrow=False, rnn_kwargs={"fill_color": A_BLUE}
        )

        offset = (
            rnn_encoder.cells[-1].right_arrow.get_center()
            - rnn_decoder.cells[0].left_arrow.get_center()
        )
        rnn_decoder.shift(offset)

        rnn = VGroup(rnn_encoder, rnn_decoder)
        rnn.center()
        rnn.scale(0.5)

        self.add(rnn)

        title = Text("Attention Mechanism", color=A_YELLOW)
        title.scale(1.5)
        title.shift(3.25 * UP)

        self.play(Write(title))
        self.wait()

        init_arrows = VGroup()
        for decoder_idx in range(4):
            for encoder_idx in range(4):
                mask = 2 * (decoder_idx & 1) - 1

                start_point = rnn_encoder.cells[encoder_idx].sq.get_bounding_box_point(
                    -mask * UP + RIGHT
                )
                end_point = rnn_decoder.cells[decoder_idx].sq.get_bounding_box_point(
                    -mask * UP + LEFT
                )

                arrow = CurvedArrow(
                    start_point,
                    end_point,
                    angle=90 * DEGREES * mask,
                    stroke_width=4,
                    tip_config={"width": 0.2, "length": 0.2},
                    color=[A_LAVENDER, A_AQUA, A_RED, A_PINK][decoder_idx],
                )
                init_arrows.add(arrow)

                self.play(ShowCreation(arrow), run_time=0.5)
        self.wait()

        self.play(Uncreate(init_arrows))

        new_dec = rnn_decoder.cells[0].deepcopy()
        new_enc = rnn_encoder.deepcopy()
        new_dec.shift(2 * RIGHT)

        new_cells = VGroup(new_enc, new_dec)
        new_cells.center()
        new_cells.shift(2.5 * DOWN)

        self.play(
            Uncreate(rnn_decoder.cells[1:]),
            Transform(rnn_encoder, new_enc),
            Transform(rnn_decoder.cells[0], new_dec),
        )
        self.wait()

        encoder_labels = VGroup()
        for i in range(4):
            lbl = Tex(f"h_{i+1}")
            lbl.move_to(rnn_encoder.cells[i].sq)
            lbl.scale(1.25)
            encoder_labels.add(lbl)

        decoder_label = Tex("s")
        decoder_label.scale(1.25)
        decoder_label.move_to(rnn_decoder.cells[0].sq)

        self.play(Write(decoder_label))
        self.play(Write(encoder_labels))
        self.wait()

        arr_1, arr_2 = VGroup(), VGroup()
        score_lbl, bars = VGroup(), VGroup()
        distrib = [0.05, 0.1, 0.65, 0.2]

        for i in range(4):
            arr = Arrow(
                rnn_decoder.cells[0].sq.get_bounding_box_point(UP) + 0.125 * UP,
                rnn_encoder.cells[i].up_arrow.get_end(),
                max_width_to_Length_ratio=float("inf"),
                stroke_width=4,
                stroke_color=A_GREY,
                buff=0.125,
            )
            arr_1.add(arr)
            self.play(ShowCreation(arr), run_time=0.5)

            lbl = Tex(
                f"s^T h_{i+1}",
                tex_to_color_map={
                    "s": A_PINK,
                    "h": A_GREEN,
                    f"{i+1}": A_UNKA,
                    "T": A_GREY,
                },
            )
            lbl.scale(1.25)
            lbl.next_to(rnn_encoder.cells[i].up_arrow, UP)
            score_lbl.add(lbl)

            bar = Rectangle(
                height=1e-6,
                width=0.5,
                stroke_width=4,
                stroke_color=WHITE,
                fill_color=A_LAVENDER,
                fill_opacity=0.75,
            )
            bar.next_to(lbl, UP)
            bars.add(bar)

        self.wait()

        s_anims, t_anims, h_anims = [], [], []
        for i in range(4):
            d_cpy = decoder_label.deepcopy()
            e_cpy = encoder_labels[i].deepcopy()

            s_anims.append(TransformMatchingShapes(d_cpy, score_lbl[i][0]))
            t_anims.append(Write(score_lbl[i][1]))
            h_anims.append(TransformMatchingShapes(e_cpy, score_lbl[i][2:]))

        self.play(*s_anims)
        self.play(*t_anims)
        self.play(*h_anims)
        self.wait()

        self.play(ShowCreation(arr_2))
        self.play(Write(bars), run_time=0.5)
        for i in range(4):
            new_bar = bars[i].deepcopy()
            new_bar.set_height(distrib[i], stretch=True)
            new_bar.move_to(bars[i], DOWN)

            self.play(Transform(bars[i], new_bar), run_time=0.5)
        self.wait()

        eq = Tex(
            r"\sum_{i=1}^{4} \sigma ( {s}^{T} {H} )_{i} {h}_{i}",
            tex_to_color_map={
                "{s}": A_PINK,
                "{T}": A_GREY,
                "{h}": A_GREEN,
                "{H}": A_GREEN,
                "{i}": A_UNKA,
                r"\sigma": A_ORANGE,
            },
        )
        eq[0][0].set_color(A_UNKA)  # 4
        eq[0][1].set_color(A_GREY)  # sum
        eq[0][2].set_color(A_UNKA)  # i
        eq[0][4].set_color(A_UNKA)  # 1
        eq.next_to(bars, UP, buff=0.75)

        for i in range(4):
            arr = Arrow(
                bars[i].get_top(),
                eq,
                max_width_to_Length_ratio=float("inf"),
                stroke_width=4,
                stroke_color=A_GREY,
                buff=0.25,
            )
            arr_2.add(arr)

            self.play(ShowCreation(arr), run_time=0.25)

        self.play(Write(eq[0]))

        score_anims, value_anims = [], []
        for i in range(4):
            score_anims += [
                TransformMatchingShapes(score_lbl[i][:2].deepcopy(), eq[3:5]),
                TransformMatchingShapes(score_lbl[i][2].deepcopy(), eq[5]),
                # Transform(score_lbl[i][3].deepcopy(), eq[4]),
            ]
            value_anims += [
                TransformMatchingShapes(encoder_labels[i][0][0].deepcopy(), eq[8]),
                Transform(encoder_labels[i][0][1].deepcopy(), eq[9]),
            ]
        self.play(*score_anims)
        self.play(Write(eq[1:3]), Write(eq[6:8]))
        self.play(*value_anims)
        self.wait()

        self.play(TransformFromCopy(eq, VMobject().move_to(decoder_label)))
        self.wait()

        self.embed()


class LSTMCell(VGroup):
    CONFIG = {
        "gate_width": 0.4,
        "circle_kwargs": {"fill_opacity": 0.5},
        "square_kwargs": {
            "fill_opacity": 1,
            "fill_color": BLACK,
            "stroke_color": A_GREY,
        },
        "arrow_kwargs": {"color": WHITE},
        "tanh_color": A_PINK,
        "sigmoid_color": A_AQUA,
        "arrow_end_buff": 0.05,
        "hidden_arrow_length": 1,
        "hidden_arrow_kwargs": {
            "max_width_to_length_ratio": float("inf"),
            "stroke_width": 7.5,
            "stroke_color": A_GREY,
            "buff": 0.125,
        },
        "hidden_tex_color_map": {
            "{c}": A_LAVENDER,
            "{x}": A_PINK,
            "{h}": A_GREEN,
            "{y}": A_BLUE,
            "{t}": A_YELLOW,
            "1": A_UNKA,
        },
    }

    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)

        self.circle_radius = self.gate_width / 1.75
        self.square_side_length = self.gate_width

        self.rect = RoundedRectangle(
            width=4.5, height=3, corner_radius=0.375, fill_color=GREY_E, fill_opacity=1
        )

        self.up_line = VGroup(Line(2.25 * LEFT, 2.25 * RIGHT))
        self.up_line.shift(1 * UP)

        self.down_line = VGroup(Line(2.25 * LEFT, 0.5 * RIGHT))
        self.down_line.shift(1 * DOWN)

        self.output_gate = VGroup(
            Line(
                0.5 * RIGHT + 1 * DOWN,
                0.5 * RIGHT + (0.5 + self.circle_radius) * DOWN,
                **self.arrow_kwargs,
            ),
            Circle(
                radius=self.circle_radius,
                color=self.sigmoid_color,
                **self.circle_kwargs,
            )
            .move_to(0.5 * RIGHT + 0.5 * DOWN)
            .add(Tex(r"\sigma").scale(0.625).move_to(0.5 * RIGHT + 0.5 * DOWN)),
            Arrow(
                (0.5 + self.circle_radius) * RIGHT + 0.5 * DOWN,
                1 * RIGHT + 0.5 * DOWN,
                buff=0,
                **self.arrow_kwargs,
            ),
        )

        gate = Square(side_length=self.square_side_length, **self.square_kwargs)
        gate.move_to(
            self.output_gate[2].get_end()
            + (self.arrow_end_buff + self.square_side_length / 2) * RIGHT
        )
        gate.add(Tex(r"\cross").scale(0.625).move_to(gate.get_center()))
        self.output_gate.add(gate)

        self.output_gate.add(
            Line(gate, gate.get_center()[0] * RIGHT + 1 * DOWN),
            Line(gate.get_center()[0] * RIGHT + 1 * DOWN, 2.25 * RIGHT + 1 * DOWN),
        )

        self.tanh_gate_1 = VGroup(
            Line(
                gate, gate.get_center()[0] * RIGHT + (0.375 - self.circle_radius) * UP
            ),
            Circle(
                radius=self.circle_radius,
                color=self.tanh_color,
                **self.circle_kwargs,
            )
            .move_to(gate.get_center()[0] * RIGHT + 0.375 * UP)
            .add(
                Tex(r"\mathrm{tanh}")
                .scale(0.375)
                .move_to(gate.get_center()[0] * RIGHT + 0.375 * UP)
            ),
            Line(
                gate.get_center()[0] * RIGHT + (0.375 + self.circle_radius) * UP,
                gate.get_center()[0] * RIGHT + 1 * UP,
            ),
        )

        self.forget_gate = VGroup(
            Line(
                (2.25 - 2.75 / 4) * LEFT + 1 * DOWN,
                (2.25 - 2.75 / 4) * LEFT + (0.5 + self.circle_radius) * DOWN,
            ),
            Circle(
                radius=self.circle_radius,
                color=self.sigmoid_color,
                **self.circle_kwargs,
            )
            .move_to((2.25 - 2.75 / 4) * LEFT + 0.5 * DOWN)
            .add(
                Tex(r"\sigma")
                .scale(0.625)
                .move_to((2.25 - 2.75 / 4) * LEFT + 0.5 * DOWN)
            ),
            Arrow(
                (2.25 - 2.75 / 4) * LEFT + (0.5 - self.circle_radius) * DOWN,
                (2.25 - 2.75 / 4) * LEFT
                + (1 - self.square_side_length / 2 - self.arrow_end_buff) * UP,
                buff=0,
            ),
            Square(side_length=self.square_side_length, **self.square_kwargs)
            .move_to((2.25 - 2.75 / 4) * LEFT + 1 * UP)
            .add(
                Tex(r"\cross").scale(0.625).move_to((2.25 - 2.75 / 4) * LEFT + 1 * UP)
            ),
        )

        self.input_gate = VGroup(
            Line(
                (2.25 - 2 * 2.75 / 4) * LEFT + 1 * DOWN,
                (2.25 - 2 * 2.75 / 4) * LEFT + (0.5 + self.circle_radius) * DOWN,
            ),
            Circle(
                radius=self.circle_radius,
                color=self.sigmoid_color,
                **self.circle_kwargs,
            )
            .move_to((2.25 - 2 * 2.75 / 4) * LEFT + 0.5 * DOWN)
            .add(
                Tex(r"\sigma")
                .scale(0.625)
                .move_to((2.25 - 2 * 2.75 / 4) * LEFT + 0.5 * DOWN)
            ),
            Line(
                (2.25 - 2 * 2.75 / 4) * LEFT + (0.5 - self.circle_radius) * DOWN,
                (2.25 - 2 * 2.75 / 4) * LEFT + 0.25 * UP,
            ),
            Arrow(
                (2.25 - 2 * 2.75 / 4) * LEFT + 0.25 * UP,
                (
                    2.25
                    - 3 * 2.75 / 4
                    + self.square_side_length / 2
                    + self.arrow_end_buff
                )
                * LEFT
                + 0.25 * UP,
                buff=0,
                **self.arrow_kwargs,
            ),
        )

        self.update_gate = VGroup(
            Line(
                (2.25 - 3 * 2.75 / 4) * LEFT + 1 * DOWN,
                (2.25 - 3 * 2.75 / 4) * LEFT + (0.5 + self.circle_radius) * DOWN,
            ),
            Circle(
                radius=self.circle_radius,
                color=self.tanh_color,
                **self.circle_kwargs,
            )
            .move_to((2.25 - 3 * 2.75 / 4) * LEFT + 0.5 * DOWN)
            .add(
                Tex(r"\mathrm{tanh}")
                .scale(0.375)
                .move_to((2.25 - 3 * 2.75 / 4) * LEFT + 0.5 * DOWN)
            ),
            Line(
                (2.25 - 3 * 2.75 / 4) * LEFT + (0.5 - self.circle_radius) * DOWN,
                (2.25 - 3 * 2.75 / 4) * LEFT
                + (0.25 - self.square_side_length / 2) * UP,
            ),
            Square(side_length=self.square_side_length, **self.square_kwargs)
            .move_to((2.25 - 3 * 2.75 / 4) * LEFT + 0.25 * UP)
            .add(
                Tex(r"\cross")
                .scale(0.625)
                .move_to((2.25 - 3 * 2.75 / 4) * LEFT + 0.25 * UP)
            ),
            Line(
                (2.25 - 3 * 2.75 / 4) * LEFT
                + (0.25 + self.square_side_length / 2) * UP,
                (2.25 - 3 * 2.75 / 4) * LEFT + (1 - self.square_side_length / 2) * UP,
            ),
            Square(side_length=self.square_side_length, **self.square_kwargs)
            .move_to((2.25 - 3 * 2.75 / 4) * LEFT + 1 * UP)
            .add(Tex(r"+").scale(0.625).move_to((2.25 - 3 * 2.75 / 4) * LEFT + 1 * UP)),
        )

        self.left_arrows = VGroup(
            Arrow(
                (2.25 + self.hidden_arrow_length) * LEFT + 1 * UP,
                2.25 * LEFT + 1 * UP,
                **self.hidden_arrow_kwargs,
            ),
            Arrow(
                (2.25 + self.hidden_arrow_length) * LEFT + 1 * DOWN,
                2.25 * LEFT + 1 * DOWN,
                **self.hidden_arrow_kwargs,
            ),
        )

        self.left_arrows.add(
            Tex(r"{c}_{{t}-1}", tex_to_color_map=self.hidden_tex_color_map)
            .scale(0.75)
            .move_to(self.left_arrows[0].get_start() + 0.375 * LEFT),
            Tex(r"{h}_{{t}-1}", tex_to_color_map=self.hidden_tex_color_map)
            .scale(0.75)
            .move_to(self.left_arrows[1].get_start() + 0.375 * LEFT),
        )

        self.right_arrows = VGroup(
            Arrow(
                2.25 * RIGHT + 1 * UP,
                (2.25 + self.hidden_arrow_length) * RIGHT + 1 * UP,
                **self.hidden_arrow_kwargs,
            ),
            Arrow(
                2.25 * RIGHT + 1 * DOWN,
                (2.25 + self.hidden_arrow_length) * RIGHT + 1 * DOWN,
                **self.hidden_arrow_kwargs,
            ),
        )

        self.right_arrows.add(
            Tex(r"{c}_{t}", tex_to_color_map=self.hidden_tex_color_map)
            .scale(0.75)
            .move_to(self.right_arrows[0].get_end() + 0.25 * RIGHT),
            Tex(r"{h}_{t}", tex_to_color_map=self.hidden_tex_color_map)
            .scale(0.75)
            .move_to(self.right_arrows[1].get_end() + 0.25 * RIGHT),
        )

        self.add(self.rect, self.up_line, self.down_line)
        self.add(self.output_gate, self.tanh_gate_1)
        self.add(self.forget_gate, self.input_gate, self.update_gate)
        self.add(self.left_arrows, self.right_arrows)

    def write(self, scene):
        scene.play(Write(self.rect))
        self.write_subpart(self.up_line, scene)
        self.write_subpart(self.down_line, scene)
        scene.play(FadeIn(self.left_arrows, RIGHT))
        scene.play(FadeIn(self.right_arrows, RIGHT))

        self.write_subpart(self.output_gate, scene)
        self.write_subpart(self.tanh_gate_1, scene)
        self.write_subpart(self.forget_gate, scene)
        self.write_subpart(self.input_gate, scene)
        self.write_subpart(self.update_gate, scene)

    def write_subpart(self, part, scene):
        for obj in part:
            if isinstance(obj, Circle) or isinstance(obj, Square):
                scene.play(GrowFromCenter(obj), run_time=0.5)
            else:
                scene.play(ShowCreation(obj), run_time=0.5)


class LSTMDemo(Scene):
    def construct(self):
        title = Text("Long Short-Term Memory", color=A_YELLOW)
        title.scale(1.5)
        title.shift(3 * UP)

        l = LSTMCell()
        l.scale(1.75)
        l.shift(0.5 * DOWN)

        self.play(Write(title))
        l.write(self)
        self.wait()

        self.embed()


class TitleColah(TitleScene):
    CONFIG = {
        "color": GREY_E,
        "text": "Chris Olah's LSTM Blog",
    }
