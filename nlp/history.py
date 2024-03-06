from manimlib import *
import tiktoken

"""
Scenes In Order:

EmailModel
MNISTClassification
NextWordPrediction
DiceProbability
NGramModel
Inference
InferenceAlgorithms
AlexNet
NeuralLM
Pendulum
Extrapolation
RNNIntro
RNNTraining
RNNBackprop
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


class Dice(VMobject):
    def __init__(
        self,
        number,
        square_width=2.0,
        dot_radius=0.15,
        dot_color=A_UNKA,
        square_color=A_GREY,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.square = RoundedRectangle(
            height=square_width,
            width=square_width,
            corner_radius=0.375,
            color=square_color,
        )
        self.dots = VGroup()

        dot_kwargs = {"radius": dot_radius, "color": dot_color}
        if number == 1:
            self.dots.add(Dot(**dot_kwargs))
        elif number == 2:
            self.dots.add(
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * RIGHT),
            )
        elif number == 3:
            self.dots.add(
                Dot(**dot_kwargs),
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * RIGHT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * LEFT),
            )
        elif number == 4:
            self.dots.add(
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * RIGHT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * RIGHT),
            )
        elif number == 5:
            self.dots.add(
                Dot(**dot_kwargs),
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * RIGHT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * RIGHT),
            )
        elif number == 6:
            self.dots.add(
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * UP + 0.5 * RIGHT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * DOWN + 0.5 * RIGHT),
                Dot(**dot_kwargs).shift(0.5 * LEFT),
                Dot(**dot_kwargs).shift(0.5 * RIGHT),
            )

        self.add(self.square, self.dots)


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


class MNISTGroup(VGroup):
    def set_opacity(self, opacity):
        for img in self:
            img.set_opacity(opacity)


class WordDistribution(VMobject):
    def __init__(
        self,
        words,
        probs,
        max_bar_width=1.5,
        word_scale=1.0,
        prob_scale=1.0,
        prob_bar_color=A_LAVENDER,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prob_bars_small = VGroup()
        self.prob_bars_large = VGroup()
        self.words = VGroup()
        self.probs = VGroup()

        widths = np.array(probs) / max(probs)

        for i, (word, prob) in enumerate(zip(words, probs)):
            bar_small = Rectangle(
                height=0.5,
                width=0,
                fill_color=prob_bar_color,
                fill_opacity=1,
                stroke_width=0,
            )

            bar_large = Rectangle(
                height=0.5,
                width=prob * max_bar_width,
                fill_color=prob_bar_color,
                fill_opacity=1,
            )

            bar_small.move_to(FRAME_HEIGHT / 10 * i * DOWN, LEFT)
            bar_large.move_to(FRAME_HEIGHT / 10 * i * DOWN, LEFT)

            word_text = Text(word)
            word_text.scale(word_scale)
            word_text.move_to(bar_large.get_bounding_box_point(LEFT) + 1 * LEFT)

            prob_text = Text(f"{prob:.4f}")
            prob_text.scale(prob_scale)
            prob_text.move_to(bar_large.get_bounding_box_point(RIGHT) + 1 * RIGHT)

            self.prob_bars_small.add(bar_small)
            self.prob_bars_large.add(bar_large)
            self.words.add(word_text)
            self.probs.add(prob_text)

        self.add(self.prob_bars_small, self.prob_bars_large, self.words, self.probs)
        self.center()

    def write(self, scene, text_run_time=1.5, prob_run_time=0.75):
        scene.play(
            Write(self.words), Write(self.prob_bars_small), run_time=text_run_time
        )

        for i in range(len(self.prob_bars_small)):
            scene.play(
                ApplyMethod(self.prob_bars_small[i].become, self.prob_bars_large[i]),
                FadeIn(self.probs[i], RIGHT),
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

        self.add_arrow(
            self.sq.get_top(),
            self.sq.get_top() + self.arrow_length * UP,
        )
        self.up_arrow = self.arrows[-1]

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
    def __init__(self, n_cells=4, **kwargs):
        super().__init__(**kwargs)
        self.n_cells = n_cells
        self.cells = VGroup()
        for i in range(4):
            if i == 0:
                c = RNNCell(add_labels=False, left_most=True)
            elif i == 2:
                c = RNNCell(add_labels=False)
            elif i == 3:
                c = RNNCell(add_labels=False, right_most=True)
            else:
                c = RNNCell(add_labels=False)

            if i != 0:
                if i == 1:
                    left = self.cells[i - 1].arrows[0].get_center()
                else:
                    left = self.cells[i - 1].arrows[1].get_center()
                right = c.arrows[0].get_center()
                c.shift(left - right)
            self.cells.add(c)

        self.add(self.cells)
        self.center()


class EmailModel(Scene):
    def construct(self):
        prompt = Text("Write an email to my mom.")
        prompt.scale(1.25)
        prompt.shift(3 * UP)

        # Found the offset by averaging the bounding boxes
        model_sq = RoundedRectangle(width=12.5)
        model_sq.set_fill(GREY_D, 1)
        model_sq.shift(0.6576390599999999 * UP)

        question = Text("?")
        question.move_to(model_sq)
        question.scale(2.5)

        arr1 = Arrow(
            prompt,
            model_sq,
            stroke_color=RED_A,
            max_width_to_Length_ratio=float("inf"),
            stroke_width=10,
        )

        response = Text(
            "Hi Mom,\nHope you're doing well.\n"
            "Just wanted to see how you're feeling today."
            "\nLove, Vivek"
        )
        response.shift(2.5 * DOWN)

        arr2 = Arrow(
            model_sq,
            response,
            stroke_color=RED_A,
            max_width_to_Length_ratio=float("inf"),
            stroke_width=10,
        )

        VGroup(prompt, model_sq, question, arr1, response, arr2).center()

        self.play(Write(prompt))
        self.play(
            Write(arr1),
            Write(model_sq),
            Write(question),
            Write(arr2),
        )
        self.play(Write(response))
        self.wait()

        self.embed()


class MNISTClassification(Scene):
    def construct(self):
        np.random.seed(0)
        mnist = np.load("mnist_data.npy")

        cols = 8
        rows = 5
        mnist_images = MNISTGroup()
        for i in range(rows):
            for j in range(cols):
                mnist_images.add(
                    MNISTImage(mnist[i * cols + j].flatten())
                    .shift(
                        [
                            -FRAME_WIDTH / 2 + FRAME_WIDTH / (cols + 1) * (j + 1),
                            -FRAME_HEIGHT / 2 + FRAME_HEIGHT / (rows + 1) * (i + 1),
                            0,
                        ]
                    )
                    .scale(0.2)
                )

        accuracy = 0.9
        actual = 0
        labels = VGroup()
        for i in range(len(mnist_images)):
            if np.random.rand() < accuracy:
                labels.add(Checkmark().scale(1.5).move_to(mnist_images[i]))
                actual += 1
            else:
                labels.add(Exmark().scale(1.5).move_to(mnist_images[i]))

        accuracy_text = Tex(
            r"\text{Accuracy: }", f"{round(100 * actual/len(mnist_images), 1)}", r"\%"
        )
        accuracy_text.scale(1.5)
        accuracy_text.shift(3.25 * UP)

        self.play(Write(mnist_images))
        self.wait()

        self.play(ApplyMethod(mnist_images.set_opacity, 0.25))
        self.play(Write(labels))
        self.wait()

        self.play(ApplyMethod(VGroup(mnist_images, labels).shift, 0.4 * DOWN))
        self.play(Write(accuracy_text))
        self.wait()

        self.embed()


class NextWordPrediction(Scene):
    def construct(self):
        probs = [
            ("the", 0.1524),
            ("blue", 0.0859),
            ("falling", 0.0729),
            ("a", 0.0508),
            ("full", 0.034),
            ("dark", 0.0299),
            ("not", 0.028),
            ("clear", 0.0237),
            ("black", 0.0189),
        ]

        rect = RoundedRectangle(
            height=2.5,
            width=2.5,
            fill_color=GREY_D,
            fill_opacity=1,
        )
        rect.shift(0.5 * LEFT)
        p_theta = Tex(r"\mathbb{P}_{\theta}")
        p_theta.scale(2)
        p_theta.move_to(rect)

        self.play(Write(rect), Write(p_theta))
        self.wait()

        prompt = Text("The sky is")
        prompt.scale(1.5)
        prompt.shift(5 * LEFT)

        arr = Arrow(
            prompt,
            rect,
            stroke_color=A_PINK,
            max_width_to_Length_ratio=float("inf"),
            stroke_width=10,
        )

        brace = Brace(prompt)
        brace_tex = brace.get_tex(
            r"k = 3", tex_to_color_map={r"k": A_YELLOW, "3": A_UNKA}
        )
        brace.add(brace_tex)

        self.play(Write(prompt), Write(arr))
        self.play(Write(brace))
        self.wait()

        prob_bars = WordDistribution(*zip(*probs))
        prob_bars.shift(4.5 * RIGHT)

        min_left_point = min(
            [
                prob_bars.words[i].get_bounding_box_point(LEFT)[0]
                for i in range(len(probs))
            ]
        )

        prob_arrows = VGroup()
        for i in range(len(probs)):
            left_point = prob_bars.words[i].get_bounding_box_point(LEFT)
            left_point[0] = min_left_point - 0.25

            prob_arrows.add(
                Arrow(
                    rect.get_bounding_box_point(RIGHT),
                    left_point,
                    stroke_color=A_PINK,
                    max_width_to_Length_ratio=float("inf"),
                    stroke_width=5,
                    buff=0,
                )
            )

        self.play(Write(prob_arrows))
        prob_bars.write(self, prob_run_time=0.5)
        self.wait()

        anims = [
            ApplyMethod(prob_arrows[i].set_opacity, 0.25)
            for i in range(len(probs))
            if i != 1
        ]
        for i in range(len(probs)):
            if i == 1:
                continue

            anims += [
                ApplyMethod(prob_bars.words[i].set_opacity, 0.25),
                ApplyMethod(prob_bars.probs[i].set_opacity, 0.25),
                ApplyMethod(prob_bars.prob_bars_large[i].set_opacity, 0.25),
            ]

        self.play(*anims)
        self.wait()

        lm_text = Text("Language\nModel", color=A_UNKA)
        c1 = lm_text[:-5].get_center()
        c2 = lm_text[-5:].get_center()
        lm_text[-5:].move_to([c1[0], c2[1], 0])

        lm_text.move_to(rect)
        lm_text.shift(2 * UP)

        self.play(Write(lm_text))

        self.embed()


class DiceProbability(Scene):
    def construct(self):
        dice_grp = VGroup()
        for i in range(1, 7):
            d = Dice(
                i,
                square_width=2.0,
                dot_radius=0.15,
                dot_color=A_UNKA,
                square_color=A_GREY,
            )
            d.shift(FRAME_WIDTH / 7 * i * RIGHT)
            d.scale(0.75)
            dice_grp.add(d)
        dice_grp.center()

        two_three = dice_grp[1:3].deepcopy()
        two_three.center()
        two_three.shift(1.5 * UP)

        hline = Line(6.25 * LEFT, 6.25 * RIGHT)

        self.play(Write(dice_grp))
        self.wait()

        self.play(ApplyMethod(dice_grp.shift, 1.5 * DOWN), Write(hline))
        self.play(TransformFromCopy(dice_grp[1:3], two_three))
        self.wait()

        self.embed()


class NGramModel(Scene):
    def construct(self):
        eq = Tex(
            r"\mathbb{P}[" r"\text{blue} \ | \ \text{the sky is}]",
            r"= {C(\text{the }\text{sky is}\text{ }\text{blue}) \over C(\text{the }\text{sky is})}",
            tex_to_color_map={
                r"\text{blue}": A_BLUE,
                r"\text{the sky is}": A_UNKA,
                r"\text{sky is}": A_UNKA,
                r"\text{the }": A_UNKA,
                r"C": A_UNKB,
                r"\mathbb{P}": A_ORANGE,
            },
        )
        eq.scale(1.5)

        self.play(Write(eq[:6]))
        self.wait(0.5)

        self.play(Write(eq[6:]))
        self.wait()

        self.play(Transform(eq, eq.copy().scale(1 / 1.5).shift(3 * UP)))

        d_grp = VGroup()
        for i in range(2):
            for j in range(4):
                d_grp.add(Document().shift(i * 3 * UP + j * 3 * RIGHT))
        d_grp.center()
        d_grp.shift(0.75 * DOWN)

        d = Document()

        anims = [TransformFromCopy(d, d_grp[i]) for i in range(len(d_grp) - 1)] + [
            Transform(d, d_grp[-1], replace_mobject_with_target_in_scene=True)
        ]
        self.play(Write(d))
        self.play(*anims)
        self.wait()

        blue_prob = 0.3
        other_words = ["red", "green", "yellow", "orange"]
        other_word_color_map = {
            "red": A_RED,
            "green": A_GREEN,
            "yellow": A_YELLOW,
            "orange": A_ORANGE,
        }

        np.random.seed(15)
        texts_is, texts_was, texts_two = VGroup(), VGroup(), VGroup()
        is_anims, was_anims, two_anims = [], [], []

        for i in range(len(d_grp)):
            if np.random.rand() < blue_prob:
                text_is = TexText(
                    "the sky is blue",
                    tex_to_color_map={"blue": A_BLUE, "the sky is": A_UNKA},
                )
                text_was = TexText(
                    "the sky was blue",
                    tex_to_color_map={"blue": A_BLUE, "sky was": A_UNKA, "the": A_UNKA},
                )
                text_two = TexText(
                    "sky is blue",
                    tex_to_color_map={"blue": A_BLUE, "sky is": A_UNKA},
                )
            else:
                other_word = np.random.choice(other_words)
                text_is = TexText(
                    f"the sky is {other_word}",
                    tex_to_color_map={
                        other_word: other_word_color_map[other_word],
                        "the sky is": A_UNKA,
                    },
                )
                text_was = TexText(
                    f"the sky was {other_word}",
                    tex_to_color_map={
                        other_word: other_word_color_map[other_word],
                        "sky was": A_UNKA,
                        "the": A_UNKA,
                    },
                )
                text_two = TexText(
                    f"sky is {other_word}",
                    tex_to_color_map={
                        other_word: other_word_color_map[other_word],
                        "sky is": A_UNKA,
                    },
                )

            curr_line = d_grp[i].lines[np.random.randint(7)]

            text_is.move_to(curr_line)
            text_is.scale(0.45)

            text_was.move_to(curr_line)
            text_was.scale(0.45)

            text_two.move_to(curr_line)
            text_two.scale(0.45)

            texts_is.add(text_is)
            texts_was.add(text_was)
            texts_two.add(text_two)

            is_anims.append(
                Transform(curr_line, text_is, replace_mobject_with_target_in_scene=True)
            )
            was_anims.append(
                TransformMatchingShapes(
                    text_is, text_was, replace_mobject_with_target_in_scene=True
                )
            )
            two_anims += [
                Uncreate(text_was[0]),
                TransformMatchingShapes(
                    text_was[1:], text_two, replace_mobject_with_target_in_scene=True
                ),
            ]

        self.play(*is_anims)
        self.wait()

        self.play(*was_anims)
        self.wait()

        b = Brace(eq[-1], RIGHT)
        b_tex = b.get_tex(r"k = 3", tex_to_color_map={r"k": A_YELLOW, "3": A_UNKA})
        b_tex.add_background_rectangle(buff=0.1)

        self.play(Write(b), Write(b_tex))
        self.wait()

        self.play(Uncreate(b), Uncreate(b_tex))

        eq2 = Tex(
            r"\mathbb{P}[" r"\text{blue} \ | \ \text{the sky is}]",
            r"= {C(\text{sky is}\text{ }\text{blue}) \over C(\text{sky is})}",
            tex_to_color_map={
                r"\text{blue}": A_BLUE,
                r"\text{the sky is}": A_UNKA,
                r"\text{sky is}": A_UNKA,
                r"C": A_UNKB,
                r"\mathbb{P}": A_ORANGE,
            },
        )
        eq2.shift(3 * UP)

        self.play(
            Transform(eq[:9], eq2[:9]),
            FadeOut(eq[9]),
            Transform(eq[10:15], eq2[9:14]),
            Uncreate(eq[15]),
            Transform(eq[16:], eq2[14:]),
        )
        self.remove(eq)
        self.add(eq2)

        self.play(*two_anims)
        self.wait()

        eq3 = Tex(
            r"\mathbb{P}[" r"\text{blue} \ | \ \text{the sky is}]",
            r"\approx {C(\text{sky is}\text{ }\text{blue}) \over C(\text{sky is})}",
            tex_to_color_map={
                r"\text{blue}": A_BLUE,
                r"\text{the sky is}": A_UNKA,
                r"\text{sky is}": A_UNKA,
                r"C": A_UNKB,
                r"\mathbb{P}": A_ORANGE,
            },
        )
        eq3.shift(3 * UP)

        self.play(TransformMatchingTex(eq2, eq3))
        self.wait()

        self.play(
            FadeOut(
                VGroup(
                    *[i for i in self.mobjects if isinstance(i, VMobject) and i != eq3]
                ),
                DOWN,
            )
        )

        eq4 = Tex(
            r"\mathbb{P}[" r"w_i \ | \ w_{i-3}w_{i-2}w_{i-1}]",
            r"\approx {C(w_{i-2}w_{i-1}w_i) \over C(w_{i-2}w_{i-1})}",
            tex_to_color_map={
                r"w_i": A_BLUE,
                r"w_{i-3}": A_UNKA,
                r"w_{i-2}": A_UNKA,
                r"w_{i-1}": A_UNKA,
                r"C": A_UNKB,
                r"\mathbb{P}": A_ORANGE,
            },
        )
        eq4.scale(1.25)

        self.play(
            TransformFromCopy(eq3[:2], eq4[:2]),
            TransformFromCopy(eq3[3], eq4[3]),
            TransformFromCopy(eq3[5:9], eq4[7:11]),
            TransformFromCopy(eq3[11:14], eq4[14:17]),
            TransformFromCopy(eq3[15:], eq4[19:]),
        )
        self.play(
            TransformFromCopy(eq3[2], eq4[2]), TransformFromCopy(eq3[10], eq4[13])
        )
        self.play(TransformFromCopy(eq3[4][:3], eq4[4]))
        self.play(
            TransformFromCopy(eq3[4][3:6], eq4[5]),
            TransformFromCopy(eq3[9][:3], eq4[11]),
            TransformFromCopy(eq3[14][:3], eq4[17]),
        )
        self.play(
            TransformFromCopy(eq3[4][6:], eq4[6]),
            TransformFromCopy(eq3[9][3:], eq4[12]),
            TransformFromCopy(eq3[14][3:], eq4[18]),
        )
        self.remove(eq3)
        self.add(eq4)

        title = Text("N-Gram Language Model", color=A_VIOLET)
        title.scale(1.5)
        title.shift(3 * UP)

        self.play(FadeOut(eq3, UP), Write(title))
        self.wait()

        b = Brace(eq4)
        b_tex = b.get_tex(r"\text{Trigram Model}")

        self.play(Write(b), Write(b_tex))
        self.wait()

        self.play(Uncreate(b), Uncreate(b_tex))

        eq5 = Tex(
            r"\mathbb{P}[" r"w_i \ | \ w_{i-3}w_{i-2}w_{i-1}]",
            r"\approx {C(w_{i-2}w_{i-1}w_i) + 1 \over C(w_{i-2}w_{i-1}) + |V|}",
            tex_to_color_map={
                r"w_i": A_BLUE,
                r"w_{i-3}": A_UNKA,
                r"w_{i-2}": A_UNKA,
                r"w_{i-1}": A_UNKA,
                r"C": A_UNKB,
                r"\mathbb{P}": A_ORANGE,
                r"V": MAROON_A,
                r"1": A_UNKA,
            },
        )
        eq5.scale(1.25)

        self.play(
            ApplyMethod(eq4[:14].move_to, eq5[:14]),
            ApplyMethod(eq4[14][0].move_to, eq5[14][0]),
            Transform(eq4[14][1:], eq5[16]),
            ApplyMethod(eq4[15:].move_to, VGroup(eq5[17:21], eq5[21][0])),
        )
        self.play(
            Write(eq5[15]), Write(eq5[14][1:]), Write(eq5[21][1:]), Write(eq5[22:])
        )
        self.wait()

        self.embed()


class NGramInference(Scene):
    def construct(self):
        enc = tiktoken.encoding_for_model("davinci")

        raw_text = "The most beautiful proof in math is \nthe only one with the other hand , \nthe same way to determine the value \nof the United States had been a man \nof the American League pennant , \nand his sister's a good time ."
        next_probs = np.load("next_probs.npy")

        def get_next_probs(idx):
            new_words, new_probs = [], []
            for w, p in next_probs[idx]:
                new_words.append(enc.decode([int(w)]).strip())
                new_probs.append(float(p))
            return new_words[:7], new_probs[:7]

        prompt = Text("The most beautiful proof in math is")
        prompt.scale(1.25)
        prompt.shift(3 * UP)

        self.play(Write(prompt))

        prob_bars = WordDistribution(*get_next_probs(0), max_bar_width=10)
        prob_bars.shift(0.5 * DOWN)

        prob_bars.write(self)
        self.wait()

        l = Line(10 * UP, 10 * DOWN)
        l.shift(2 * RIGHT)

        n_gram_text = Text("Trigram Model", color=A_VIOLET)
        n_gram_text.scale(1.25)
        n_gram_text.shift(4.5 * RIGHT + 3 * UP)

        new_prob_bars = WordDistribution(
            *get_next_probs(0), max_bar_width=1.5, word_scale=0.75, prob_scale=0.75
        )
        new_prob_bars.shift(4.5 * RIGHT + 0.5 * DOWN)
        new_prob_bars.scale(0.9)

        self.remove(prob_bars.prob_bars_small)
        prob_bars.remove(prob_bars.prob_bars_small)

        self.play(
            ApplyMethod(prompt.become, prompt.copy().scale(1 / 1.25).shift(2.5 * LEFT)),
            Transform(prob_bars.words, new_prob_bars.words),
            Transform(prob_bars.probs, new_prob_bars.probs),
            Transform(prob_bars.prob_bars_large, new_prob_bars.prob_bars_large),
            Write(l),
            Write(n_gram_text),
        )
        self.wait()

        text = TexText(
            *[i.replace("\n", r"\\") + " " for i in raw_text.split(" ")], alignment=""
        )
        text.move_to(prompt, UP + LEFT)
        self.remove(prob_bars.prob_bars_small)

        words = raw_text.split(" ")
        for i in range(7, len(words)):
            if i > 15:
                run_time = 0.5
            else:
                run_time = 1

            next_word = words[i]
            prob_word_obj = None

            for word_obj in prob_bars.words:
                if word_obj.text.strip() == next_word.strip():
                    prob_word_obj = word_obj
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
            new_prob_word_obj.move_to(text[i])

            self.play(
                TransformFromCopy(prob_word_obj, new_prob_word_obj),
                run_time=run_time,
            )
            self.wait(0.5 * run_time)

            new_dist = WordDistribution(
                *get_next_probs(i - 6),
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

        self.embed()


class InferenceAlgorithms(Scene):
    def construct(self):
        probs = [
            ("a", 0.08705387523141485),
            ("the", 0.07940873709043825),
            ("not", 0.05225144958160702),
            ("to", 0.027861431060304042),
            (",", 0.024988107409824458),
            ("that", 0.023519410891033286),
            ("in", 0.017348392643645783),
        ]

        l = Line(10 * UP, 10 * DOWN)
        self.add(l)

        top_k_head = Tex(
            r"\text{Top }k\text{-sampling}",
            tex_to_color_map={r"k": A_YELLOW},
        )
        top_k_head.scale(1.5)
        top_k_head.shift(FRAME_WIDTH / 4 * LEFT + 3.125 * UP)

        cot_head = Text("Chain-of-Thought")
        cot_head.scale(1.5)
        cot_head.shift(FRAME_WIDTH / 4 * RIGHT + 3.125 * UP)

        prob_bars = WordDistribution(*zip(*probs), prob_bar_color=A_AQUA)
        prob_bars.shift(FRAME_WIDTH / 4 * LEFT + 0.5 * DOWN)

        rect = SurroundingRectangle(
            VGroup(
                prob_bars.prob_bars_large[3:],
                prob_bars.words[3:],
                prob_bars.probs[3:],
            ),
            fill_color=BLACK,
            fill_opacity=0.75,
            stroke_width=0,
            stroke_opacity=0,
            stroke_color=BLACK,
        )

        self.play(Write(l), Write(top_k_head))
        prob_bars.write(self, text_run_time=0.5, prob_run_time=0.25)
        self.play(Write(rect))
        self.wait()

        cot_texts = [
            "Roger has 5 tennis balls.\nHe buys 2 more cans"
            " of\ntennis balls. Each can has 3\ntennis balls."
            " How many tennis\nballs does he have now?",
            "Roger started with 5 balls.\n2 cans of 3 tennis balls\n"
            "each is 6 tennis balls",
            "5 + 6 = 11",
            "The answer is 11.",
        ]
        offsets = [0, 2.125, 3.625, 4.875]

        texts, arrows = VGroup(), VGroup()
        for i in range(len(cot_texts)):
            text = TexText(cot_texts[i].replace("\n", r"\\"))
            text.shift(offsets[i] * DOWN)
            text.scale(0.5)
            texts.add(text)

            if i != 0:
                arrows.add(
                    Arrow(
                        texts[i - 1].get_bottom(),
                        texts[i].get_top(),
                        stroke_color=A_PINK,
                        max_width_to_Length_ratio=float("inf"),
                        stroke_width=10,
                    )
                )

        grp = VGroup(texts, arrows)
        grp.center()
        grp.shift(FRAME_WIDTH / 4 * RIGHT + 0.5 * DOWN)

        self.play(Write(cot_head))
        for i in range(len(texts)):
            if i != 0:
                self.play(
                    FadeIn(texts[i], DOWN), FadeIn(arrows[i - 1], DOWN), run_time=1
                )
            else:
                self.play(FadeIn(texts[i], DOWN), run_time=1)
        self.wait()

        self.embed()


class AlexNet(Scene):
    def construct(self):
        pass


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
            FadeOut(words, DOWN), FadeOut(rnn.cells[0], DOWN), FadeOut(rnn.cells[2:], DOWN)
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


class RNNBackprop(Scene):
    def construct(self):
        self.embed()
