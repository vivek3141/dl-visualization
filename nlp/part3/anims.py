from manimlib import *

"""
Scenes In Order:

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

EPS = 1e-6


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


class MNISTGroup(VGroup):
    def set_opacity(self, opacity):
        for img in self:
            img.set_opacity(opacity)


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
    CONFIG = {"color": None, "text": None}

    def construct(self):
        if self.text is None:
            raise NotImplementedError

        brect = Rectangle(
            height=FRAME_HEIGHT, width=FRAME_WIDTH, fill_opacity=1, color=self.color
        )

        title = TexText(self.text)
        title.scale(1.5)
        title.to_edge(UP)

        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)

        self.add(brect)
        self.play(FadeIn(rect, DOWN), Write(title), run_time=2)
        self.wait()


class MachineTranslation(Scene):
    def construct(self):
        title = Text("Machine Translation", color=A_YELLOW)
        title.scale(1.5)
        title.shift(3 * UP)

        self.play(Write(title))

        lm_rect = RoundedRectangle(
            height=3,
            width=4.5,
            fill_color=GREY_D,
            fill_opacity=1,
        )
        lm_rect.shift(0.5 * DOWN)

        lm_text = Text("Language\nModel")
        c1 = lm_text[:-5].get_center()
        c2 = lm_text[-5:].get_center()
        lm_text[-5:].move_to([c1[0], c2[1], 0])
        lm_text.scale(1.25)
        lm_text.move_to(lm_rect)

        in_text = Text("My name\nis Vivek")
        c1 = in_text[:10].get_center()
        c2 = in_text[11:].get_center()
        in_text[11:].move_to([c1[0], c2[1], 0])
        in_text.scale(1.25)
        in_text.shift(5.5 * LEFT + 0.5 * DOWN)

        arr1 = Arrow(
            in_text,
            lm_rect,
            stroke_color=A_PINK,
            max_width_to_Length_ratio=float("inf"),
            stroke_width=10,
        )

        out_text = Text("मेरा नाम\nविवेक है")
        c1 = out_text[:10].get_center()
        c2 = out_text[11:].get_center()
        out_text[11:].move_to([c1[0], c2[1], 0])
        out_text.shift(5.5 * RIGHT + 0.5 * DOWN)
        out_text.scale(1.25)

        arr2 = Arrow(
            lm_rect,
            out_text,
            stroke_color=A_PINK,
            max_width_to_Length_ratio=float("inf"),
            stroke_width=10,
        )

        self.play(Write(in_text), run_time=0.5)
        self.play(FadeIn(arr1, RIGHT), run_time=0.5)
        self.play(GrowFromCenter(VGroup(lm_rect, lm_text)), run_time=0.5)
        self.play(FadeIn(arr2, RIGHT), run_time=0.5)
        self.play(Write(out_text), run_time=0.5)
        self.wait()

        self.play(Uncreate(VGroup(in_text, out_text, lm_rect, lm_text, arr1, arr2)))

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

        output_text_copies = VGroup()
        for i in range(1, 4):
            prev_word_copy = output_text[i - 1].deepcopy()
            prev_word_copy.move_to(
                rnn_decoder.cells[i].down_arrow.get_start() + 0.5 * DOWN
            )
            output_text_copies.add(prev_word_copy)

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

        self.play(Uncreate(title))
        self.wait()

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
            Uncreate(output_text_copies),
        )

        self.embed()


class Softmax(Scene):
    def construct(self):
        input_vec = [-1.5, 2.7, 0.3, -0.5, 1.6]
        exp_vec = np.exp(input_vec)
        # output_vec = softmax(input_vec, axis=0)
        output_vec = [0.01, 0.68, 0.06, 0.03, 0.23]

        input_tex = Tex(
            f"[{', '.join([str(x) for x in input_vec])}]",
            tex_to_color_map={f"{i}": A_YELLOW for i in input_vec},
        )
        input_tex.scale(1.5)
        input_tex.shift(2.75 * UP)

        exp_tex = Tex(
            f"[{', '.join([f'{x:.2f}' for x in exp_vec])}]",
            tex_to_color_map={f"{x:.2f}": A_YELLOW for x in exp_vec},
        )
        exp_tex.scale(1.5)
        exp_tex.shift(0.5 * UP)

        softmax_tex = Tex(
            f"[{', '.join([f'{x:.2f}' for x in output_vec])}]",
            tex_to_color_map={f"{x:.2f}": A_YELLOW for x in output_vec},
        )
        softmax_tex.scale(1.5)
        softmax_tex.shift(1.75 * DOWN)

        axes = Axes(
            x_range=(0, 1),
            y_range=(0, 1),
            height=FRAME_HEIGHT - 3,
            width=FRAME_WIDTH - 3,
            x_axis_config={"include_ticks": False},
        )
        axes.shift(0.75 * DOWN)

        bars = VGroup()
        for i in range(len(output_vec)):
            bar = Rectangle(
                width=1 / (2 * len(output_vec)) * (FRAME_WIDTH - 3),
                height=EPS,
                # height=output_vec[i] * (FRAME_HEIGHT - 3),
                fill_color=A_LAVENDER,
                fill_opacity=1,
                stroke_opacity=1,
            )
            bar.move_to(axes.c2p((2 * i + 1) / (2 * len(output_vec))), DOWN)
            bars.add(bar)

        self.play(Write(input_tex))
        self.play(Write(axes))

        self.play(
            *[
                TransformFromCopy(input_tex[2 * i + 1], bars[i])
                for i in range(len(bars))
            ]
        )

        labels, stretch_anims = VGroup(), []
        for i in range(len(bars)):
            new_bar = bars[i].deepcopy()
            new_bar.set_height(output_vec[i] * (FRAME_HEIGHT - 3), stretch=True)
            new_bar.move_to(axes.c2p((2 * i + 1) / (2 * len(output_vec))), DOWN)

            label = Tex(f"{output_vec[i]:.2f}")
            label.next_to(new_bar, UP)
            labels.add(label)

            stretch_anims.append(Transform(bars[i], new_bar))
            stretch_anims.append(FadeIn(labels[i], UP))
        self.play(*stretch_anims)
        self.wait()

        plus, move_up_anims, move_back_anims = VGroup(), [], []
        for i in range(len(bars)):
            mask = i == (len(bars) - 1)
            p = Tex("+" if not mask else "= 1.00")
            p.move_to(axes.c2p((2 * (i + 1) + 0.25 * mask) / (2 * len(output_vec))))
            p.shift(4 * UP)
            plus.add(p)

            old_center = labels[i].get_center().copy()
            new_center = [old_center[0], p.get_center()[1], 0]
            move_up_anims.append(
                ApplyMethod(
                    labels[i].move_to,
                    new_center,
                )
            )

            move_back_anims.append(
                ApplyMethod(
                    labels[i].move_to,
                    old_center,
                )
            )

        self.play(*move_up_anims, FadeIn(plus, UP))
        self.wait()

        self.play(*move_back_anims, FadeOut(plus, UP))
        self.wait()

        self.play(Uncreate(VGroup(axes, bars, labels)))

        arr1 = Arrow(
            input_tex.get_bottom(),
            exp_tex.get_top(),
            stroke_width=10,
            max_width_to_Length_ratio=float("inf"),
            stroke_color=A_GREY,
        )
        lbl1 = Tex(
            "e^{x}",
            tex_to_color_map={"x": A_PINK, "e": A_GREEN},
        )
        lbl1.next_to(arr1, RIGHT, buff=0.5)

        arr2 = Arrow(
            exp_tex.get_bottom(),
            softmax_tex.get_top(),
            stroke_width=10,
            max_width_to_Length_ratio=float("inf"),
            stroke_color=A_GREY,
        )
        lbl2 = Tex(
            r"1/{\Sigma e^{x}}",
            tex_to_color_map={"x": A_PINK, "e": A_GREEN, r"\Sigma": A_GREY},
        )
        lbl2.next_to(arr2, RIGHT, buff=0.5)

        eq = Tex(
            r"\text{softmax}({x}) = {{e}^{x} \over \sum {e}^{x}}",
            tex_to_color_map={
                "{x}": A_PINK,
                "{e}": A_GREEN,
                r"\sum": A_GREY,
                r"\text{softmax}": A_AQUA,
            },
        )
        eq.scale(1.25)
        eq.shift(3 * DOWN)

        self.play(FadeIn(arr1, DOWN), Write(lbl1))
        self.play(FadeIn(exp_tex, DOWN))
        self.wait()

        self.play(FadeIn(arr2, DOWN), Write(lbl2))
        self.play(FadeIn(softmax_tex, DOWN))
        self.wait()

        group = VGroup(*[mob for mob in self.mobjects if isinstance(mob, VMobject)])
        self.play(group.shift, 0.25 * UP)
        self.play(Write(eq))
        self.wait()

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


class AttentionScores(Scene):
    def construct(self):
        english_sent = "The agreement on the European Economic Area was signed in August 1992 . <end>"
        english_sent.split(" ")

        french_sent = "L' accord sur la zone économique européenne a été signé en août 1992 . <end>"
        french_sent.split(" ")

        image = Image.open("img/attention_scores.png")
        image = image.convert("RGB")

        pixels = np.array(image.getdata())
        pixels = pixels.reshape((image.height, image.width, 3))

        HEIGHT = 15
        WIDTH = 14

        attention_scores = np.zeros((HEIGHT, WIDTH))

        for i in range(HEIGHT):
            for j in range(WIDTH):
                h = int((i + 0.5) * image.height / HEIGHT)
                w = int((j + 0.5) * image.width / WIDTH)

                block = pixels[h : h + 10, w : w + 10]
                avg = np.mean(block, axis=(0, 1))
                attention_scores[i][j] = np.mean(avg)

        grid = VGroup()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                color = attention_scores[i][j] / 255 * np.array([1, 1, 1])
                color = rgb_to_hex(color)
                rect = Rectangle(
                    height=1,
                    width=1,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0,
                )
                rect.move_to((i - HEIGHT / 2) * DOWN + (j - WIDTH / 2) * RIGHT)
                grid.add(rect)
        grid.scale(0.4)

        french_labels = VGroup()
        for i, word in enumerate(french_sent.split(" ")):
            label = Text(word)
            label.scale(0.5)
            label.next_to(grid[i * WIDTH], LEFT, buff=0.125)
            french_labels.add(label)

        english_labels = VGroup()
        for i, word in enumerate(english_sent.split(" ")):
            label = Text(word)
            label.scale(0.5)
            label.rotate(90 * DEGREES)
            label.next_to(grid[i], UP, buff=0.125)
            english_labels.add(label)

        VGroup(french_labels, english_labels, grid).center()

        self.play(Write(french_labels), Write(english_labels))
        self.play(Write(grid))
        self.wait()

        l1 = Line(
            grid[5].get_top(),
            grid[5 * WIDTH + 5].get_center(),
            color=A_YELLOW,
            stroke_width=8,
        )
        l2 = Line(
            grid[5 * WIDTH + 5].get_center(),
            grid[5 * WIDTH].get_left(),
            color=A_YELLOW,
            stroke_width=8,
        )

        self.play(ShowCreation(l1))
        self.play(ShowCreation(l2))
        self.wait()

        self.embed()

        self.play(Uncreate(l1), Uncreate(l2))

        l1 = Line(
            grid[7].get_top(),
            grid[8 * WIDTH + 7].get_center(),
            color=A_YELLOW,
            stroke_width=8,
        )
        l2 = Line(
            grid[8 * WIDTH + 7].get_center(),
            grid[8 * WIDTH].get_left(),
            color=A_YELLOW,
            stroke_width=8,
        )
        l3 = Line(
            grid[7 * WIDTH + 7].get_center(),
            grid[7 * WIDTH].get_left(),
            color=A_YELLOW,
            stroke_width=8,
        )

        self.play(ShowCreation(l1))
        self.play(ShowCreation(l2), ShowCreation(l3))
        self.wait()

        self.play(Uncreate(l1), Uncreate(l2), Uncreate(l3))

        l1 = Line(
            grid[8].get_top(),
            grid[9 * WIDTH + 8].get_center(),
            color=A_YELLOW,
            stroke_width=8,
        )
        l2 = Line(
            grid[9 * WIDTH + 8].get_center(),
            grid[9 * WIDTH].get_left(),
            color=A_YELLOW,
            stroke_width=8,
        )
        l3 = Line(
            grid[8 * WIDTH + 8].get_center(),
            grid[8 * WIDTH].get_left(),
            color=A_YELLOW,
            stroke_width=8,
        )

        self.play(ShowCreation(l1))
        self.play(ShowCreation(l2), ShowCreation(l3))
        self.wait()

        self.play(Uncreate(l1), Uncreate(l2), Uncreate(l3))
        self.wait()

        self.embed()


class Translations(Scene):
    def construct(self):
        source = Text(
            """This kind of experience is part of Disney's efforts to "extend the lifetime of its series and build new relationships with audiences via digital platforms that are becoming ever more important," he added.""",
        )
        source.scale(0.75)
        source.move_to((-FRAME_HEIGHT / 2 + 5 * FRAME_HEIGHT / 6) * UP + 1.5 * RIGHT)

        source_lbl = Text("Source", color=A_PINK)
        source_lbl.move_to(
            (4.25 + (FRAME_WIDTH / 2 - 4.25) / 2) * LEFT
            + (-FRAME_HEIGHT / 2 + 5 * FRAME_HEIGHT / 6) * UP
        )

        reference = Text(
            """Ce type d'experience entre dans le cadre des efforts de Disney pour « étendre la dur durée vie de ses séries et construire de nouvelles relations avec son public grâce à des plateformes numériques qui sont de plus en plus importantes », a-t-il ajout é"""
        )
        reference.scale(0.75)
        reference.move_to(-(FRAME_HEIGHT / 2 - 3 * FRAME_HEIGHT / 6) * UP + 1.5 * RIGHT)

        reference_lbl = Text("Reference", color=A_PINK)
        reference_lbl.move_to(
            (4.25 + (FRAME_WIDTH / 2 - 4.25) / 2) * LEFT
            + (-FRAME_HEIGHT / 2 + 3 * FRAME_HEIGHT / 6) * UP
        )

        output = Text(
            """Ce genre d'expérience fait partie des efforts de Disney pour « prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes », a-t-il ajouté."""
        )
        output.scale(0.75)
        output.move_to(-(FRAME_HEIGHT / 2 - 1 * FRAME_HEIGHT / 6) * UP + 1.5 * RIGHT)

        output_lbl = Text("Attention\nModel", color=A_PINK)
        c1 = output_lbl[:9].get_center()
        c2 = output_lbl[-5:].get_center()
        output_lbl[-5:].move_to([c1[0], c2[1], 0])
        output_lbl.move_to(
            (4.25 + (FRAME_WIDTH / 2 - 4.25) / 2) * LEFT
            + (-FRAME_HEIGHT / 2 + 1 * FRAME_HEIGHT / 6) * UP
        )

        l_left = Line(10 * UP, 10 * DOWN)
        l_left.shift(4.25 * LEFT)

        l1 = Line(10 * LEFT, 10 * RIGHT)
        l1.move_to((-FRAME_HEIGHT / 2 + FRAME_HEIGHT / 3) * UP)

        l2 = Line(10 * LEFT, 10 * RIGHT)
        l2.move_to((-FRAME_HEIGHT / 2 + 2 * FRAME_HEIGHT / 3) * UP)

        self.play(ShowCreation(l_left))
        self.play(ShowCreation(l1), ShowCreation(l2))
        self.play(Write(source_lbl), Write(source))
        self.play(Write(reference_lbl), Write(reference))
        self.play(Write(output_lbl), Write(output))
        self.wait()

        self.embed()
