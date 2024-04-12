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
