from manimlib import *
import tiktoken
import scipy

"""
Scenes In Order:

EmailModel
MNISTClassification
NextWordPrediction
DiceProbability
NGramModel
Inference
InferenceAlgorithms
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


class TitleU(TitleScene):
    CONFIG = {"color": "#200f21", "text": "Upcoming"}


class TitleChatGPT(TitleScene):
    CONFIG = {"color": "#5c608f", "text": ["ChatGPT", "The Generated Animation"]}


class Title3b1b(TitleScene):
    CONFIG = {
        "color": GREY_E,
        "text": "3blue1brown",
        "tex_to_color_map": {"blue": BLUE, "brown": "#CD853F"},
    }


class Timeline(Scene):
    CONFIG = {
        "color": GREY_E,
        "text": "3blue1brown",
        "tex_to_color_map": {"blue": BLUE, "brown": "#CD853F"},
        "EPS_R": 0.1,
        "image_paths": [
            "img/nlp.png",
            "img/n_gram.png",
            "img/rnn.png",
            "img/attention.png",
            "img/transformer.png",
        ],
        "zoomed_out_point": np.array([-15, -5.131797, 0.0]),
    }

    def construct(self):
        brect = Rectangle(
            height=10 * FRAME_HEIGHT,
            width=10 * FRAME_WIDTH,
            fill_opacity=1,
            color=self.color,
        )

        title = TexText(
            self.text if isinstance(self.text, str) else self.text[0],
            tex_to_color_map=self.tex_to_color_map,
        )
        title.scale(1.5)
        title.to_edge(UP)

        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)

        t_img = ImageMobject("img/transformer.png", height=5.95)
        t_img.move_to(rect)

        self.add(brect, rect, title, t_img)

        left_most_point = 30 * LEFT
        up_most_point = rect.get_center()[1] * UP
        down_most_point = 10 * DOWN
        gap = RIGHT * (rect.get_center() - left_most_point)[0] / 4
        t = ValueTracker(0)  # Max Value of 1

        control_points = left_most_point + np.array(
            [
                i * gap + (i & 1 ^ 1) * up_most_point + (i & 1) * down_most_point
                for i in range(5)
            ]
        )

        spline = scipy.interpolate.CubicSpline(
            control_points[:, 0],
            control_points[:, 1],
            bc_type="clamped",
        )

        x_min, x_max = control_points[0][0], control_points[-1][0]
        curve = FunctionGraph(
            spline,
            x_range=[x_min, x_max, 0.01],
            stroke_width=30,
            color=WHITE,
        )

        def curve_updater(curve):
            t_val = t.get_value()
            x = x_min + (1 - t_val) * (x_max - x_min)
            new_curve = FunctionGraph(
                spline,
                x_range=[x, x_max],
                stroke_width=30,
                color=WHITE,
            )
            curve.become(new_curve)

        curve.add_updater(curve_updater)

        def compute_alpha(n):
            t_val = t.get_value()
            curr_bin = ((3 - n) / 3) - self.EPS_R

            if curr_bin <= t_val <= curr_bin + self.EPS_R:
                return (t_val - curr_bin) / self.EPS_R
            elif t_val > curr_bin + self.EPS_R:
                return 1
            else:
                return 0

        def rect_updater(rect):
            alpha = compute_alpha(rect.update_n_value)
            new_rect = ScreenRectangle(height=6 * alpha)
            new_rect.move_to(rect)
            rect.become(new_rect)

        def img_updater(img):
            alpha = compute_alpha(img.update_n_value)
            new_img = ImageMobject(
                self.image_paths[img.update_n_value], height=5.95 * alpha
            )
            new_img.move_to(img)
            img.become(new_img)

        rects, images = VGroup(), Group()
        for n, cp in enumerate(control_points):
            r = ScreenRectangle(height=6)
            r.update_n_value = n
            r.move_to(cp)
            r.add_updater(rect_updater)

            rects.add(r)

            img = ImageMobject(self.image_paths[n], height=5.95)
            img.update_n_value = n
            img.move_to(cp)
            img.add_updater(img_updater)
            images.add(img)

        zoomed_out_frame = self.camera.frame.copy()
        zoomed_out_frame.set_width(3 * FRAME_WIDTH)
        zoomed_out_frame.move_to(self.zoomed_out_point)

        self.add(curve, rects, images)

        self.play(
            AnimationGroup(
                Transform(self.camera.frame, zoomed_out_frame, run_time=10),
                ApplyMethod(t.set_value, 1, run_time=10),
                lag_ratio=DEFAULT_LAGGED_START_LAG_RATIO,
            )
        )
        self.wait()

        for i in range(5):
            new_frame = self.camera.frame.copy()
            new_frame.set_width(images[i].get_width())
            new_frame.move_to(images[i])

            self.play(
                Transform(self.camera.frame, new_frame, run_time=5),
            )
            self.wait()

            self.play(
                Transform(self.camera.frame, zoomed_out_frame, run_time=5),
            )
            self.wait()

        self.embed()


class NLP(Scene):
    def construct(self):
        MAIN_GAP = 4
        MAIN_COLOR = WHITE

        SUB_SCALE = 0.75
        SUB_COLOR = A_GREY

        arrow_kwargs = {
            "stroke_color": A_PINK,
            "max_width_to_Length_ratio": float("inf"),
            "stroke_width": 5,
        }

        title = Text("Natural Language Processing (NLP)", color=A_YELLOW)
        title.scale(1.5)
        title.shift(3 * UP)

        speech = Text("Speech", color=MAIN_COLOR)
        speech.shift(MAIN_GAP * RIGHT + 0.5 * UP)

        tts = TexText(r"Text-To-Speech\\(TTS)", color=SUB_COLOR)
        tts.scale(SUB_SCALE)
        tts.next_to(speech, DOWN)
        tts.shift(1.5 * DL)

        asr = TexText(r"Automatic Speech\\Recognition (ASR)", color=SUB_COLOR)
        asr.scale(SUB_SCALE)
        asr.next_to(speech, DOWN)
        asr.shift(1.5 * DR)

        speech_arrows = VGroup(
            Arrow(speech, tts, **arrow_kwargs),
            Arrow(speech, asr, **arrow_kwargs),
        )

        lm = TexText(r"Language\\Modeling", color=MAIN_COLOR)
        lm.shift(MAIN_GAP * LEFT + 0.5 * UP)

        tr = TexText(r"Machine\\Translation", color=SUB_COLOR)
        tr.scale(SUB_SCALE)
        tr.next_to(lm, DOWN)
        tr.shift(DL + 0.5 * LEFT)

        ts = TexText(r"Text\\Summarization", color=SUB_COLOR)
        ts.scale(SUB_SCALE)
        ts.next_to(lm, DOWN)
        ts.shift(2 * DOWN)

        qa = TexText(r"Question\\Answering", color=SUB_COLOR)
        qa.scale(SUB_SCALE)
        qa.next_to(lm, DOWN)
        qa.shift(DR + 0.5 * RIGHT)

        lm_arrows = VGroup(
            Arrow(lm, tr, **arrow_kwargs),
            Arrow(lm, ts, **arrow_kwargs),
            Arrow(lm, qa, **arrow_kwargs),
        )

        grounding = Text("Grounding", color=MAIN_COLOR)
        grounding.shift(0.5 * UP)

        ti = TexText(r"Text to Image", color=SUB_COLOR)
        ti.scale(SUB_SCALE)
        ti.next_to(grounding, DOWN)
        ti.shift(DOWN)

        grounding_arrows = VGroup(
            Arrow(grounding, ti, **arrow_kwargs),
        )

        category_arrows = VGroup(
            Arrow(title, speech, **arrow_kwargs),
            Arrow(title, lm, **arrow_kwargs),
            Arrow(title, grounding, **arrow_kwargs),
        )

        self.add(title, speech, tts, asr, lm, tr, ts, qa, grounding, ti)
        self.add(speech_arrows, lm_arrows, grounding_arrows, category_arrows)
        self.wait()

        self.play(Indicate(speech))
        self.play(Indicate(tts), run_time=0.5)
        self.play(Indicate(asr), run_time=0.5)
        self.wait()

        self.play(Indicate(lm))
        self.play(Indicate(tr), run_time=0.5)
        self.play(Indicate(ts), run_time=0.5)
        self.play(Indicate(qa), run_time=0.5)
        self.wait()

        self.play(Indicate(grounding))
        self.play(Indicate(ti), run_time=0.5)
        self.wait()

        arr = CurvedArrow(
            tr.get_bounding_box_point(RIGHT),
            tts.get_bounding_box_point(UP) + 0.25 * UL,
            angle=-TAU / 4,
            color=RED,
            stroke_width=10,
            max_width_to_length_ratio=float("inf"),
            buff=0.375,
        )

        img = ImageMobject("img/slt.jpeg")
        img.scale(0.5)
        img.shift(2 * LEFT)
        img.add_background_rectangle(
            stroke_width=8, stroke_opacity=1, opacity=0, color=A_GREY
        )

        lbl = Text("Spoken Language Translation")
        lbl.next_to(img, UP)

        img_grp = Group(lbl, img)
        img_grp.add_background_rectangle(opacity=0.875)

        self.play(
            ShowCreation(arr, run_time=2.5),
            GrowFromCenter(img_grp, run_time=1),
        )
        self.wait()

        self.play(Uncreate(arr), Uncreate(img_grp))
        self.wait()

        self.embed()


class StolenPainting(Scene):
    def construct(self):
        text = Text("The stolen painting was found by a tree")
        text.scale(1.5)
        text.shift(3 * UP)

        l = Line(10 * LEFT, 10 * RIGHT)
        l.next_to(text, DOWN)

        image = ImageMobject("img/evil_tree.jpg")
        image.scale(1.675)
        image.move_to(l, UP)
        image.shift(0.025 * DOWN)

        svg = SVGMobject("img/evil_tree.svg")
        svg.set_opacity(0.375)
        svg.apply_function(lambda p: [p[0], -p[1], 0])

        svg.scale(2 * 1.675)
        svg.move_to(l, UP)
        svg.shift(0.15 * UL)

        self.play(Write(text), Write(l))
        self.play(Write(svg), FadeIn(image), run_time=10)
        self.remove(svg)
        self.wait()

        self.embed()


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
        mnist = np.load("data/mnist_data.npy")

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
        next_probs = np.load("n_gram/next_probs.npy")

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

        n_gram_text = Text("Trigram Model", color=A_YELLOW)
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
