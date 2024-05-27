from manimlib import *
from scipy.optimize import fsolve

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


class BiasVariance(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 10],
            y_range=[0, 5],
            axis_config={
                "include_ticks": False,
                "stroke_width": 6,
            },
            height=FRAME_HEIGHT - 2,
            width=FRAME_WIDTH - 3,
        )

        func = lambda x: 0.2 * (x - 5) ** 2 + 1
        x1 = fsolve(lambda x: func(x) - 5, 2)[0]
        x2 = fsolve(lambda x: func(x) - 5, 8)[0]

        curve = axes.get_graph(
            func,
            color=A_AQUA,
            stroke_width=6,
            x_range=[x1, x2],
        )

        EPS = 0.001

        def add_tip_at(x, func, tip_len=0.2, rotate=True):
            deriv = (func(x + EPS) - func(x)) / EPS
            angle = np.arctan(deriv) + PI / 2 + rotate * PI

            offset1 = tip_len * (rotation_about_z(angle) @ (DOWN + LEFT))
            offset2 = tip_len * (rotation_about_z(angle) @ (DOWN + RIGHT))

            point = axes.c2p(x, func(x))
            tip = VGroup(
                Line(point, point + offset1),
                Line(point, point + offset2),
            )
            tip.set_stroke(color=A_AQUA, width=6)

            return tip

        tip1 = add_tip_at(x1, func, rotate=False)
        tip2 = add_tip_at(x2, func, rotate=True)
        curve.add(tip1, tip2)

        x_label = Text("Model Complexity")
        x_label.scale(1.25)
        x_label.next_to(axes.x_axis.get_bottom(), DOWN, buff=0.25)

        y_label = Text("Error")
        y_label.scale(1.25)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis.get_left(), LEFT, buff=0.25)

        line = DashedLine(axes.c2p(5, 0), axes.c2p(5, 5), positive_space_ratio=0.25)

        opt = Text("Optimal")
        opt.scale(0.75)
        opt.rotate(90 * DEGREES)
        opt.next_to(line, RIGHT, buff=0.25)
        opt.shift(2 * UP)

        grp = VGroup(
            axes,
            curve,
            x_label,
            y_label,
            line,
            opt,
        )
        grp.center()

        self.play(ShowCreation(axes))
        self.play(
            ShowCreation(curve),
            FadeIn(x_label, UP),
            FadeIn(y_label, LEFT),
        )
        self.play(
            ShowCreation(line),
            FadeIn(opt, RIGHT),
        )
        self.wait()

        self.embed()


class MMLU(Scene):
    def construct(self):
        data = [
            (1.5 + 0.05, ("GPT-2 XL", 32.4)),  # Add 0.05 for better spacing
            (11.0, ("UnifiedQA", 48.9)),
            (70.0, ("Chinchilla", 75)),
            (175.0, ("GPT-3", 53.9)),
            (280.0, ("Gopher", 60.0)),
            (540.0, ("Flan-U-PaLM", 74.10)),
            (1760.0, ("GPT-4", 86.40)),
            (1760.0, ("Gemini Ultra", 90)),
        ]
        data = [(math.log10(x), y) for x, y in data]

        axes = Axes(
            x_range=[0, 3.5],
            y_range=[0, 110],
            axis_config={
                "include_ticks": False,
                "stroke_width": 6,
            },
            height=FRAME_HEIGHT - 2,
            width=FRAME_WIDTH - 3,
            y_axis_config={
                "include_tip": False,
            },
        )
        axes.y_axis.add_numbers([25, 50, 75, 100], font_size=36)

        x_axis_labels = VGroup()
        for label in [1, 10, 100, 1000]:
            x_label = Text(f"{label}", font_size=36)
            x_label.next_to(axes.c2p(math.log10(label), 0), DOWN)
            x_axis_labels.add(x_label)
        axes.x_axis.add(x_axis_labels)

        lines = VGroup()
        for i in range(25, 101, 25):
            line = Line(
                axes.c2p(0, i),
                axes.c2p(3.5, i),
                stroke_width=3,
                stroke_color=A_GREY,
                stroke_opacity=0.25,
            )
            lines.add(line)

        x_label = Text("Model Size (Billion Parameters)")
        x_label.scale(1.25)
        x_label.next_to(axes.x_axis.get_bottom(), DOWN, buff=0.25)

        y_label = Text("MMLU (Average %)")
        y_label.scale(1.25)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis.get_left(), LEFT, buff=0.25)

        grp = VGroup(axes, x_label, y_label, lines)
        grp.center()

        self.play(ShowCreation(axes), ShowCreation(lines))
        self.play(FadeIn(x_label, UP), FadeIn(y_label, LEFT))

        points = VGroup()
        for x, y in data:
            point = Dot(axes.c2p(x, y[1]), color=A_LAVENDER)
            label = Text(y[0]).scale(0.5)
            label.next_to(point, UP)
            label.set_color(A_LAVENDER)

            points.add(VGroup(point, label))

        self.play(GrowFromCenter(points[0]))
        for i in range(1, len(points)):
            l = Line(
                points[i - 1][0].get_center(),
                points[i][0].get_center(),
                color=A_LAVENDER,
                stroke_width=6,
            )

            self.play(
                AnimationGroup(
                    ShowCreation(l, rate_func=linear),
                    GrowFromCenter(points[i], rate_func=linear),
                    lag_ratio=DEFAULT_ANIMATION_LAG_RATIO,
                ),
            )
        self.wait()

        self.embed()


class Block(VGroup):
    CONFIG = {
        "text": "",
        "text_scale": 0.75,
        "rect_width": 4.0,
        "rect_height": 0.75,
        "rect_corner_radius": 0.125,
        "fill_color": GREY_D,
        "fill_opacity": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = RoundedRectangle(
            width=self.rect_width,
            height=self.rect_height,
            fill_color=self.fill_color,
            fill_opacity=self.fill_opacity,
            corner_radius=self.rect_corner_radius,
        )
        self.text = Text(self.text)
        self.text.scale(self.text_scale)
        self.text.move_to(self.block)

        self.add(self.block, self.text)


class TransformerBlock(Block):
    CONFIG = {
        "text": "Transformer Block",
    }


class PositionalEncodingBlock(Block):
    CONFIG = {
        "text": "Positional Encoding",
        "text_scale": 0.5,
        "rect_height": 0.375,
        "rect_width": 2.5,
    }


class SoftmaxBlock(Block):
    CONFIG = {
        "text": "Softmax",
        "text_scale": 0.5,
        "rect_height": 0.375,
        "rect_width": 1.25,
    }


class TransformerIntro(Scene):
    def construct(self):
        title = Text("Transformer", color=A_AQUA)
        title.scale(1.25)
        title.shift(3.25 * UP)

        input_text = Text("Input")
        input_text.scale(0.75)
        input_text.shift(3.5 * DOWN)

        output_text = Text("Output")
        output_text.scale(0.75)

        diagram = VGroup(
            input_text,
            PositionalEncodingBlock(),
            TransformerBlock(),
            TransformerBlock(),
            SoftmaxBlock(),
            output_text,
        )
        arrows = VGroup()

        for i in range(len(diagram) - 1):
            # buff = 1.25 if i == 0 or i == len(diagram) - 2 else 1.0
            diagram[i + 1].next_to(diagram[i], UP, buff=0.625)
            arrow = Arrow(
                diagram[i],
                diagram[i + 1],
                stroke_color=A_PINK,
                stroke_width=6,
                buff=0.125,
            )
            arrows.add(arrow)

        self.add(title, input_text)
        self.add(diagram, arrows)
        self.embed()
