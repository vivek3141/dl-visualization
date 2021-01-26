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