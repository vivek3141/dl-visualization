from manimlib.imports import *


class PerceptronThreeViz(Scene):
    def construct(self):
        frame = self.camera.frame

        axes = ThreeDAxes()

        surface = ParametricSurface(
            lambda u, v: [u, v, 0.2 * (u + v) + 2],
            u_range=(-4, 4),
            v_range=(-4, 4),
            color=RED
        )

        frame = self.camera.frame
        frame.set_rotation(phi=0.35 * PI)

        self.play(
            frame.set_theta, 0,
            frame.set_phi, 0.35 * PI,
            run_time=0.01
        )

        plane = NumberPlane()

        self.play(Write(plane), Write(axes))

        axes = ThreeDAxes()

        self.play(FadeIn(surface))
        self.wait()

        rotate = True
        frame.add_updater(
            lambda m, dt: m.become(m.rotate(-0.2 * dt)) if rotate else None
        )

        self.wait(30)
