from manimlib.imports import *
import torch


class DecisionQuad(VGroup):
    def __init__(self, func1, point, func2, color, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.add(Polygon(ORIGIN, ))


class DecisionsOld(VGroup):
    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        M1 = [8, 0.15, -1.65]
        M2 = [1.75, 0.1]
        poly_args = {
            "fill_opacity": 0.45,
            "stroke_width": 6,
        }
        self.add(
            Polygon(
                ORIGIN,
                [FRAME_HEIGHT/(2 * 8), FRAME_HEIGHT/2, 0],
                [FRAME_WIDTH/2, FRAME_HEIGHT/2, 0],
                [FRAME_WIDTH/2, 0.15 * FRAME_WIDTH/2, 0],
                **poly_args, color=YELLOW),
            Polygon(
                ORIGIN,
                [FRAME_WIDTH/2, 0.15 * FRAME_WIDTH/2, 0],
                [FRAME_WIDTH/2, -FRAME_HEIGHT/2, 0],
                [-FRAME_HEIGHT/(2 * -1.65), -FRAME_HEIGHT/2, 0],
                **poly_args, color=RED),
            Polygon(
                ORIGIN,
                [-FRAME_HEIGHT/(2 * -1.65), -FRAME_HEIGHT/2, 0],
                [-FRAME_HEIGHT/(2 * 1.75), -FRAME_HEIGHT/2, 0],
                **poly_args, color=PURPLE),
            Polygon(
                ORIGIN,
                [-FRAME_HEIGHT/(2 * 1.75), -FRAME_HEIGHT/2, 0],
                [-FRAME_WIDTH/2, -FRAME_HEIGHT/2, 0],
                [-FRAME_WIDTH/2, 0.1 * -FRAME_WIDTH/2, 0],
                **poly_args, color=BLUE),
            Polygon(
                ORIGIN,
                [-FRAME_WIDTH/2, 0.1 * -FRAME_WIDTH/2, 0],
                [-FRAME_WIDTH/2, FRAME_HEIGHT/2, 0],
                [FRAME_HEIGHT/(2 * 8), FRAME_HEIGHT/2, 0],
                **poly_args, color=GREEN),
        )
        """
        self.add(*[
            FunctionGraph(lambda x: i * x, x_min=0) for i in M1
        ],
            *[
            FunctionGraph(lambda x: i * x, x_max=0) for i in M2
        ])
        """


class Decisions(VGroup):
    def __init__(self, n=150, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        """
        mesh = np.arange(-1.1, 1.1, 0.01)
        xx, yy = np.meshgrid(mesh, mesh)
        with torch.no_grad():
            data = torch.tensor(
                np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
            Z = model(data).detach()
        Z = np.argmax(Z, axis=1).reshape(xx.shape)[0].numpy()
        print(xx)

        n = len(xx)
        for i in range(n):
            z = Z[i]
            print(z)
            self.add(
                Rectangle(
                    height=FRAME_HEIGHT/n,
                    width=FRAME_WIDTH/n,
                    color=colors[z],
                    fill_opacity=0.2,
                    stroke_opacity=0.2).shift([xx[i], yy[i], 0])
            )
        #plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
        """
        xval = np.linspace(-FRAME_WIDTH/2, FRAME_WIDTH/2, num=n)
        yval = np.linspace(-FRAME_HEIGHT/2, FRAME_HEIGHT/2, num=n)

        ParametricSurface

        for x in xval:
            for y in yval:
                inp = torch.tensor([x, y], dtype=torch.float32)
                c = colors[np.argmax(model[3:].forward(inp).detach().numpy())]
                self.add(
                    Rectangle(
                        height=FRAME_HEIGHT/(n-1),
                        width=FRAME_WIDTH/(n-1),
                        color=c,
                        fill_opacity=0.45,
                        stroke_width=0).shift([x, y, 0])
                )


class NNTest(Scene):
    def construct(self):
        final_dots = VGroup(
            *[
                Dot(self.function([point[0], point[1], 0]), color=colors[Y[index]],
                    radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
            ]
        )
        d = Decisions()
        self.add(final_dots, d)

    def function(self, point):
        x, y, z = point
        inp = torch.tensor([x, y], dtype=torch.float32)
        x, y = model[:3].forward(inp).detach().numpy()
        return 0.5 * (x * RIGHT + y * UP)
