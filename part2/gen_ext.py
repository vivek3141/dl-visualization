from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


def get_data(n=1000, d=2, c=3, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    for i in range(c):
        index = 0
        r1 = torch.linspace(0.2, 1, 100)
        t1 = torch.linspace(
            i * 2 * math.pi / c,
            (i + 2) * 2 * math.pi / c,
            100
        ) + torch.randn(100) * std

        r2 = torch.linspace(1, 10, n-100)
        t2 = torch.linspace(
            (i + 2) * 2 * math.pi / c,
            (i + 20) * 2 * math.pi / c,
            n-100
        ) + torch.randn(n-100) * std

        r = torch.cat((r1, r2))
        t = torch.cat((t1, t2))

        for ix in range(n * i, n * (i + 1)):
            X[ix] = r[index] * torch.FloatTensor((
                math.sin(t[index]), math.cos(t[index])
            ))
            y[ix] = i
            index += 1
    return X.numpy(), y.numpy()


torch.manual_seed(231)

ASPECT_RATIO = 16.0 / 9.0
FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO

XRES = 1920
YRES = 1080

RED = (252, 98, 85)
YELLOW = (255, 255, 0)
GREEN = (131, 193, 103)
BLUE = (88, 196, 221)
PURPLE = (154, 114, 172)
BLACK = (0, 0, 0)
colors = [RED, YELLOW, GREEN, BLUE, PURPLE, BLACK]

x_min = -FRAME_WIDTH/2
x_max = FRAME_WIDTH/2
y_min = -FRAME_HEIGHT/2
y_max = FRAME_HEIGHT/2

x_values = np.linspace(x_min, x_max, XRES+1)
y_values = np.linspace(y_min, y_max, YRES+1)

pixels = []
X, y = get_data(c=5)


def func(inp):
    ans = -1
    min_d = float("inf")

    for i, x in enumerate(X):
        xi, yi = x
        dist = (inp[0] - xi)**2 + (inp[1] - yi)**2
        if dist < min_d:
            ans = y[i]
            min_d = dist

    return ans


for i in range(len(y_values) - 1)[::-1]:
    pixels.append([])
    for j in range(len(x_values) - 1):
        x1, x2 = x_values[j:j + 2]
        y1, y2 = y_values[i:i + 2]

        inp = [(x1 + x2)/2, (y1 + y2)/2]
        c = colors[func(inp)]

        pixels[-1].append(c)


array = np.array(pixels, dtype=np.uint8)

new_image = Image.fromarray(array)
new_image.save("relu_ext_decisions.png")
new_image.show()
