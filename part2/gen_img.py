import torch
from PIL import Image
import numpy as np

path = './model/model.pth'
model = torch.load(path)
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

colors = [RED, YELLOW, GREEN, BLUE, PURPLE]

x_min = -FRAME_WIDTH/2
x_max = FRAME_WIDTH/2
y_min = -FRAME_HEIGHT/2
y_max = FRAME_HEIGHT/2

x_values = np.linspace(x_min, x_max, XRES+1)
y_values = np.linspace(y_min, y_max, YRES+1)

pixels = []

for i in range(len(y_values) - 1)[::-1]:
    pixels.append([])
    for j in range(len(x_values) - 1):
        x1, x2 = x_values[j:j + 2]
        y1, y2 = y_values[i:i + 2]

        inp = torch.tensor(
            [(x1 + x2)/2, (y1 + y2)/2], dtype=torch.float32)
        c = colors[np.argmax(model[3:].forward(inp).detach().numpy())]

        pixels[-1].append(c)


array = np.array(pixels, dtype=np.uint8)

new_image = Image.fromarray(array)
new_image.save("output_decisions.png")
new_image.show()
