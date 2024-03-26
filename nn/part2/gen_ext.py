from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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

activation = F.relu


class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 100)
        self.linear3 = torch.nn.Linear(H, 100)
        self.linear4 = torch.nn.Linear(100, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear4(F.relu(self.linear3(self.linear2(x))))

    def forward2(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)


path = './model3/model.pth'
model = torch.load(path)

x_values = np.linspace(x_min, x_max, XRES+1)
y_values = np.linspace(y_min, y_max, YRES+1)

pixels = []

for i in range(len(y_values) - 1)[::-1]:
    print(f"Done with {YRES-i}/{YRES}")
    pixels.append([])
    for j in range(len(x_values) - 1):
        x1, x2 = x_values[j:j + 2]
        y1, y2 = y_values[i:i + 2]

        inp = torch.tensor(
            [(x1 + x2)/2, (y1 + y2)/2], dtype=torch.float32)
        c = colors[np.argmax(model(inp).detach().numpy())]
        pixels[-1].append(c)


array = np.array(pixels, dtype=np.uint8)

new_image = Image.fromarray(array)
new_image.save("relu_ext_decisions.png")
new_image.show()
