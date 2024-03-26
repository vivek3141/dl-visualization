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


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def relu(*x):
    return [max(i, 0) for i in x]


class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2)
        self.linear3 = torch.nn.Linear(2, D_out)

    def forward(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = activation(x)
        return self.linear3(self.linear2(x))

    def forward2(self, x, nb_relu_dim=100):
        x = self.linear1(x)
        x = activation(x)
        return self.linear2(x)


path = './model/model.pth'
model = torch.load(path)

print(model[3])

x_values = np.linspace(x_min, x_max, XRES+1)
y_values = np.linspace(y_min, y_max, YRES+1)

pixels = []

w = model[3].weight.detach().numpy()
b = model[3].bias.detach().numpy()

for i in range(len(y_values) - 1)[::-1]:
    pixels.append([])
    for j in range(len(x_values) - 1):
        x1, x2 = x_values[j:j + 2]
        y1, y2 = y_values[i:i + 2]

        # inp = torch.tensor(
        #     [(x1 + x2)/2, (y1 + y2)/2], dtype=torch.float32)

        #print(w, b)

        x, y = (x1 + x2)/2, (y1 + y2)/2

        #print(w.dot(np.array([[x], [y]])))

        y = np.argmax((np.array([[x, y]]).dot(w.T) + b)[0][:4])
        # print(y)

        #y = np.argmax(model[3](torch.tensor([[x, y]], dtype=torch.float32)).detach())

        c = colors[y]
        pixels[-1].append(c)


array = np.array(pixels, dtype=np.uint8)

new_image = Image.fromarray(array)
new_image.save("img/plane4.png")
new_image.show()
