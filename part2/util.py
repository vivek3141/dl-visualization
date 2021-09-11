from manimlib import *

import torch
import pickle
import gzip

from utils import get_data

path = './model/model.pth'
model = torch.load(path)
torch.manual_seed(231)

colors = [RED, YELLOW, GREEN, BLUE, PURPLE]

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


X, Y = get_data(c=5)

def load_data():
    f = gzip.open('../mnist/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def heaviside(x):
    return int(x >= 0)


def get_dots(func):
    return VGroup(
        *[
            Dot(func([point[0], point[1], 0]), color=colors[Y[index]],
                radius=0.75*DEFAULT_DOT_RADIUS) for index, point in enumerate(X)
        ]
    )