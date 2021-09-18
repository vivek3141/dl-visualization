import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import utils
import torch.nn.functional as F
import math

# %matplotlib notebook
utils.set_default(figsize=(5, 5))

def get_data(n=2000, d=2, c=3, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    for i in range(c):
        index = 0
        r = torch.linspace(0.2, 10, n)
        t = torch.linspace(
            i * 2 * math.pi / c,
            (i + 2) * 2 * math.pi / c,
            n
        ) + torch.randn(n) * std

        for ix in range(n * i, n * (i + 1)):
            X[ix] = r[index] * torch.FloatTensor((
                math.sin(t[index]), math.cos(t[index])
            ))
            y[ix] = i
            index += 1
    return X, y


X, y = get_data(c=5)

# visualise the data
f = plt.figure()
f.add_axes([0, 0, 1, 1])
utils.plot_data(X, y)


# NN model
class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2)
        self.linear3 = torch.nn.Linear(2, D_out)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear3(self.linear2(x))
    
    def forward2(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)

# Show training curves interactively
fig, ax1 = plt.subplots(figsize=(9, 3))
ax2 = plt.twinx(ax1); ax = ax1, ax2

# Generate and train a model
model = Model(2, 100, 5)
acc_hist, loss_hist = utils.train(model, X, y, fig, ax, max_epochs=1500)

# Save model to file
utils.save_model('model3', model, (acc_hist, loss_hist))

# Display classification regions and decesion boundary
f = plt.figure()
f.add_axes([0, 0, 1.005, 1.005])
utils.plot_decision(model)
utils.plot_data(X, y)
plt.savefig(f'decision.png')