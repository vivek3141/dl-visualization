import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import utils
import torch.nn.functional as F
import math

# %matplotlib notebook
utils.set_default(figsize=(5, 5))

def get_data(n=3000, d=2, c=3, std=0.2):
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

# Show training curves interactively
fig, ax1 = plt.subplots(figsize=(9, 3))
ax2 = plt.twinx(ax1); ax = ax1, ax2

# Generate and train a model
model = Model(2, 100, 5)
acc_hist, loss_hist = utils.train(model, X, y, fig, ax, max_epochs=1800)

# Save model to file
utils.save_model('model3', model, (acc_hist, loss_hist))

# Display classification regions and decesion boundary
f = plt.figure()
f.add_axes([0, 0, 1.005, 1.005])
utils.plot_decision(model)
utils.plot_data(X, y)
plt.savefig(f'decision.png')