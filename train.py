import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import utils

# %%
# %matplotlib notebook
utils.set_default(figsize=(5, 5))

# %%
X, y = utils.get_data(c=5)

# %%
# visualise the data
f = plt.figure()
f.add_axes([0, 0, 1, 1])
utils.plot_data(X, y)


# %%
# NN model
def n_layer_network(in_size, out_size, nb_hidden, h_size):
    
    modules = list()
    in_d = in_size

    for l in range(nb_hidden):
        modules.append(nn.Linear(in_d, h_size))
        modules.append(nn.LeakyReLU())
        in_d = h_size
    # modules.append(nn.Linear(H, H))  # added layer
    modules.append(nn.Linear(h_size, 2)); h_size = 2  # added layer
    modules.append(nn.Linear(h_size, out_size))
    return nn.Sequential(*modules)


# %%
# Show training curves interactively
fig, ax1 = plt.subplots(figsize=(9, 3))
ax2 = plt.twinx(ax1); ax = ax1, ax2

# %%
# Generate and train a model
model = n_layer_network(in_size=2, out_size=5, nb_hidden=1, h_size=100)
acc_hist, loss_hist = utils.train(model, X, y, fig, ax, max_epochs=1500)

# %%
# Save model to file
utils.save_model('K-5_2-Linear_H-100', model, (acc_hist, loss_hist))

# %%
# Display classification regions and decesion boundary
f = plt.figure()
f.add_axes([0, 0, 1.005, 1.005])
utils.plot_decision(model)
utils.plot_data(X, y)
plt.savefig(f'decision.png')