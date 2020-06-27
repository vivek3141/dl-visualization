import os
import torch
import math
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from os import path
from IPython import display
import time


torch.manual_seed(12345)
min_x, max_x, delta_x = None, None, None


def set_default(figsize=(10, 10)):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize)


def get_data(n=1000, d=2, c=3, std=0.2):
    X = torch.zeros(n * c, d)
    y = torch.zeros(n * c, dtype=torch.long)
    for i in range(c):
        index = 0
        r = torch.linspace(0.2, 1, n)
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


def plot_data(X, y, ratio='1:1', d=0, zoom=1, save_path=None):
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    if ratio == '1:1': plt.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    elif ratio == '16:9': plt.axis(np.array((-2.0, 2.0, -2 / 16 * 9, 2 / 16 * 9)) * zoom)
    else: raise ValueError('Ratio not supported')
    plt.axis('off')

    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)
    if save_path: plt.savefig(f'{save_path}{d:04d}.png')


def plot_curves(fig, ax, acc_hist, loss_hist):
    ax1, ax2 = ax
    ax1.cla(); ax2.cla()
    ax1.plot(acc_hist, 'C0')
    ax2.plot(loss_hist, 'C1')
    ax1.set_ylabel('Accuracy', color='C0')
    ax2.set_ylabel('Loss', color='C1')
    fig.canvas.draw()


def train(model, X, y, fig, ax, max_epochs=3000):
    # Or train from scratch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc_hist = list()
    loss_hist = list()

    # Training
    for t in range(max_epochs + 1):

        # Feed forward to get the logits
        perm = torch.randperm(X.size(0))
        y_pred = model(X[perm])

        # Compute the loss and accuracy
        loss = criterion(y_pred, y[perm])
        score, predicted = torch.max(y_pred.data, 1)
        correct = (y[perm] == predicted).int().sum().item()
        acc = correct / len(y)

        print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
        display.clear_output(wait=True)
        acc_hist.append(acc)
        loss_hist.append(loss.item())
        if t % 100 == 0 or correct - len(y) == 0 and t % 100 == 0:
            plot_curves(fig, ax, acc_hist, loss_hist)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if correct - len(y) == 0: break
    return acc_hist, loss_hist


def get_model(model_name, c=3):
    modules = list()
    if model_name == '5-Linear-4-LeakyReLU':
        for l in range(4):
            modules.append(nn.Linear(2, 2))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(2, 2))
    elif model_name == '2-Linear_H-100' or model_name == 'K-5_2-Linear_H-100':
        modules.append(nn.Linear(2, 100))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(100, 2))
    elif model_name == '1-Linear':
        modules.append(nn.Linear(2, 2))
    else: raise ValueError('Model name not existent')
    modules.append(nn.Linear(2, c))
    model = nn.Sequential(*modules)
    model.load_state_dict(torch.load(path.join(model_name, 'model_dict.pth')))
    acc_hist, loss_hist = torch.load(path.join(model_name, 'acc_loss.pth'))
    return model, (acc_hist, loss_hist)


def save_model(model_path, model, hist=None):
    if path.isdir(model_path):
        raise FileExistsError('Model directory already existent. Aborting.')
    os.mkdir(model_path)
    torch.save(model, path.join(model_path, 'model.pth'))
    torch.save(model.state_dict(), path.join(model_path, 'model_dict.pth'))
    if hist is not None:
        torch.save(hist, path.join(model_path, 'acc_loss.pth'))
    print(model_path, 'saved successfully.')


def plot_decision(model):
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.tensor(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)


def get_grid(ratio):
    global min_x, max_x, delta_x
    if ratio == '1:1':
        min_x, max_x, delta_x = -5.5, 5.5, .1
        min_y, max_y, delta_y = -5.5, 5.5, .1
    elif ratio == '16:9':
        min_x, max_x, delta_x = -10, 10, .1
        min_y, max_y, delta_y = -5.6, 5.6, .1
    else: raise ValueError('Ratio not supported')

    mesh_x = np.arange(min_x, max_x + delta_x, delta_x)
    mesh_y = np.arange(min_y, max_y + delta_y, delta_y)
    xx, yy = np.meshgrid(mesh_x, mesh_y)
    data = torch.tensor(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
    return data


def plot_grid(data):
    N_x = int((max_x - min_x) / delta_x + 1)
    d = data.numpy().reshape(-1, N_x, 2)
    x, y = d[:,:,0], d[:,:,1]
    plt.plot(x, y, c='.5', lw=1, zorder=0)
    plt.plot(x.T, y.T, c='.5', lw=1, zorder=0)


def plot_bases(bases, plotting=True, width=0.04):
    bases[2:] -= bases[:2]
    # if plot_bases.a: plot_bases.a.set_visible(False)
    # if plot_bases.b: plot_bases.b.set_visible(False)
    if plotting:
        plot_bases.a = plt.arrow(*bases[0], *bases[2], width=width, color='r', zorder=10, alpha=1., length_includes_head=True)
        plot_bases.b = plt.arrow(*bases[1], *bases[3], width=width, color='g', zorder=10, alpha=1., length_includes_head=True)


plot_bases.a = None
plot_bases.b = None


def in_out_interpolation(module, X_in, y, steps, action):
    I = torch.eye(2)
    X_in = torch.cat((I * 0, I, X_in))
    with torch.no_grad():
        X_out = module(X_in)
    interpolate(X_in, X_out, y, steps, action, plotting_bases=isinstance(module, nn.Linear))
    return X_out[4:]


count = 0


def interpolate(
    X_in, X_out, y, steps, action, p=1/50, plotting_bases=False,
    plotting_grid=False, K=3, ratio='1:1', single=False
):
    global count
    N = K * 1000 + 4
    for t in range(steps):
        # a = (t / (steps - 1)) ** p
        a = ((p + 1)**(t / (steps - 1)) - 1) / p
        plt.gca().cla()
        plt.text(0, 5, action, color='w', horizontalalignment='center', verticalalignment='center')
        plot_data(a * X_out[4:N] + (1 - a) * X_in[4:N], y, zoom=5, ratio=ratio)  #, d=steps*i+t)
        plot_bases(a * X_out[:4] + (1 - a) * X_in[:4], plotting=plotting_bases)
        if plotting_grid: plot_grid(a * X_out[N:] + (1 - a) * X_in[N:])
        plt.gcf().canvas.draw()
        # plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_K-5_2-Linear_H-100/{t:04d}.png')
        # plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_2-Linear-noise/{t:04d}.png')
        # plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
        count += 1
        plt.gcf().canvas.set_window_title(f'{t + 1}/{steps}')
    if not single:
        time.sleep(0.5)
        # for z in range(15):
        #     plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
        #     count += 1
        I = torch.eye(2)
        plot_bases(torch.cat((I * 0, I)), plotting=not plotting_bases)
        plt.gcf().canvas.draw()
        # for z in range(15):
        #     plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
        #     count += 1
        time.sleep(0.5)


def interpolate_rot(X_in, y, steps, M):
    global count
    theta = torch.atan2(M[1, 0], M[0, 0])
    I = torch.eye(2)
    X_in = torch.cat((I * 0, I, X_in))
    for t in range(steps):
        alpha = t / (steps - 1) * theta
        R = torch.tensor([
            [torch.cos(alpha), -torch.sin(alpha)],
            [torch.sin(alpha), torch.cos(alpha)]
        ])
        plt.gca().cla()
        plt.text(0, 5, 'Rotating', color='w', horizontalalignment='center', verticalalignment='center')
        plot_data(X_in[4:] @ R, y, zoom=5)  # , d=steps*i+t)
        plot_bases(X_in[:4] @ R)
        plt.gcf().canvas.draw()
        # plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
        count += 1
    X_out = X_in[4:] @ R
    time.sleep(0.5)
    # for z in range(15):
    #     plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
    #     count += 1
    plot_bases(torch.cat((I * 0, I)), plotting=False)
    plt.gcf().canvas.draw()
    time.sleep(0.5)
    # for z in range(15):
    #     plt.savefig(f'/Users/atcold/Scratch/Spiral/in_out_per_transformation/{count:04d}.png')
    #     count += 1

    # Reflection?
    if torch.det(M) < 0:
        F = torch.pinverse(R) @ M
        X_in = torch.cat((I * 0, I, X_out))
        X_out = X_in @ F
        interpolate(X_in, X_out, y, steps, 'Reflecting', plotting_bases=True)
        X_out = X_out[4:]

    return X_out


def animate(module, X_in, y, steps, decompose=False):
    if isinstance(module, nn.Linear) and decompose:
        X_out = decompose_affine_transformation(module, X_in, y, steps)
        # X_out = in_out_interpolation(module, X_in, steps, module.__class__.__name__)
    else:
        X_out = in_out_interpolation(module, X_in, y, steps, module.__class__.__name__)
    return X_out


def decompose_affine_transformation(module, X_in, y, steps):
    A = module._parameters.get('weight').data  # get tensor out of Parameter
    b = module._parameters.get('bias').data  # get tensor out of Parameter
    U, S, V = torch.svd(A)  # rotation, non-uniform scaling, rotation
    S = torch.diag(S)  # A = U @ S @ V'
    steps = int(steps / 2)
    I = torch.eye(2)

    # Full affine transformation
    # X_out = X_in @ A' + b = X_in @ V @ S @ U' + b

    # Rotation 1
    X_out = interpolate_rot(X_in, y, steps, V)
    X_in = X_out

    # Scaling
    X_in = torch.cat((I * 0, I, X_in))
    X_out = X_in @ S
    interpolate(X_in, X_out, y, steps, 'Scaling', plotting_bases=True)
    X_in = X_out[4:]

    # Rotation 2
    X_out = interpolate_rot(X_in, y, steps, U.t())
    X_in = X_out

    # Translation
    X_in = torch.cat((I * 0, I, X_in))
    X_out = X_in + b
    interpolate(X_in, X_out, y, steps, 'Translating', plotting_bases=True)

    return X_out[4:]


def plot_output_decision(model, ratio):
    if ratio == '1:1': mesh = np.arange(-5.5, 5.5, 0.01)
    elif ratio == '16:9': mesh = np.arange(-10, 10, 0.01)
    else: raise ValueError('Ratio not supported')

    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.tensor(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        output_layer_idx = list(model._modules.keys())[-1]
        Z = model._modules[output_layer_idx](data)
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)


def play_in_out(model, X, y, steps, K=3, ratio='1:1', p=2):
    grid = get_grid(ratio=ratio)
    warm_up(X, y, grid, ratio=ratio)

    I = torch.eye(2)
    X_in_bak = torch.cat((I * 0, I, X, grid))
    X_in = X_in_bak
    for i in range(len(model._modules) - 1):
        module = model._modules[str(i)]
        with torch.no_grad():
            X_out = module(X_in)
        X_in = X_out
    interpolate(X_in_bak, X_out, y, steps, '', p=p, plotting_grid=True, K=K,
        ratio=ratio, single=True)


def play_layer_by_layer(model, X, y, steps, decompose=False):
    warm_up(X, y, plotting_bases=True)
    X_in = X
    for i in range(len(model._modules) - 1):
        module = model._modules[str(i)]
        X_in = animate(module, X_in, y, steps, decompose)


def warm_up(X, y, grid=None, plotting_bases=False, ratio='1:1'):
    # Warm up
    plt.gca().cla()
    plot_data(X, y, zoom=5, ratio=ratio)
    I = torch.eye(2)
    if plotting_bases: plot_bases(torch.cat((I * 0, I)))
    if grid is not None: plot_grid(grid)
    plt.gcf().canvas.draw()
    time.sleep(5)


def plot_output_data(model, X, y, ratio):
    X_in = X
    for i in range(len(model._modules) - 1):
        module = model._modules[str(i)]
        with torch.no_grad():
            X_out = module(X_in)
        X_in = X_out
    plot_data(X_out, y, ratio, zoom=5)



def show_scatterplot(X, colors, title='', axis=True):
    colors = zieger[colors[:,0], colors[:,1]]
    X = X.numpy()
    # plt.figure()
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=30)
    # plt.grid(True)
    plt.title(title)
    plt.axis('off')
    _m, _c = 0, '.15'
    if axis:
        plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
        plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)