import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors

def plot_gammas(xs, gammas, func):
    cmap = mpl.colormaps['copper']
    norm = mcolors.LogNorm(vmin=min(gammas), vmax=max(gammas))
    for gamma in gammas:
        plt.plot(xs, func(xs, gamma), color=cmap(norm(gamma)))
    ax = plt.gca()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set an array for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("$\\gamma$")
    plt.xlabel("$x$")

def plot_congestion(xs, gammas):
    func = lambda xs, gamma: xs**(-gamma)
    plot_gammas(xs, gammas, func)
    plt.ylabel(r'$1/s_\gamma(x)$')
    fname = './img/gamma_congestion.png'
    print('Saving to', fname)
    plt.savefig(fname)

def plot_welfare(xs, gammas):
    func = lambda xs, gamma: np.maximum(xs**(1-gamma), np.ones_like(xs))
    # print(xs**(1-0.5))
    # print(np.ones_like(xs))
    # print(np.max(xs**(1-0.5), np.ones_like(xs)))
    # print(func(xs, 1.0))
    plot_gammas(xs, gammas, func)
    plt.ylabel(r'$\max(x/s_\gamma(x), 1)$')
    fname = './img/gamma_welfare.png'
    print('Saving to', fname)
    plt.savefig(fname)

if __name__ == '__main__':
    gammas = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 20])
    xs = np.arange(1, 16, 1)
    plot_congestion(xs, gammas)
    plot_welfare(xs, gammas)
