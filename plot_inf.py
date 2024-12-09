import numpy as np
import matplotlib.pyplot as plt
from math import comb
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

P_EPS = 0.05
# P_EPS = 0.0005

def binomial(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)

def binomial_expectation(n, p, f):
    s = 0
    for i in range(n + 1):
        s += f(i) * binomial(n, i, p)
    return s

def get_conditional_utilities(ds, n, p, gamma):
    f = lambda x: (1+x)**(-gamma)
    return ds[0] * binomial_expectation(n-1, p, f), ds[1] * binomial_expectation(n-1, 1-p, f)

def get_social_welfare(ds, n, p, gamma):
    if gamma < 1:
        f = lambda x: x ** (1 - gamma)
    else:
        # f = lambda x: 1 if x > 0 else 0
        return ds[0] * (1 - (1-p)**n) + ds[1] * (1 - p**n)
    return binomial_expectation(n, p, f) * ds[0] + binomial_expectation(n, 1-p, f) * ds[1]

def plot_utilities(ds, n, gamma):
    ps = np.arange(0.5, 1, P_EPS)
    us = np.zeros((len(ps), 2))
    for i, p in enumerate(ps):
        us[i] = get_conditional_utilities(ds, n, p, gamma)
    plt.plot(ps, us[:, 0], label='Class 1')
    plt.plot(ps, us[:, 1], label='Class 2')
    min_index = np.argmin(np.abs(us[:,0] - us[:,1]))
    eq_p = ps[min_index]
    plt.scatter(eq_p, us[min_index, 0], color='k')
    plt.xlabel('p')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()

def get_eq(ds, n, gamma):
    # TODO fix for small n
    ps = np.arange(0.5, 1, P_EPS)
    us = np.zeros((len(ps), 2))
    # TODO could binary search
    for i, p in enumerate(ps):
        us[i] = get_conditional_utilities(ds, n, p, gamma)
    min_index = np.argmin(np.abs(us[:,0] - us[:,1]))
    if us[-1, 0] >= us[-1, 1]:
        min_index = len(ps) - 1
    return ps[min_index], us[min_index, 0]

def get_eq_efficient(ds, n, gamma, eps=1e-4):
    p = 0.5
    inc = 0.5
    u1, u2 = 0, 0
    while inc > eps and p <= 1:
        p_prime = p + inc
        u1, u2 = get_conditional_utilities(ds, n, p_prime, gamma)
        if u1 < u2:
            inc /= 2
        else:
            p = p_prime
    if p > 1:
        p = 1.0
    return p, u1

def get_sw(ds, n, gamma):
    ps = np.arange(0.5, 1, P_EPS)
    sw = np.zeros(len(ps))
    for i, p in enumerate(ps):
        sw[i] = get_social_welfare(ds, n, p, gamma)
    max_ind = np.argmax(sw)
    return ps[max_ind], sw[max_ind]

def get_du_small_gamma(n, p, gamma):
    assert gamma < 1
    f = lambda x: (1+x)**(-gamma)
    return binomial_expectation(n-1, p, f) + (n-1) * (binomial_expectation(n-1, p, f) - binomial_expectation(n-2, p, f))

def get_social_welfare_deriv_sign(ds, n, p, gamma):
    if gamma < 1:
        # f = lambda x: (1+x)**(-gamma)
        # f_bar = lambda x: (2+x)**(-gamma) - (1+x)**(-gamma)
        # print(binomial_expectation(n-1, p, f))
        # print((n-1)*binomial_expectation(n-2, p, f_bar))
        # print((n-1)*(binomial_expectation(n-1, p, f) - binomial_expectation(n-2, p, f)))
        # print(binomial_expectation(n-1, 1-p, f))
        # print((n-1)*binomial_expectation(n-2, 1-p, f_bar))
        # print((n-1)*(binomial_expectation(n-1, 1-p, f) - binomial_expectation(n-2, 1-p, f)))
        du0 = ds[0] * get_du_small_gamma(n, p, gamma)
        du1 = ds[1] * get_du_small_gamma(n, 1-p, gamma)
        # print('du0', du0)
        # print('du1', du1)
        return du0 - du1
        # return ds[0] * binomial_expectation(n-1, p, f_bar) - ds[1] * binomial_expectation(n-1, 1-p, f_bar) > 0
    else:
        # f = lambda x: 1 if x > 0 else 0
        return ds[0] * (1 - p)**(n-1) - ds[1] * p**(n-1)

def get_sw_efficient(ds, n, gamma, eps=1e-4):
    # dp = 1e-4
    p = 0.5
    inc = 0.5
    while inc > eps and p <= 1:
        # print(p, inc)
        p_prime = p + inc
        deriv_sign = get_social_welfare_deriv_sign(ds, n, p_prime, gamma)
        # sw_minus = get_social_welfare(ds, n, p_prime - dp, gamma)
        # sw_plus = get_social_welfare(ds, n, p_prime, gamma)
        if deriv_sign < 0:
            inc /= 2
        else:
            p = p_prime
    if p > 1:
        p = 1.0
    return p, get_social_welfare(ds, n, p, gamma)

def plot_social_welfare(ds, n, gamma):
    ps = np.arange(0.5, 1, P_EPS)
    sw = np.zeros(len(ps))
    for i, p in enumerate(ps):
        sw[i] = get_social_welfare(ds, n, p, gamma)
    plt.plot(ps, sw)
    plt.xlabel('p')
    plt.ylabel('Social Welfare')
    theoretical_p = get_theoretical_ps(ds, gamma)[0]
    plt.axvline(x=theoretical_p, color='r', linestyle='--', label='Theoretical p')
    max_sw = np.max(sw)
    max_p = ps[np.argmax(sw)]
    plt.scatter(max_p, max_sw, color='r', label='Max SW')
    plt.show()

def get_theoretical_ps(ds, gamma):
    ps = ds ** (1 / gamma)
    return ps / np.sum(ps)

CM = 'Dark2'

def make_plot(ds):
    ns = range(1, 51)
    gammas = [.1, 0.25, .5, 0.75, 0.99, 0.999, 1.0, 2.0]
    # gammas = [.1, 0.25, .5, 0.999, 1.0, 2.0]
    # for key in mpl.colormaps.keys():
        # print(key)
    print(mpl.color_sequences)
    cycle = True
    norm = mcolors.Normalize(vmin=min(gammas), vmax=max(gammas))
    _, ax = plt.subplots()
    def get_cmap():
        return mpl.color_sequences[CM]
    cmap = get_cmap()
    cmap_iter = iter(cmap)
    def get_next_color(gamma):
        if cycle:
            return next(cmap_iter)
        else:
            return cmap(norm(gamma))
    for gamma in gammas:
        p_opts = []
        for n in ns:
            p_opt, _ = get_sw(ds, n, gamma)
            p_opts.append(p_opt)
        plt.plot(ns, p_opts, label=f'{gamma}', color=get_next_color(gamma))
    cmap_iter = iter(cmap)
    for gamma in gammas:
        p_eqs = []
        for n in ns:
            p_eq, _ = get_eq_efficient(ds, n, gamma)
            p_eqs.append(p_eq)
        ax.plot(ns, p_eqs, ls='--', color=get_next_color(gamma))
    cmap_iter = iter(cmap)
    for gamma in gammas:
        ax.scatter([max(ns)], [get_theoretical_ps(ds, gamma)[0]], marker='X', color=get_next_color(gamma))

    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # You need to set an array for ScalarMappable
    # cbar = plt.colorbar(sm, ax=ax)
    plt.xlabel('$n$')
    plt.ylabel('$p_1$')
    plt.legend()
    plt.savefig('./img/ps_to_inf.png', dpi=300)
    plt.show()

def plot_surface(ds):
    ns = range(1, 26)
    gammas = [.1, 0.25, .5, 0.75, 0.99, 0.999, 1.0, 2.0]
    gammas = np.arange(0.1, 2.5, 0.005)
    n_grid, gamma_grid = np.meshgrid(ns, gammas)
    p_eqs = np.zeros_like(n_grid, dtype=float)
    p_opts = np.zeros_like(n_grid, dtype=float)
    # TODO improve
    for i, n in enumerate(ns):
        for j, gamma in enumerate(gammas):
            p_eq, _ = get_eq_efficient(ds, n, gamma)
            p_opt, _ = get_sw_efficient(ds, n, gamma)
            p_eqs[j, i] = p_eq
            p_opts[j, i] = p_opt


    cmap='copper'
    cmap='gist_earth'
    norm = mcolors.Normalize(vmin=np.min(p_opts), vmax=np.max(p_opts)+.03)
    print(mpl.colormaps)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(n_grid, gamma_grid, p_opts, cmap=cmap, norm=norm)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'$\mathbf{p}_1$')
    # TODO rotate z axis label
    view = (None, None, (0.5, 1.0), 13, 40, 0)
    ax._set_view(view=view)
    plot_3d_lim(ds, ax, ns, gammas)
    plt.tight_layout()
    plt.savefig('./img/lim_opt_3d.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(n_grid, gamma_grid, p_eqs, cmap=cmap, norm=norm)
    ax._set_view(view=view)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'$\mathbf{p}_1$')
    plot_3d_lim(ds, ax, ns, gammas)
    plt.tight_layout()
    plt.savefig('./img/lim_eq_3d.png', dpi=300)
    plt.show()

def plot_3d_lim(ds, ax, ns, gammas):
    max_n = max(ns)
    p_lims = [get_theoretical_ps(ds, gamma)[0] for gamma in gammas]
    ax.plot([max_n] * len(gammas), gammas, p_lims, color='m', alpha=1.0, lw=3, zorder=10)

if __name__ == '__main__':
    ds = np.array([5, 2])
    plot_surface(ds)
    # make_plot(ds)



    # n = 2
    # gamma = 1
    # print(get_sw(ds, n, gamma))
    # print(get_eq(ds, n, gamma))
    # plot_utilities(ds, n, gamma)
    # plot_social_welfare(ds, n, gamma)
    # print(get_eq(ds, n, gamma))
    # print(get_conditional_utilities(ds, n, p, gamma))
    # print(get_social_welfare(ds, n, p, gamma))
    # print(get_social_welfare(ds, n, 0.6, gamma))
