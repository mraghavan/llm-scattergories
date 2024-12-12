import numpy as np
import matplotlib.pyplot as plt
from math import comb
import matplotlib.colors as mcolors

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
        return binomial_expectation(n, p, f) * ds[0] + binomial_expectation(n, 1-p, f) * ds[1]
    else:
        return ds[0] * (1 - (1-p)**n) + ds[1] * (1 - p**n)

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

def get_du_small_gamma(n, p, gamma):
    assert gamma < 1
    f = lambda x: (1+x)**(-gamma)
    return binomial_expectation(n-1, p, f) + (n-1) * (binomial_expectation(n-1, p, f) - binomial_expectation(n-2, p, f))

def get_social_welfare_deriv_sign(ds, n, p, gamma):
    if gamma < 1:
        du0 = ds[0] * get_du_small_gamma(n, p, gamma)
        du1 = ds[1] * get_du_small_gamma(n, 1-p, gamma)
        return du0 - du1
    else:
        return ds[0] * (1 - p)**(n-1) - ds[1] * p**(n-1)

def get_sw_efficient(ds, n, gamma, eps=1e-4):
    # dp = 1e-4
    p = 0.5
    inc = 0.5
    while inc > eps and p <= 1:
        p_prime = p + inc
        deriv_sign = get_social_welfare_deriv_sign(ds, n, p_prime, gamma)
        if deriv_sign < 0:
            inc /= 2
        else:
            p = p_prime
    if p > 1:
        p = 1.0
    return p, get_social_welfare(ds, n, p, gamma)

def get_theoretical_ps(ds, gamma):
    ps = ds ** (1 / gamma)
    return ps / np.sum(ps)

def plot_surface(ds):
    ns = range(1, 26)
    gammas = np.arange(0.1, 2.5, 0.01)
    print(len(gammas))
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


    cmap='gist_earth'
    norm = mcolors.Normalize(vmin=np.min(p_opts), vmax=np.max(p_opts)+.03)
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(n_grid, gamma_grid, p_opts, cmap=cmap, norm=norm, rstride=1, cstride=1)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\gamma$')
    # TODO rotate z axis label
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\mathbf{p}_1$', rotation=0)
    view = (None, None, (0.5, 1.0), 13, 40, 0)
    ax._set_view(view=view)
    plot_3d_lim(ds, ax, ns, gammas)
    plt.tight_layout()
    plt.savefig('./img/lim_opt_3d.png', dpi=300)
    plt.show()

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(n_grid, gamma_grid, p_eqs, cmap=cmap, norm=norm, rstride=1, cstride=1)
    ax._set_view(view=view)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\gamma$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\mathbf{p}_1$', rotation=0)
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
