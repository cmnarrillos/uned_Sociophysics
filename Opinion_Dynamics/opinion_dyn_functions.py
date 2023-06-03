import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def update_support_even(x, r, k):
    try:
        y = np.zeros(x.shape)
    except:
        y = 0
    for m in range(r // 2, r):
        m = m + 1
        y = y + math.comb(r, m) * x ** m * (1 - x) ** (r - m)
    y = y + k * math.comb(r, r // 2) * x ** (r // 2) * (1 - x) ** (r // 2)

    return y


def critical_support_even(r, k):
    if r < 4:
        if abs(k - 0.5) < 1e-5:
            y = 0.5
        else:
            y = 0.5*(1-np.sign(k-0.5))
    else:
        # Define the equation: x - update_support_even(x, r, k) = 0
        equation = lambda x: x - update_support_even(x, r, k)
        x0_bracket = [0.2, 0.8]

        # Use root_scalar to find the root of the equation
        a_c = root_scalar(equation, bracket=x0_bracket, method='brentq')
        y = a_c.root

    return y


def update_support_odd(x, r):
    try:
        y = np.zeros(x.shape)
    except:
        y = 0
    for m in range(r//2, r):
        m = m + 1
        y = y + math.comb(r, m) * x**m * (1-x)**(r-m)

    return y


def plot_threshold(x, a_t1, group_size, k=None, folder='results'):
    # Enable LaTeX rendering
    # plt.rc('text', usetex=True)
    if k is None:
        k = []
    plt.figure()
    plt.plot(x, x, color='k', linewidth=2, label=r'$a_{t+1}=a_t$')
    for jj, y in enumerate(a_t1):
        plt.plot(x, y, label=r'$r = ' + str(group_size[jj]) + '$')
    plt.plot(x, x, color='k', linewidth=2)
    if not k:
        plt.title(r'Threshold for odd groups')
    else:
        plt.title(r'Threshold for even groups with $k=' + str(k) + '$')
    plt.xlabel(r'$a_t$')
    plt.ylabel(r'$a_{t+1}$')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid('minor')
    plt.legend()
    if not k:
        plt.savefig(f'./{folder}/threshold_odd.png')
    else:
        plt.savefig(f'./{folder}/threshold_even_k=' + str(k) + '.png')
    plt.close()