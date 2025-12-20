import numpy as np
from algorithms.euler_maruyama import euler_maruyama
def exact_sim():
    return None 

def b_goal4(x_n, t_n):
    frac = (4 * 1.5 * 0.08) / 0.04
    exp = np.exp(-1.5 * t_n)
    return frac * exp - (0.08 * x_n)

def sig_goal4(x_n, t_n):
    return 0.2

