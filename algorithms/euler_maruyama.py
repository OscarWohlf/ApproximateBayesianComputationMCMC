import numpy as np
from scipy.stats import norm

def euler_maruyama(d_t, T, b, sig):
    n = int(np.ceil(T/d_t))
    xs = np.zeros(n+1)
    ts = np.linspace(0.0, n * d_t, n + 1)
    for i in range(n):
        prev_x = xs[i]
        prev_t = ts[i]
        b_val = b(prev_x, prev_t)
        sig_val = sig(prev_x, prev_t)
        d_W = np.random.normal(0,np.sqrt(d_t))
        new_x = prev_x + b_val * d_t + sig_val * d_W
        xs[i+1] = new_x
    return xs, ts


