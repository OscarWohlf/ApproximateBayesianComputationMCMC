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

def sample_theta_goal4():
    log_Ke = np.random.normal(-2.7, 0.6)
    log_Ka = np.random.normal(0.14, 0.4)
    log_Cl = np.random.normal(-3,0.8)
    log_sig = np.random.normal(-1.1, 0.3)
    return np.array([np.exp(log_Ke),np.exp(log_Ka),np.exp(log_Cl), np.exp(log_sig)])

def main():
    xs, ts = euler_maruyama(0.01, 12, b_goal4, sig_goal4)
    print(xs)
    print(ts)

if __name__ == "__main__":
    main()