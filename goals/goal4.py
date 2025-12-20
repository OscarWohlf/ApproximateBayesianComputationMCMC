import numpy as np
from algorithms.euler_maruyama import euler_maruyama

def exact_sim():
    return None 

def b_goal4(x_n, t_n, theta):
    Ke, Ka, Cl, sigma = theta
    frac = (4 * Ka * Ke) / Cl
    exp = np.exp(-Ka* t_n)
    return frac * exp - (Ke * x_n)

def sig_goal4(x_n, t_n, theta):
    Ke, Ka, Cl, sigma = theta
    return sigma

def sample_theta_goal4():
    log_Ke = np.random.normal(-2.7, 0.6)
    log_Ka = np.random.normal(0.14, 0.4)
    log_Cl = np.random.normal(-3,0.8)
    log_sig = np.random.normal(-1.1, 0.3)
    return np.array([np.exp(log_Ke),np.exp(log_Ka),np.exp(log_Cl), np.exp(log_sig)])

def simulate_dataset_em(theta):
    times = np.array([0.25,0.5,1,2,3.5,5,7,9,12])
    dt = 0.01
    xs, ts = euler_maruyama(dt, 12, b_goal4, sig_goal4, theta)
    idx = (times / dt).astype(int)
    return xs[idx]

def main():
    theta = sample_theta_goal4()
    print(simulate_dataset_em(theta))

if __name__ == "__main__":
    main()