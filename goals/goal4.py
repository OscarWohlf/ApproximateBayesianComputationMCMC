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

def simulate_dataset_em(theta, dt):
    times = np.array([0.25,0.5,1,2,3.5,5,7,9,12])
    xs, ts = euler_maruyama(dt, 12, b_goal4, sig_goal4, theta)
    idx = np.round((times / dt).astype(int))
    return xs[idx]

def create_dataset(n, dt):
    X = np.zeros((n,10))
    Y = np.zeros((n,4))

    for i in range(n):
        theta_i = sample_theta_goal4()
        D_i = simulate_dataset_em(theta_i, dt)
        X[i,0] = 1.0
        X[i, 1:] = D_i
        Y[i,:] = theta_i
    
    return X, Y

def get_betas(X, Y):
    betas, *_ = np.linalg.lstsq(X,Y)
    return betas

def summary_stat(data, betas):
    x = np.zeros(10)
    x[0] = 1.0
    x[1:] = data
    return x @ betas
    
def main():
    given_theta = np.array([0.08,1.5,0.04,0.2])
    data_obs = simulate_dataset_em(given_theta, 0.01)
    print(data_obs)
    X, Y = create_dataset(1000, 0.01)
    print(X.shape)
    print(Y.shape)
    print(get_betas(X,Y))

if __name__ == "__main__":
    main()