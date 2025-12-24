import numpy as np
from algorithms.euler_maruyama import euler_maruyama
from algorithms.exact_sim import exact_sim


def sample_theta_goal4():
    log_Ke = np.random.normal(-2.7, 0.6)
    log_Ka = np.random.normal(0.14, 0.4)
    log_Cl = np.random.normal(-3,0.8)
    log_sig = np.random.normal(-1.1, 0.3)
    return np.array([np.exp(log_Ke),np.exp(log_Ka),np.exp(log_Cl), np.exp(log_sig)])

def simulate_dataset(theta, method='exact', dt=0.01):
    if method == 'exact':
        return exact_sim(theta=theta)

    elif method == 'em':
        return euler_maruyama(theta=theta, dt=dt)

    else:
        raise ValueError("Method Not Supported: must be either 'exact' or 'em' (Euler-Maruyama).")

def create_dataset(n, method='exact', dt=0.01):
    X = np.zeros((n,10))
    Y = np.zeros((n,4))

    for i in range(n):
        theta_i = sample_theta_goal4()
        D_i = simulate_dataset(theta=theta_i, method=method, dt=dt)
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
    # Generate true data
    given_theta = np.array([0.08,1.5,0.04,0.2])
    data_obs = simulate_dataset(theta=given_theta, method='exact')

    print("Observed Ground-Truth Data:")
    print(data_obs.tolist())

    # Generate sample parameters from prior: Y, and corresponding data: X
    X, Y = create_dataset(n=10000, method='exact')

    # Estimated regression coefficients
    betas = get_betas(X,Y)
    print("\nEstimated Regression Coefficients:")
    print(betas)

    # Verification: check if the estimated coeffs. recover the ground-truth parameters with the summarised statistics
    S_D = summary_stat(data_obs, betas)
    print("\nValidation:")
    print(f"True Params: {given_theta.tolist()}")
    print(f"Est. Params: {np.round(S_D, 4).tolist()}")

if __name__ == "__main__":
    main()