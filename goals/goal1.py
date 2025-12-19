import numpy as np
from algorithms.algorithm1 import algorithm1
from scipy.stats import norm
import matplotlib.pyplot as plt

def abs_diff_discrepancy(gen_data, data):
    mean_gen_data = np.mean(gen_data)
    mean_data = np.mean(data)
    return np.abs(mean_gen_data - mean_data)

def prior_g1():
    mu = 0
    var = 3
    return np.random.normal(mu,np.sqrt(var))

def model_g1(theta, M):
    a = 1
    var = 0.1
    distr = np.random.randint(0,2)
    if distr == 0:
        return np.random.normal(theta, np.sqrt(var), M)
    else: 
        return np.random.normal(theta + a, np.sqrt(var), M)

def true_posterior_g1(theta):
    a = 1
    M = 100
    var_1 = 0.1
    var = 3
    data_mean = 0

    alpha = 1 / (1 + np.exp(a * (data_mean - (a / 2)) * (M / (M * var + var_1))))
    mu_1 = (var / (var + var_1 / M)) * data_mean 
    mu_2 = (var / (var + var_1 / M)) * (data_mean - a)
    var_post = var_1 / (M + (var_1 / var))
    std = np.sqrt(var_post)

    return alpha * norm.pdf(theta, loc = mu_1, scale = std) + (1 - alpha) * norm.pdf(theta, loc = mu_2, scale = std)

    

def main():
    M = 100
    eps = [0.75, 0.25, 0.1, 0.025]
    data = np.zeros(M)
    N = 500

    grid = np.linspace(-3, 3, 1000)
    true_pdf = true_posterior_g1(grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, e in zip(axes, eps):
        thetas, num_props = algorithm1(
            N, prior_g1, model_g1, M, abs_diff_discrepancy, data, e
        )
        acc = N / num_props

        ax.hist(thetas, bins=40, density=True, alpha=0.6, label="ABC accepted θ")
        ax.plot(grid, true_pdf, label="True posterior pdf")
        ax.set_title(f"ε={e} | acc={acc:.3f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()