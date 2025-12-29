import numpy as np
from algorithms.algorithm1 import algorithm1
from scipy.stats import norm
import matplotlib.pyplot as plt

def abs_diff_discrepancy(gen_data, data):
    """
    Absolute difference discrepancy metric
    """
    # Calculating the mean of the two datasets
    mean_gen_data = np.mean(gen_data)
    mean_data = np.mean(data)
    # Returning the absolute difference between the means
    return np.abs(mean_gen_data - mean_data)

def prior_goal1():
    """
    Prior distribution goal 1
    """
    # Simulating the prior with the parameters given for goal 1
    mu = 0
    var = 3
    return np.random.normal(mu,np.sqrt(var))

def model_goal1(theta, M):
    """
    Model for simulating data for goal 1.
    """
    # Generate a dataset of size M from the model givne in exercise 1
    a = 1
    var = 0.1
    # Probability 1/2 to be drawn from each of the distributions
    distr = np.random.randint(0,2)
    if distr == 0:
        return np.random.normal(theta, np.sqrt(var), M)
    else: 
        return np.random.normal(theta + a, np.sqrt(var), M)

def true_posterior_goal1(theta):
    """
    True closed form posterior for goal 1. 
    """
    # Clsoed form posterior as given in the exercise description
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

    

def plots_goal1():
    """
    Code for creating the plots for experiment 1
    """
    M = 100
    eps = [0.75, 0.25, 0.1, 0.025]
    data = np.zeros(M) # Observed data
    N = 500

    # True posterior curve for plots
    grid = np.linspace(-3, 3, 1000)
    true_pdf = true_posterior_goal1(grid)

    # 2x2 plot, one panel per tolerance
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, e in zip(axes, eps):
        # Run ABC rejection
        thetas, num_props = algorithm1(
            N, prior_goal1, model_goal1, M, abs_diff_discrepancy, data, e
        )
        acc = N / num_props

        # Plot accepted histogram and overlay true pdf
        ax.hist(thetas, bins=40, density=True, alpha=0.6, label="ABC accepted θ")
        ax.plot(grid, true_pdf, label="True posterior pdf")
        ax.set_title(f"ε={e} | acc={acc:.3f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plots_goal1()