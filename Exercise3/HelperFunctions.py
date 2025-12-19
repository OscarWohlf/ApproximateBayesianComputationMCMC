import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def q(theta,variance):
    """
    Proposal random walk function for the next parameter theta* given the current theta.

    Inputs:
        theta (int): Value of parameter theta in the current iteration.
        variance (int): Variance of the proposal random walk (hyperparameter to be 'tuned').
    """
    return np.random.normal(theta, np.sqrt(variance)) # Since the function takes the std, not variance.

def pi(theta):
    """
    Prior distribution of a parameter theta.

    Input:
        theta (int): Value of parameter theta in the current iteration.
    """
    std = np.sqrt(3)
    return norm.pdf(theta, 0, std) # Again, the function takes the std, not variance.

def P(theta, M=100):
    """
    Generate data from the underlying model given a parameter theta.

    Input:
        theta (float): Value of parameter theta in the current iteration.

    Output:
        P(·|theta) (np.array): Generated data given theta.
    """
    a = 1
    std1 = np.sqrt(0.1)

    # Randomly choose component 0 or 1 for all M samples at once
    components = np.random.binomial(1, 0.5, size=M)

    means = theta + components * a
    return np.random.normal(means, std1)


def rho(D):
    """
    Discrepancy metric between the summary statistics (mean for this problem) of a
    data sample D* (generated given theta*) and of the observed data D (assumed to be 0)

    Inputs:
        D (np.array): Data sample
    """

    return np.abs(np.mean(D))


def run_chain(chain, iterations, var=0.1, eps=0.1):
    accepted = 0
    for _ in range(iterations):
        theta = chain[-1]
        theta_next = q(theta, var)
        D = P(theta_next)

        alpha = pi(theta_next) / pi(theta)
        if (rho(D) < eps) and (np.random.uniform(0, 1) < alpha):
            chain.append(theta_next)
            accepted += 1
        else:
            chain.append(theta)

    # This modifies the chain in-place, thus no need to return it.
    return accepted


def calculate_ESS(chain, M=100):
    """
    Calculate the Effective Sample Size of a given chain.
    """
    N = len(chain)

    # Determine block size T
    T = N // M

    # Ensure the chain length is a multiple of M for precise reshaping
    if N % M != 0:
        N = M * T
        chain = chain[-N:]

    # Reshape into matrix MxT --> each row represents a batch
    blocks = chain.reshape(M, T)
    # Compute the mean over batches
    mu_i = np.mean(blocks, axis=1)

    # Estimate sigma_MCMC^2 = T * Sample_Var(batch_means)
    sigma2_mu1 = np.var(mu_i, ddof=1) # ddof=1 --> denominator is 1/(M-1) (as the slides!)
    sigma2_MCMC = T * sigma2_mu1

    # Calculate c(0) --> variance of the chain
    c0 = np.var(chain, ddof=1)

    # Final check --> 0 variance (in the weird case all bath means were the same)
    if sigma2_MCMC == 0:
        return 0

    # Final ESS calculation: ESS = N * (c0 / sigma2_MCMC)
    ESS = N * (c0 / sigma2_MCMC)

    return ESS


def plot_trace_plots(collection_of_chains, var_values, acceptance_rates):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, var in enumerate(var_values):
        var = var_values[i]
        chain = collection_of_chains[i]
        acceptance_rate = acceptance_rates[i]

        axes_idx = axes[i // 2, i % 2]
        axes_idx.plot(chain, color='steelblue', lw=0.5)
        axes_idx.set_title(rf"Trace plot with $\nu^2 = {var}$ --> Acceptance Rate = {acceptance_rate}")
        axes_idx.set_xlabel("Iteration index")
        axes_idx.set_ylabel(r"$\theta$")
        axes_idx.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("VarianceAnalysis.png")
    plt.show()

def plot_hist_vs_post_distribution(collection_of_chains, var_values, acceptance_rates):
    # Parameters from true posterior distribution (equation 3 in the project description PDF)
    M, sigma1_sq, sigma_sq, a, x_bar = 100, 0.1, 3, 1, 0

    # Calculate Posterior Parameters
    posterior_var = sigma1_sq / (M + sigma1_sq/sigma_sq)
    posterior_std = np.sqrt(posterior_var)
    mu1 = 0 # Since x_bar = 0
    mu2 = (sigma_sq / (sigma_sq + sigma1_sq / M)) * (x_bar - a)
    alpha = 1 / (1 + np.exp(a * (x_bar - a / 2) * (M / (M * sigma_sq + sigma1_sq))))

    # PDF range
    x = np.linspace(-1.5, 0.5, 10000)
    true_pdf = alpha * norm.pdf(x, mu1, posterior_std) + (1 - alpha) * norm.pdf(x, mu2, posterior_std)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, var in enumerate(var_values):
        ax = axes[i//2, i%2]
        ax.hist(collection_of_chains[i], bins=100, density=True, alpha=0.6, color='steelblue', label='ABC-MCMC')
        ax.plot(x, true_pdf, 'r-', lw=2, label='True Posterior')
        ax.set_title(rf"$\nu^2 = {var}$ | Acc: {acceptance_rates[i]:.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("VarianceAnalysis.png")
    plt.show()