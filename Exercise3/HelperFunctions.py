import numpy as np
from scipy.stats import norm


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
    Generate data from the underlying model given a set of parameters.

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
    for _ in range(iterations):
        theta = chain[-1]
        theta_next = q(theta, var)
        D = P(theta_next)

        alpha = pi(theta_next) / pi(theta)
        if (rho(D) < eps) and (np.random.uniform(0, 1) < alpha):
            chain.append(theta_next)
        else:
            chain.append(theta)
    # This modifies the chain in-place, thus no need to return it.


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