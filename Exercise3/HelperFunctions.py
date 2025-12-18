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