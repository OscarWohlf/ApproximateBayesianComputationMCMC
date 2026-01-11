import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from goals.goal1 import model_goal1, abs_diff_discrepancy, true_posterior_goal1
from algorithms.algorithm2 import algorithm2


def q_proposal(theta,variance):
    """
    Proposal random walk function for the next parameter theta* given the current theta.

    Inputs:
        theta (int): Value of parameter theta in the current iteration.
        variance (int): Variance of the proposal random walk (hyperparameter to be 'tuned').
    """
    return np.random.normal(theta, np.sqrt(variance)) # Since the function takes the std, not variance.

def pi_density(theta):
    """
    Return the value of the N(0, 3) probability density function evaluated at theta.
    """
    return norm.pdf(theta, 0, np.sqrt(3))


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


def main():
    """
    Run Algorithm 2 for several ABC tolerance values, compare the resulting
    approximate posterior distributions with the true posterior, and visualise
    the effect of the proposed tolerances on the acceptance rate and posterior accuracy.
    """

    # Setting seed for reproducibility
    np.random.seed(42)

    check_ESS = 2000  # Check the ESS of the current chain every 2000 iterations.
    max_iter = 200000 # Maximum number of iterations allowed before stopping the algorithm (in case ESS doesn't reach 500)
    M = 100
    data = [0.0]

    eps = [0.75, 0.25, 0.1, 0.025] # Tolerances considered
    var = [1, 0.3, 0.25, 0.25] # Corresponding proposed variances

    grid = np.linspace(-3, 3, 1000)
    true_pdf = true_posterior_goal1(grid)
    bin_edges = np.linspace(-2, 1, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for idx, e in enumerate(eps):
        print(f"Plot {idx+1}: ε={e}")
        v = var[idx]
        # Run Algorithm 2 (ABC-based) and collect the chain and the number of accepted samples
        thetas, accepted = algorithm2(
            check_ESS, v, q_proposal, pi_density, model_goal1, M, calculate_ESS, abs_diff_discrepancy, data, e, max_iter
        )
        acc_rate = accepted / len(thetas) # acceptance rate.

        # Plot of the results
        ax = axes[idx//2, idx%2]
        ax.hist(thetas, bins=bin_edges, density=True, alpha=0.6, label="ABC accepted θ")
        ax.plot(grid, true_pdf, label="True posterior pdf")
        ax.set_title(rf"$\epsilon={e}$ | $\nu^2={v}$ | acc={acc_rate:.3f}")
        ax.set_xlim([-2, 1])
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()