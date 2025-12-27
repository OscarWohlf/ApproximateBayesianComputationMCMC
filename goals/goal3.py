import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from goals.goal1 import model_g1, abs_diff_discrepancy, true_posterior_g1
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
    ...
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
    check_ESS = 2000  # Check the ESS of the current chain every 500 iterations.
    max_iter = 200000
    M = 100
    data = [0.0]

    eps = [0.75, 0.25, 0.1, 0.025]
    var = [1, 0.5, 0.3, 0.3]

    grid = np.linspace(-3, 3, 1000)
    true_pdf = true_posterior_g1(grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for idx, e in enumerate(eps):
        print(f"Plot {idx+1}: ε={e}")
        v = var[idx]
        thetas, accepted = algorithm2(
            check_ESS, v, q_proposal, pi_density, model_g1, M, calculate_ESS, abs_diff_discrepancy, data, e, max_iter
        )
        acc_rate = accepted / len(thetas)

        ax = axes[idx//2, idx%2]
        ax.hist(thetas, bins=40, density=True, alpha=0.6, label="ABC accepted θ")
        ax.plot(grid, true_pdf, label="True posterior pdf")
        ax.set_title(rf"ε={e} | $\nu^2$={v} | acc={acc_rate:.3f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"TryDiffVars.png")
    plt.show()

if __name__ == "__main__":
    main()