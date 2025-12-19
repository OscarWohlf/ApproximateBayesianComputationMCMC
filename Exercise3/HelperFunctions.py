import numpy as np


def prior_g1():
    """
    Prior distribution of a parameter theta.

    Input:
        theta (int): Value of parameter theta in the current iteration.
    """
    mu = 0
    var = 3
    return np.random.normal(mu, np.sqrt(var))


def model_g1(theta, M):
    """
    Generate data from the underlying model given a parameter theta.

    Input:
        theta (float): Value of parameter theta in the current iteration.

    Output:
        P(·|theta) (np.array): Generated data given theta.
    """
    a = 1
    var = 0.1
    distr = np.random.randint(0, 2)
    if distr == 0:
        return np.random.normal(theta, np.sqrt(var), M)
    else:
        return np.random.normal(theta + a, np.sqrt(var), M)


def abs_diff_discrepancy(gen_data, data):
    """
    Discrepancy metric between the summary statistics (mean for this problem) of a
    data sample D* (generated given theta*) and of the observed data D (assumed to be 0)

    Inputs:
        D (np.array): Data sample
    """

    mean_gen_data = np.mean(gen_data)
    mean_data = np.mean(data)
    return np.abs(mean_gen_data - mean_data)


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

    return alpha * norm.pdf(theta, loc=mu_1, scale=std) + (1 - alpha) * norm.pdf(theta, loc=mu_2, scale=std)


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