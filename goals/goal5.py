import numpy as np
from goals.goal4 import *
from scipy.stats import norm
import matplotlib.pyplot as plt

def discrep_goal5(s_star, s_obs):
    theta_0 = np.array([0.07,1.15,0.05,0.33])
    s_diff = s_star - s_obs
    return np.sqrt(np.sum((s_diff / theta_0)**2))

def q_proposal_goal5(log_theta_curr, prop_sd):
    rvs = np.random.normal(0,prop_sd,4)
    next_log_thetas = log_theta_curr + rvs
    return next_log_thetas

def pi_density_goal5(log_theta_star):
    Ke, Ka, Cl, sigma = log_theta_star
    log_Ke = norm.logpdf(Ke, -2.7, 0.6)
    log_Ka = norm.logpdf(Ka,0.14, 0.4)
    log_Cl = norm.logpdf(Cl,-3,0.8)
    log_sig = norm.logpdf(sigma, -1.1, 0.3)
    return log_Ke + log_Ka + log_Cl + log_sig

def algorithm2_goal5(N, prop_sd, q_proposal, pi_density, model, S, discrep, obs_data, eps, theta0 = [0.07,1.15,0.05,0.33]):
    theta = [theta0]
    accepted = 0
    sum_obs_data = S(obs_data)

    for i in range(N):
        theta_curr = theta[-1]
        log_theta_curr = np.log(theta_curr)
        log_theta_star = q_proposal(log_theta_curr, prop_sd)
        theta_star = np.exp(log_theta_star)
        gen_data = model(theta_star)
        sum_gen_data = S(gen_data)
        diff = discrep(sum_gen_data, sum_obs_data)

        log_alpha_cond = pi_density(log_theta_star) - pi_density(log_theta_curr)
        alpha = min(0.0, log_alpha_cond)
        if (diff < eps) and (np.log(np.random.uniform()) < alpha):
            theta.append(theta_star)
            accepted += 1
        else:
            theta.append(theta_curr)

    return np.asarray(theta), accepted

def summary_stat_goal5(betas):
    def summary_stat(data):
        x = np.zeros(10)
        x[0] = 1.0
        x[1:] = data
        return x @ betas
    return summary_stat

def main():
    eps_list = [0.25, 0.7, 1.0]
    thetas_eps = {}
    acc_eps = {}
    prop_sd = np.array([0.15, 0.10, 0.20, 0.08])
    given_theta = np.array([0.08,1.5,0.04,0.2])
    data_obs = simulate_dataset(theta=given_theta, method='exact')

    N = 10000
    X, Y = create_dataset(2000, method='exact')
    betas = get_betas(X,Y)
    sum_stat = summary_stat_goal5(betas)
    for e in eps_list:
        thetas, accepted = algorithm2_goal5(
            N, prop_sd, q_proposal_goal5, pi_density_goal5,
            exact_sim, sum_stat, discrep_goal5, data_obs, e
        )
        thetas_eps[e] = thetas
        acc_eps[e] = accepted / N
        print(f"eps={e} acc={acc_eps[e]:.3f}")

        
    param_names = ["Ke", "Ka", "Cl", "sigma"]
    burn = 2000  # adjust or set to 0
    bins = 40

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex='col')

    for r, e in enumerate(eps_list):
        samples = thetas_eps[e][burn:]  # shape (N+1-burn, 4)

        for c in range(4):
            ax = axes[r, c]
            ax.hist(samples[:, c], bins=bins, density=True, alpha=0.7)

            if r == 0:
                ax.set_title(param_names[c])
            if c == 0:
                ax.set_ylabel(f"ε={e}\n(density)")
            if r == 2:
                ax.set_xlabel("value")

    # add acceptance rates on the left side
    for r, e in enumerate(eps_list):
        axes[r, 0].text(
            0.02, 0.95, f"acc={acc_eps[e]:.3f}",
            transform=axes[r, 0].transAxes,
            va="top"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()