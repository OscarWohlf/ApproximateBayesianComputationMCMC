import numpy as np
from goals.goal4 import *

def discrep_goal5(s_star, s_obs):
    theta_0 = np.array([0.07,1.15,0.05,0.33])
    s_diff = s_star - s_obs
    return np.sqrt(np.sum((s_diff / theta_0)**2))

def algorithm2_goal5(N, var, q_proposal, pi_density, model, S, discrep, obs_data, eps, theta0 = [0.07,1.15,0.05,0.33]):
    theta = [theta0]
    accepted = 0

    for i in range(N):
        theta_curr = theta[-1]
        theta_star = q_proposal(theta_curr, var)
        gen_data = model(theta_star)
        sum_gen_data = S(gen_data)
        sum_obs_data = S(obs_data)
        diff = discrep(gen_data, sum_obs_data)

        alpha_cond = pi_density(theta_star)  / pi_density(theta_curr)
        alpha = min(1, alpha_cond)
        if (diff < eps) and (np.random.uniform(0, 1) < alpha):
            theta.append(theta_star)
            accepted += 1
        else:
            theta.append(theta_curr)

    return np.asarray(theta), accepted

def main():
    X, Y = create_dataset(2000, 0.01)
    betas = get_betas(X,Y)

if __name__ == "__main__":
    main()