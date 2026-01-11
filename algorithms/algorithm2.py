import numpy as np

def algorithm2(check_ESS, var, q_proposal, pi_density, model, M, calculate_ESS, discrep, data, eps, max_iter):
    """
    ABC-MCMC Algorithm
    """
    theta = [0.0]
    accepted = 0
    current_ESS = 0.0
    target_ESS = 500.0
    n_iter = 0

    # Keep generating samples until we reach an ESS of 500.
    while current_ESS < target_ESS:
        for _ in range(check_ESS):
            theta_curr = theta[-1]

            # Propose a new theta
            theta_star = q_proposal(theta_curr, var)

            # Simulate dataset
            gen_data = model(theta_star, M)
            diff = discrep(gen_data, data)

            # Metropolis-Hastings acceptance probability
            alpha = pi_density(theta_star) / pi_density(theta_curr)

            if (diff < eps) and (np.random.uniform(0, 1) < alpha):
                theta.append(theta_star)
                accepted += 1
            else:
                theta.append(theta_curr)

            n_iter += 1

            # Case where algorithm doesn't reach ESS = 500.
            if n_iter > max_iter:
                print(f"[Warning]: Iteration limit reached. Using N_eff = {current_ESS}")
                return np.asarray(theta), accepted

        # After every "check_ESS" = 2000 iterations, recompute the ESS on the chain.
        current_ESS = calculate_ESS(np.array(theta))

    return np.asarray(theta), accepted

def main():
    return None


if __name__ == "__main__":
    main()