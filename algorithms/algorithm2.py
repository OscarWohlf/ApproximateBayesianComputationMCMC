import numpy as np

def algorithm2(check_ESS, var, q_proposal, pi_density, model, M, calculate_ESS, discrep, data, eps, max_iter):
    theta = [0.0]
    accepted = 0
    current_ESS = 0.0
    target_ESS = 500.0
    n_iter = 0

    while current_ESS < target_ESS:
        for _ in range(check_ESS):
            theta_curr = theta[-1]
            theta_star = q_proposal(theta_curr, var)
            gen_data = model(theta_star, M)
            diff = discrep(gen_data, data)

            alpha = pi_density(theta_star) / pi_density(theta_curr)

            if (diff < eps) and (np.random.uniform(0, 1) < alpha):
                theta.append(theta_star)
                accepted += 1
            else:
                theta.append(theta_curr)

            n_iter += 1
            if n_iter > max_iter:
                return np.asarray(theta), accepted

        current_ESS = calculate_ESS(np.array(theta))

    return np.asarray(theta), accepted

def main():
    return None


if __name__ == "__main__":
    main()