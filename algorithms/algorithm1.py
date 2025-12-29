import numpy as np

def algorithm1(N, prior, model, M, discrep, data, eps):
    """
    Simple ABC Algorithm
    """
    theta = []
    n_attempts = 0

    # Keep proposing from the prior until we have N accepted samples
    while len(theta) < N:
        # Propose a parameter from the prior
        theta_star = prior()

        # Simulate dataset
        gen_data = model(theta_star, M)

        # Compute discrepancy between simulated and observed data
        diff = discrep(gen_data, data)
        n_attempts += 1

        # Accept the sample if it is within the tollerance
        if diff < eps:
            theta.append(theta_star)
    return np.asarray(theta), n_attempts

def main():
    return None


if __name__ == "__main__":
    main()



