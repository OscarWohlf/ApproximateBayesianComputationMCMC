import numpy as np

def algorithm1(N, prior, model, M, discrep, data, eps):
    theta = []
    n_attempts = 0

    while len(theta) < N:
        theta_star = prior()
        gen_data = model(theta_star, M)
        diff = discrep(gen_data, data)
        n_attempts += 1

        if diff < eps:
            theta.append(theta_star)
    return np.asarray(theta), n_attempts

def main():
    return None


if __name__ == "__main__":
    main()



