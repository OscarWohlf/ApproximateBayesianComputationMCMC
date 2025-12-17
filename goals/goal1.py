import numpy as np
from algorithms.algorithm1 import algorithm1

def abs_diff_discrepancy(gen_data, data):
    mean_gen_data = np.mean(gen_data)
    mean_data = np.mean(data)
    return np.abs(mean_gen_data - mean_data)

def prior_g1():
    mu = 0
    var = 3
    return np.random.normal(mu,np.sqrt(var))

def model_g1(theta, M):
    a = 1
    var = 0.1
    data = np.zeros(M)
    distrs = np.random.randint(0,2,M)
    for i in range(M):
        point = 0
        if distrs[i] == 0:
            point = np.random.normal(theta, np.sqrt(var))
        else: 
            point = np.random.normal(theta + a, np.sqrt(var))
        data[i] = point 
    return data


def main():
    M_1 = 100
    eps = [0.75, 0.25, 0.1, 0.025]
    data = np.zeros(M_1)
    for e in eps:
        res, num_props = algorithm1(500, prior_g1, model_g1, M_1, abs_diff_discrepancy, data, e)
        print(500 / num_props)

if __name__ == "__main__":
    main()