import numpy as np

def exact_sim(theta):
    D = 4.0
    Ke, Ka, Cl, sigma = theta
    alpha = (D*Ka*Ke) / Cl

    times = np.array([0, 0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12])
    N = len(times)
    sim_data = np.zeros(N)

    for i in range(N-1):
        dt_i = times[i+1] - times[i]
        X_i = sim_data[i]

        # Deterministic Term
        first_det_term = np.exp(-Ke * dt_i) * X_i
        second_det_term = alpha/(Ke - Ka) * (np.exp(-Ka * times[i+1]) - np.exp(-Ke * dt_i)*np.exp(-Ka * times[i]))
        det_term = first_det_term + second_det_term

        # Stochastic Noise
        var = sigma**2/(2*Ke) * (1 - np.exp(-2*Ke*dt_i))
        std = np.sqrt(var)
        noise = np.random.normal(0,std)

        X_next = det_term + noise
        sim_data[i+1] = X_next

    return np.array(sim_data[1:])