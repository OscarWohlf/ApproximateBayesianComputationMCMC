import numpy as np

def euler_maruyama(theta, dt):
    D = 4.0
    Ke, Ka, Cl, sigma = theta
    alpha = (D*Ka*Ke) / Cl

    times = np.array([0, 0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12])
    T = times[-1]
    N = int(np.ceil(T/dt))
    xs = np.zeros(N+1)
    ts = np.linspace(0.0, N*dt, N+1)

    for i in range(N):
        prev_x = xs[i]
        prev_t = ts[i]

        # Deterministic Term
        exp_term = np.exp(-Ka * prev_t)
        b_val = alpha * exp_term - (Ke * prev_x)
        det_term = b_val * dt

        # Stochastic Noise
        dW = np.random.normal(0,np.sqrt(dt))
        noise = sigma * dW

        new_x = prev_x + det_term + noise
        xs[i+1] = new_x

    idx = np.round(times / dt).astype(int)
    return xs[idx][1:]


