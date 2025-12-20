import numpy as np


def discrep_goal5(s_star, s_obs):
    theta_0 = np.array([0.07,1.15,0.05,0.33])
    s_diff = s_star - s_obs
    return np.sqrt(np.sum((s_diff / theta_0)**2))

