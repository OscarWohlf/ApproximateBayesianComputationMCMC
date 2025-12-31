import numpy as np
from scipy.stats import norm

def efficient_estimator_goal6(posterior_samples, obs_data, threshold=4.8, dt=3.0):
    """
    implements the Rao-Blackwellized Monte Carlo estimator for Goal 6.
    
    Inputs:
        posterior_samples: N x 4 array of [Ke, Ka, Cl, sigma] from Goal 5.
        obs_data: the 9 observed values {x1...x9}. The last value is X12.
        threshold: the concentration threshold c = 4.8.
        dt: time difference between T=12 and T=15 (3 hours).
    """
    D = 4.0
    # X12 is the last element of the observed data sequence
    x12 = obs_data[-1] 
    t_start = 12.0
    t_end = 15.0
    
    p_values = []
    
    for theta in posterior_samples:
        Ke, Ka, Cl, sigma = theta
        alpha = (D * Ka * Ke) / Cl
        
        # calculate the analytical Mean at T=15 given X12
        # formula based on exact_sim.py transition logic
        mean_term1 = np.exp(-Ke * dt) * x12
        mean_term2 = (alpha / (Ke - Ka)) * (np.exp(-Ka * t_end) - np.exp(-Ke * dt) * np.exp(-Ka * t_start))
        mu_15 = mean_term1 + mean_term2
        
        # calculate the analytical Variance at T=15
        # formula based on exact_sim.py stochastic noise logic
        var_15 = (sigma**2 / (2 * Ke)) * (1 - np.exp(-2 * Ke * dt))
        std_15 = np.sqrt(var_15)
        
        # calculate P(X15 > c | theta, X12) using the Normal Survival Function
        # this is 1 - CDF
        p_i = norm.sf(threshold, loc=mu_15, scale=std_15)
        p_values.append(p_i)
        
    # final estimator: average of the probabilities
    return np.mean(p_values)
