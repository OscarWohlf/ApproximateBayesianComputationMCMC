from HelperFunctions import *

# Main ABC MCMC algorithm
N = 10000
var = 0.1 # We'll have to experiment with different values of this parameter.
eps = 0.1 # We'll have to experiment with the values: 0.75, 0.25, 0.1, 0.025.
theta_chain = np.zeros(N+1)

for i in range(N):
    theta = theta_chain[i] # Current theta.
    theta_next = q(theta, var) # Sample candidate theta*
    D = P(theta_next) # Generate data from the underlying model given theta*

    alpha = pi(theta_next)/pi(theta)
    if (rho(D) < eps) and (np.random.uniform(0,1) < alpha):
        theta_chain[i+1] = theta_next
    else:
        theta_chain[i+1] = theta


print(theta_chain[-50:])