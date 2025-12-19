from HelperFunctions import *
import matplotlib.pyplot as plt

# ABC MCMC Algorithm with no burn-in time
chain = [0.0]
_ = run_chain(chain=chain, iterations=50000, var=0.015, eps=0.025)
approx_mean_value=np.mean(chain[3000:])

# Study on the trace plot of the chain with different burn-in times
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
eps_values = [0.75, 0.25, 0.1, 0.025]
# Take some sample variance for every epsilon.
var_values = [0.1, 0.05, 0.02, 0.01] # We analyse the variance in more detail in the Exercise3.py code.
# Number of iterations for every tolerance, chosen after a careful visual evaluation on the convergence of each chain.
N_values = [200, 500, 10000, 100000]

for i, eps in enumerate(eps_values):
    theta_chain = [0.0]
    N = N_values[i]
    var = var_values[i]
    _ = run_chain(chain=theta_chain, iterations=N, var=var, eps=eps)

    ax = axes[i//2, i%2]
    ax.plot(theta_chain, color='#2c3e50', lw=0.5)
    ax.axhline(y=approx_mean_value, color='red', linestyle='--', alpha=0.3, label='Approx. Posterior Mean')
    ax.set_title(rf"Tolerance $\epsilon = {eps}$")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel(r"$\theta$")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("BurnInTimeAnalysis.png")
plt.show()

# After analysing, we select:
# eps: 0.75 --> burn-in time: 50
# eps: 0.25 --> burn-in time: 100
# eps: 0.1 --> burn-in time: 500
# eps: 0.025 --> burn-in time: 5000