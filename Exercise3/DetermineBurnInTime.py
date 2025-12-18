from HelperFunctions import *
import matplotlib.pyplot as plt

# ABC MCMC Algorithm with no burn-in time
N = 100000
var = 0.1
eps = 0.1
theta_chain = [0]

_ = run_chain(chain=theta_chain, iterations=N, var=var, eps=eps)

# Study on the trace plot of the chain with different burn-in times
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
burn_in_values = [1, 5, 50, 500]

for i, b in enumerate(burn_in_values):
    iterations = range(b, len(theta_chain))
    clean_chain = theta_chain[b:]

    axes_idx = axes[i//2,i%2]
    axes_idx.plot(iterations, clean_chain, color='#2c3e50', lw=0.5)
    axes_idx.set_title(f"Burn-in: {b} iterations (Remaining: {len(clean_chain)})")
    axes_idx.set_xlabel("Iteration index")
    axes_idx.set_ylabel(r"$\theta$")
    axes_idx.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("BurnInTimeAnalysis.png")
plt.show()