from HelperFunctions import *
import matplotlib.pyplot as plt


var = 0.1 # We'll have to experiment with different values of this parameter.
eps = 0.1 # We'll have to experiment with the values: 0.75, 0.25, 0.1, 0.025.
theta_chain = [0]

burn_in = 50

# Run initial iterations until the burn in time (these will be discarded --> non-stationarity)
run_chain(chain=theta_chain, iterations=burn_in+1, var=var, eps=eps)

theta_chain = [theta_chain[-1]] # Discard the previous chain, stay only with the last value
target_ESS = 500 # Target value for ESS (Effective Sample Size).
current_ESS = 0
check_ESS = 1000 # Check the ESS of the current chain every 500 iterations.
max_iter = 100000

# Now, the actual chain running starts.
while current_ESS < target_ESS:
    # Run the chain for 'check_ESS' iterations
    run_chain(chain=theta_chain, iterations=check_ESS, var=var, eps=eps)
    # Calculate ESS on the current chain
    current_ESS = calculate_ESS(chain=np.array(theta_chain), M=100)

    if len(theta_chain) > max_iter:
        print("Maximum iterations reached without hitting target ESS.")
        break


# Show trace Plot
plt.figure(figsize=(15, 5))
plt.plot(theta_chain, color='steelblue', lw=0.5)
plt.title(f"Trace Plot")
plt.xlabel("Iteration")
plt.ylabel(r"$\theta$")

plt.tight_layout()
plt.show()