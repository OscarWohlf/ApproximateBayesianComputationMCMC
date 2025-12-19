from HelperFunctions import *
import matplotlib.pyplot as plt

def analyse_variances(eps, var_values, burn_in, show_trace_plots=False):

    collection_of_chains = []
    acceptance_rates = []

    for var in var_values:
        chain = [0.0]
        # Run initial iterations until the burn in time (these will be discarded --> non-stationarity)
        _ = run_chain(chain=chain, iterations=burn_in+1, var=var, eps=eps)

        chain = [chain[-1]] # Discard the previous chain, stay only with the last value
        target_ESS = 500 # Target value for ESS (Effective Sample Size).
        current_ESS = 0
        check_ESS = 2000 # Check the ESS of the current chain every 500 iterations.
        max_iter = 100000
        accepted = 0

        # Now, the actual chain running starts.
        while current_ESS < target_ESS:
            # Run the chain for 'check_ESS' iterations
            accepted += run_chain(chain=chain, iterations=check_ESS, var=var, eps=eps)
            # Calculate ESS on the current chain
            current_ESS = calculate_ESS(chain=np.array(chain), M=50)
            print(current_ESS)
            if len(chain) > max_iter:
                print("\n[Warning]: Maximum iterations reached without hitting target ESS.")
                break

        n_total_iters = len(chain)-1
        acceptance_rate = 100*(accepted / n_total_iters)

        collection_of_chains.append(chain)
        acceptance_rates.append(acceptance_rate)

    # Show trace plots
    if show_trace_plots:
        plot_trace_plots(collection_of_chains, var_values, acceptance_rates)

    for var, acc_r in zip(var_values,acceptance_rates):
        print(f"Variance: {var} --> Acceptance Rate: {acc_r}")


# Running example
eps_values = [0.75, 0.25, 0.1, 0.025]
var_values = [0.01, 0.02, 0.05, 0.1]
burn_in_values = [50, 100, 2000, 100000]

analyse_variances(eps_values[2], var_values, burn_in_values[2])