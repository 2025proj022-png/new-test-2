import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from Dictionary_Causal_Estimator import get_ccm_results 

def map_cause_string_to_int(cause_string):
    """Maps the 'cause' string output to the integer categories for plotting."""
    if cause_string == 'x':
        return 1  # X causes Y
    if cause_string == 'y':
        return 2  # Y causes X
    if cause_string == 'n_or_m':
        return 3  # Undetermined/Mutual/No-causality
    return 0 

def plot_causality_results(coupling_values, counts_matrix, method_name):
    """Generates a grouped bar plot for a specific method."""
    
    categories = [0, 1, 2, 3]
    
    x = np.arange(len(coupling_values))
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    plt.figure(figsize=(16, 7))

    for i, cat in enumerate(categories):
        if i < counts_matrix.shape[1]:
            plt.bar(x + offsets[i], counts_matrix[:, i], width=width, label=f"value = {cat}")

    plt.xticks(x, [f"{v:.1f}" for v in coupling_values], rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Causal Strength (η)", fontsize=22)
    plt.ylabel(f"Count of {method_name} Output", fontsize=22)
    plt.title(f"Causal Direction Results using {method_name}", fontsize=24)
    plt.legend(fontsize=18)
    plt.tight_layout()

    import os

    if not os.path.exists("Results"):
        os.makedirs("Results")
    
    plt.savefig(f'Results/{method_name}_output.png')
    plt.close()

# -----------------------------------------------------
# LOAD .MAT DATA AND INITIALIZE
# -----------------------------------------------------

data = loadmat("Dataset/causal_symbolic_data.mat") 

coupling_values = np.arange(0, 1.1, 0.1)
coupling_keys = [f"strength_{v:.1f}" for v in coupling_values]

# Separate results dictionaries for each method
results_ETCP = {float(f"{v:.1f}"): [] for v in coupling_values}
results_ETCE = {float(f"{v:.1f}"): [] for v in coupling_values}
results_LZP = {float(f"{v:.1f}"): [] for v in coupling_values}


# -----------------------------------------------------
# RUN EXPERIMENT
# -----------------------------------------------------

for eta, key in zip(coupling_values, coupling_keys):
    eta_clean = float(f"{eta:.1f}")

    print(f"Processing causal strength η = {eta_clean:.1f}")

    D = data[key][0][0]
    X_sets = D["X"]
    Y_sets = D["Y"]

    num_samples = X_sets.shape[0]

    for i in range(num_samples): 
        Xs = "".join(X_sets[i])
        Ys = "".join(Y_sets[i])

        ccm_results = get_ccm_results(Xs, Ys)

        results_ETCP[eta_clean].append(map_cause_string_to_int(ccm_results['ETCP_cause']))
        results_ETCE[eta_clean].append(map_cause_string_to_int(ccm_results['ETCE_cause']))
        results_LZP[eta_clean].append(map_cause_string_to_int(ccm_results['LZP_cause']))

print("\nProcessing complete. Generating plots.")

# -----------------------------------------------------
# GENERATE COUNTS AND PLOTS
# -----------------------------------------------------

results_mapping = {
    'ETCP': results_ETCP, 
    'ETCE': results_ETCE, 
    'LZP': results_LZP
}
categories = [0, 1, 2, 3]

for method_name, results_dict in results_mapping.items():
    counts_matrix = []
    
    for eta in coupling_values:
        vals = results_dict[float(f"{eta:.1f}")]
        counts_matrix.append([vals.count(c) for c in categories])

    counts_matrix = np.array(counts_matrix)
    
    plot_causality_results(coupling_values, counts_matrix, method_name)
    print(f"Generated plot: {method_name}_output.png")

print("\nAll plots generated successfully.")