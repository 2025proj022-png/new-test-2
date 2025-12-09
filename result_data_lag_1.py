import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Dictionary_Causal_Estimator import causal_direction


# -----------------------------------------------------
# LOAD .MAT DATA
# -----------------------------------------------------

data = loadmat("Dataset/causal_symbolic_data.mat")

# strengths: 0.0, 0.1, ..., 1.0
coupling_values = np.arange(0, 1.1, 0.1)

# MATLAB keys: "strength_0.0", "strength_0.1", ..., "strength_1.0"
coupling_keys = [f"strength_{v:.1f}" for v in coupling_values]

results = {float(f"{v:.1f}"): [] for v in coupling_values}


# -----------------------------------------------------
# RUN EXPERIMENT
# -----------------------------------------------------

for eta, key in zip(coupling_values, coupling_keys):
    eta_clean = float(f"{eta:.1f}")

    print(f"Processing causal strength η = {eta_clean:.1f}")

    # Access the single element of the (1, 1) MATLAB structured array
    D = data[key][0][0]

    # D["X"] and D["Y"] are (100, 300) numpy arrays of characters
    X_sets = D["X"]
    Y_sets = D["Y"]

    # Iterate over all 100 samples/rows in the dataset
    num_samples = X_sets.shape[0]

    for i in range(num_samples): 
        Xs = [str(x[0]) for x in X_sets[i]]
        Ys = [str(y[0]) for y in Y_sets[i]]
        Xs = "".join(X_sets[i])
        Ys = "".join(Y_sets[i])

        # Run your causality estimator
        value = causal_direction(Xs, Ys, analysis_print=False)
        # print(value)
        # value  = 1 means X causes Y
        # value = 2 means Y causes X
        # value  = 0 means independent
        # value  = 3 means undetermined.

        results[eta_clean].append(value)
    # print("="*30)

print("\nProcessing complete. Results:")
print(results)


# -----------------------------------------------------
# GROUPED BAR PLOT
# -----------------------------------------------------

categories = [0, 1, 2, 3]  # causal_direction outputs
counts_matrix = []

for eta in coupling_values:
    vals = results[float(f"{eta:.1f}")]
    counts_matrix.append([vals.count(c) for c in categories])

counts_matrix = np.array(counts_matrix)
print(counts_matrix)

x = np.arange(len(coupling_values))
width = 0.18
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]


plt.figure(figsize=(16, 7))

for i, cat in enumerate(categories):
    plt.bar(x + offsets[i], counts_matrix[:, i], width=width, label=f"value = {cat}")

plt.xticks(x, [f"{v:.1f}" for v in coupling_values], rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Causal Strength (η)", fontsize=25)
plt.ylabel("Count of causal_direction output", fontsize=25)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('Results/output.png')
plt.show()
