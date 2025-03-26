import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the dataset
data = pd.read_csv('cnn_model_20250204_143136.csv')

# Extract true and predicted intelligibility values and convert to percentage (0-100)
true_intelligibility = data['Expected'] * 100
predicted_intelligibility = data['Predicted'] * 100

# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Set up the colormap (using 'viridis')
cmap = cm.get_cmap("viridis")

# ----------------------------
# Histogram for Predicted Intelligibility (without annotations)
# ----------------------------
ax = axes[0]
# Compute the histogram for predicted values
hist_values_pred, bin_edges_pred, patches_pred = ax.hist(
    predicted_intelligibility, bins=50, density=False, edgecolor='black'
)

# Normalize and color each bar in the predicted histogram
norm_pred = plt.Normalize(vmin=min(hist_values_pred), vmax=max(hist_values_pred))
for patch, value in zip(patches_pred, hist_values_pred):
    patch.set_facecolor(cmap(norm_pred(value)))

# Labeling and title for predicted histogram
ax.set_xlabel('Predicted Intelligibility [%]')
ax.set_ylabel('# Occurrences [-]')
ax.set_title('Distribution of Predicted Intelligibility')

# ----------------------------
# Histogram for True Intelligibility (without annotations)
# ----------------------------
ax = axes[1]
# Compute the histogram for true values
hist_values_true, bin_edges_true, patches_true = ax.hist(
    true_intelligibility, bins=50, density=False, edgecolor='black'
)

# Normalize and color each bar in the true histogram
norm_true = plt.Normalize(vmin=min(hist_values_true), vmax=max(hist_values_true))
for patch, value in zip(patches_true, hist_values_true):
    patch.set_facecolor(cmap(norm_true(value)))

# Labeling and title for true histogram
ax.set_xlabel('True Intelligibility [%]')
ax.set_ylabel('# Occurrences [-]')
ax.set_title('Distribution of True Intelligibility')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()