# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, pearsonr


# Load the data
file_path = "cnn_model_20250206_110633.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Normalize the values to be in percentage (0-100)
ref = data['Expected'] * 100  # True correctness percentage
tst = data['Predicted'] * 100   # Predicted correctness percentage

# Compute metrics
rmse = root_mean_squared_error(ref, tst)
mae = mean_absolute_error(ref, tst)
r2 = r2_score(ref, tst)
tau, _ = kendalltau(ref, tst)
cc, _ = pearsonr(ref, tst)

# Print the metrics
print(f"RMSE: {rmse:.4f}")
print(f"CC: {cc:.4f}")
print(f"Tau: {tau:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# Create a square figure (8x8 inches)
plt.figure(figsize=(6, 6))

# Plot the data points as dots
plt.plot(
    ref, 
    tst, 
    ".", 
    # color="blue"
)


# Make a trendline
# Overlay a smoothed trend line using LOWESS smoothing with Seaborn's regplot
sns.regplot(
    x=ref,
    y=tst,
    scatter=False,              # Do not replot scatter points
    lowess=True,                # Enable LOWESS smoothing
    ci=None,                    # Disable confidence interval shading
    color="#646464",      # Alternate to rgb, use #646464 color codes or just color names
    line_kws={"linestyle": "--", "linewidth": 2}
)

# Formatting the plot
plt.xlabel('True [%]', fontweight='bold')
plt.ylabel('Predicted [%]', fontweight='bold')
plt.title(f'True vs. Predicted Speech Intelligibility', fontweight='bold')

# Optionally, set the x and y limits to ensure both axes range from 0 to 100
plt.xlim(-1, 101)
plt.ylim(-1, 101)

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

plt.show()