import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('cnn_model_20250204_233721.csv')

# Normalize the true values to be in the range 0..1
ref = data['Expected'] * 100
tst = data['Predicted'] *100 # Predicted values

# Compute RMSE
rmse = np.sqrt(np.mean((ref - tst) ** 2.0))
print("RMSE:", rmse)

# Compute Pearson Correlation Coefficient (CC)
cc = np.corrcoef(ref, tst)[0, 1]
print("Correlation Coefficient (CC):", cc)

# Plot scatter plot
plt.plot(ref, tst, '.', alpha=0.5)
plt.xlabel('True Intelligibility')
plt.ylabel('Tested Intelligibility')
plt.title(f'True vs. Tested (CC = {cc:.2f}) (RMSE = {rmse:.2f})')

plt.show()
