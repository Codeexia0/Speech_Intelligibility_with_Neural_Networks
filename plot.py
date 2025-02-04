import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cnn_model_20250204_143136.csv')

# 'true_value' column is divided by 100 here, so if the original data ranged 0..100,
# now 'ref' will be 0..1
ref = data['Expected']
# 'tst' is presumably your predicted values (e.g., STOI or some other measure).
tst = data['Predicted']

# Compute RMSE between ref and tst
rmse = np.sqrt(np.mean((ref - tst) ** 2.0))
print("RMSE:", rmse)

# Plot a scatter of ref (x-axis) vs. tst (y-axis).
# Using `plt.plot(..., '.', ...)` draws a scatter with '.' markers.
plt.plot(ref, tst, '.')
plt.xlabel('True Intelligibility')
plt.ylabel('Tested Intelligibility')
plt.title('True vs. Tested')

plt.show()