# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np  # Import numpy to use np.ndarray

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Find the mode of the 'petal width (cm)' column
mode_petal_width_result = stats.mode(X['petal width (cm)'])

# Print the ModeResult object to inspect it
print("Mode result:", mode_petal_width_result)

# Check if 'mode' and 'count' attributes are arrays or scalars
if isinstance(mode_petal_width_result.mode, (list, np.ndarray)):
    mode_petal_width = mode_petal_width_result.mode[0]  # Mode value
else:
    mode_petal_width = mode_petal_width_result.mode  # Directly take the mode value

# Similarly for 'count'
if isinstance(mode_petal_width_result.count, (list, np.ndarray)):
    frequency_of_mode = mode_petal_width_result.count[0]  # Frequency of mode
else:
    frequency_of_mode = mode_petal_width_result.count  # Directly take the count value

# Print the mode and its frequency
print(f"Mode of petal width: {mode_petal_width}")
print(f"Frequency of mode: {frequency_of_mode}")

# Plot histogram of 'petal width (cm)' with a vertical line for the mode
plt.figure(figsize=(8, 6))
plt.hist(X['petal width (cm)'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(mode_petal_width, color='red', linestyle='dashed', linewidth=2, label=f'Mode: {mode_petal_width}')
plt.title('Distribution of Petal Width (cm) with Mode', fontsize=15)
plt.xlabel('Petal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
