import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def detect_outliers_z_score(data, threshold=2):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Compute Z-scores
    z_scores = [(x - mean) / std_dev for x in data]
    
    # Find and return outliers
    outliers = [data[i] for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    return outliers, z_scores

# Test the function
data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 10, 30, 12, 14, 13, 12, 10, 10, 11, 12, 15, 13]

# TODO: show multi-modal data (doesn't quite work). Why?
# iris = load_iris()
# data = iris.data[:, 2]

outliers, z_scores = detect_outliers_z_score(data)

# Create the plot
plt.figure(figsize=(10, 6))
for i, z in enumerate(z_scores):
    if abs(z) > 2:
        plt.scatter(i, z, color="red")
    else:
        plt.scatter(i, z, color="blue")

plt.axhline(2, color="grey", linestyle="--")
plt.axhline(-2, color="grey", linestyle="--")

plt.title("Z-Scores of Data Points")
plt.xlabel("Index")
plt.ylabel("Z-Score")
plt.savefig("z-score.png")

print("Outliers:", outliers)
