import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# reproducibility
random.seed(0)
np.random.seed(0)

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, 2:4]  # Petal width and petal height

# add in some outliers ...
#X = np.vstack((X, np.array([[7.6, 1.2], [7.1, 1.5], [7.7, 0.5], [7.0, 1.0], [7.0, 1.5]])))
#X = np.vstack((X, np.array([[12.5, -2.0]])))

# Initialize parameters
n_clusters = 3
max_iterations = 5
centers = np.array([[7.0, 0.0], [4.0, 1.0], [1.0, 2.5]])
colors = ["blue", "orange", "green"]

# Iterative K-means
for iteration in range(max_iterations):
    # Step 1: Assign each data point to the nearest center
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    
    # Plot current state
    plt.figure(figsize=(6, 6))

    plt.title(f"Iteration {iteration + 1}")
    plt.xlabel("Petal Width")
    plt.ylabel("Petal Height")

    # plot the cluster centroids
    plt.scatter(centers[:, 0], centers[:, 1], c=colors, marker="X", edgecolors="red", s=300, label="Centroids")
    
    # plot the points
    for i in range(n_clusters):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f"Cluster {i}")
    
    # save the figure
    plt.savefig(f"./iris-iter{iteration}.png")
    
    # Step 2: Update centers
    for i in range(n_clusters):
        centers[i] = np.mean(X[labels == i], axis=0)
    
    # plot the new centroids
    plt.scatter(centers[:, 0], centers[:, 1], c=colors, marker="X", edgecolors="black", s=300, label="Centroids")

    # save the figure
    plt.savefig(f"./iris-iter{iteration}-next.png")


