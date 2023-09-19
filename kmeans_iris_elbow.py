import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, 2:4]  # Petal width and petal height

# Initialize variables
wcss_list = []
cluster_range = range(1, 11)

# Compute K-means and calculate WCSS for different number of clusters
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, n_init=1, random_state=0)
    kmeans.fit(X)
    wcss = kmeans.inertia_  # Inertia is the within-cluster sum of squares
    wcss_list.append(wcss)

# Plotting the Elbow graph
plt.figure()
plt.plot(cluster_range, wcss_list, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("iris-elbow.png")
