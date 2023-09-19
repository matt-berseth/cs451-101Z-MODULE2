import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
np.random.seed(0)

# Load MNIST dataset from OpenML
logging.info("Checking if mnist data exists ...")
if not os.path.exists(".data"):
    logging.info("mnist data does not exist, downloading and saving locally ...")
    mnist = fetch_openml("mnist_784")

    # for now, only interested in 1000 images
    n = 10000
    os.makedirs(".data", exist_ok=True)
    np.savetxt(os.path.join(".data", "y.txt"), mnist.target[:n], fmt="%s")
    np.savetxt(os.path.join(".data", "X.txt"), mnist.data[:n], fmt="%1.2f")

logging.info("loading mnist data ...")
X = np.loadtxt(os.path.join(".data", "X.txt"))
y = np.loadtxt(os.path.join(".data", "y.txt"))
logging.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Perform K-means clustering on 2D data
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X_reduced)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# TODO: show the points without the labels

# Visualize clusters in 2D space
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, s=50, cmap="viridis")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", s=300, alpha=0.75)
plt.title("K-means Clustering on 2D MNIST Data")
plt.savefig(f"./digits.png")

# TODO: how well does this work as a classifier?