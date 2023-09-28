import logging
import os
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from autoencode import decode, encode

# Fetch MNIST data
# Load MNIST data from OpenML
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
np.random.seed(0)

# Load MNIST dataset from OpenML
logging.info("Checking if mnist data exists ...")
if not os.path.exists(".data"):
    logging.info("mnist data does not exist, downloading and saving locally ...")
    mnist = fetch_openml("mnist_784")

    # for now, only interested in 1000 images
    os.makedirs(".data", exist_ok=True)
    np.savetxt(os.path.join(".data", "y.txt"), mnist.target, fmt="%s")
    np.savetxt(os.path.join(".data", "X.txt"), mnist.data, fmt="%1.2f")

logging.info("loading mnist data ...")
X = np.loadtxt(os.path.join(".data", "X.txt"))
y = np.loadtxt(os.path.join(".data", "y.txt"))
X = X.astype("float32") / 255.0


# TODO:
# try some other digits.


# narrow it down to just 7's
X = X[y==8]
logging.info(f"Focusing only on 7's. X.shape = {X.shape}.")

# what layer does the encoding come from?
code_layer_n = 3

# encode these digits into 2 dimensions
autoencoder = MLPRegressor(
    hidden_layer_sizes=(256, 64, 10, 64, 256),
    activation="relu", solver="sgd",
    max_iter=100, alpha=0.00001, batch_size=10,
    verbose=True, random_state=42
)

# train the model
logging.info("Training the auto-encoder ...")
autoencoder.fit(X, X)

# plot the digits across these 2 dimensions.
logging.info("Using the trained auto-encoder to encode the original data ...")
X_encoded_10d = np.array([encode(autoencoder, x, code_layer_n) for x in X])
logging.info(f"After encoding, X_encoded_10d.shape = {X_encoded_10d.shape}.")

# use k-means to cluser these images into k-clusters.
wcss_list = []
cluster_range = range(2, 15)

# Compute K-means and calculate WCSS for different number of clusters
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, n_init=1, random_state=0)
    kmeans.fit(X_encoded_10d)
    wcss = kmeans.inertia_  # Inertia is the within-cluster sum of squares
    wcss_list.append(wcss)

# Plotting the Elbow graph
plt.close("all")
plt.plot(cluster_range, wcss_list, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("autoencoder-mnist-dim10d-elbow.png")

# generate images from the cluster centroids
# Perform K-means clustering on 2D data
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X_encoded_10d)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

for i in range(n_clusters):
    centroid = np.array(cluster_centers[i]).reshape(1, 10)
    logging.info(f"Cluster: {i}, Centroid: {centroid}")
    x0 = np.array(decode(autoencoder, centroid, code_layer_n))
    x0 = x0.reshape(28, 28) # reshape into a 28x28 matrix
    plt.close("all")
    plt.imshow(x0, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig(f"autoencoder-mnist-dim10d-kmeans-cluster-{int(i)}.png")
