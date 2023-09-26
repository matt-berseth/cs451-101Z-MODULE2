import logging
import os
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

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
    n = 10000
    os.makedirs(".data", exist_ok=True)
    np.savetxt(os.path.join(".data", "y.txt"), mnist.target[:n], fmt="%s")
    np.savetxt(os.path.join(".data", "X.txt"), mnist.data[:n], fmt="%1.2f")

logging.info("loading mnist data ...")
X = np.loadtxt(os.path.join(".data", "X.txt"))
y = np.loadtxt(os.path.join(".data", "y.txt"))

X = X[y==7]
X = X.astype("float32") / 255.0

# Add noise to the data
noise_factor = 0.1
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Create the autoencoder model
# TODO: try lower dimensions
autoencoder = MLPRegressor(hidden_layer_sizes=(512, 128, 64, 128, 512),
                           activation="relu",
                           solver="adam",
                           max_iter=100,
                           alpha=0.00001,
                           verbose=True,
                           random_state=42)

# Train the model on noisy data
autoencoder.fit(X_noisy, X)

# Evaluate on test set
score = autoencoder.score(X_noisy, X)
print(f"Autoencoder Test Score: {score}")

# Predict the reconstructed images
decoded_imgs = autoencoder.predict(X_noisy)

# Visualize the reconstructed images
n = 10
plt.figure(figsize=(30, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + (2*n))
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
plt.savefig("autoencoder-mnist.png")

