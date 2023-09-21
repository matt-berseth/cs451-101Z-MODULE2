import logging
import os
import warnings

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
logging.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

logging.info("Print out the count of each label")
for i in range(0, 10):
    count = np.sum(y==i)
    logging.info(f"{i}: {count} / {round(count/len(y)*100, 4)}%")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with PCA and Random Forest Classifier
# try different n_components
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),  # Reduce dimensions
    ("mlp", MLPClassifier(hidden_layer_sizes=(5), learning_rate_init=.001, random_state=0))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Test the pipeline on the test data
y_pred = pipeline.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
