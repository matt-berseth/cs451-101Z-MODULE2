import logging
import os
import warnings

import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from autoencode import decode

# Fetch MNIST data
# Load MNIST data from OpenML
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
np.random.seed(0)

# try some other digits.
digit = 5
output_dirpath = os.path.join(".model", f"{digit}")

# train the model
logging.info("Loading the trained auto-encoder ...")
autoencoder = joblib.load(os.path.join(output_dirpath, "model.pkl"))

# plot some extreme points:
code = np.array([4., 4.]).reshape(1, 2)
x0 = np.array(decode(autoencoder, code, 3))
x0 = x0.reshape(28, 28) # reshape into a 28x28 matrix
plt.close("all")
plt.imshow(x0, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.savefig(os.path.join(output_dirpath, "inference.png"))