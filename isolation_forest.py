import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# generate the data
n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

# stack it together
X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
)


# plot the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor="k")
handles, labels = scatter.legend_elements()
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.title("Gaussian inliers with \nuniformly distributed outliers")
plt.savefig("isolation-forest-data.png")


# train the model - notice no labels are used!
clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)


# show boundary on the test data
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_test,
    response_method="predict",
    alpha=0.5,
)
disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor="k")
disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.savefig("isolation-forest-prediction.png")


n_correct = np.sum(clf.predict(X_test) == y_test)
print(f"Accuracy: {n_correct} out of {X_test.shape[0]} -> {n_correct / X_test.shape[0]}")