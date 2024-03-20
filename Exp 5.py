import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Calculate coefficients
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Plot regression line
def plot_regression_line(X, y, theta):
    plt.plot(X, y, "b.")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([0, 2, 0, 15])
    plt.plot(X, X_b.dot(theta), "r-")
    plt.show()

# Plot regression line
plot_regression_line(X, y, theta_best)

