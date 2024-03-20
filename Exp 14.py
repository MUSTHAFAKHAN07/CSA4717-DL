import numpy as np

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Modify data to introduce outliers
X_outliers = np.array([[0.1], [0.15], [0.2]])
y_outliers = np.array([[30], [35], [40]])
X = np.vstack((X, X_outliers))
y = np.vstack((y, y_outliers))

# Gradient Descent Function
def gradient_descent(X, y, iterations=1000, learning_rate=0.01, stopping_threshold=1e-5):
    # Initialize parameters
    theta = np.random.randn(2, 1)
    m = len(X)
    
    # Gradient Descent loop
    for i in range(iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
        # Calculate loss
        loss = np.linalg.norm(X.dot(theta) - y)**2 / m
        if loss < stopping_threshold:
            break
    return theta

# Estimation of optimal parameters
theta_optimal = gradient_descent(np.c_[np.ones((X.shape[0], 1)), X], y)
print("Optimal Parameters (theta):", theta_optimal)

