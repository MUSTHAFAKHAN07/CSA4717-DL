import numpy as np

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

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
theta_optimal = gradient_descent(np.c_[np.ones((100, 1)), X], y)
print("Optimal Parameters (theta):", theta_optimal)

