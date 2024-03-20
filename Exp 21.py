import matplotlib.pyplot as plt

# Define two linearly separable clusters
X1 = [[1, 2], [2, 3], [3, 4], [4, 5]]
X2 = [[8, 7], [7, 6], [6, 5], [5, 4]]

# Plot the clusters
plt.scatter(*zip(*X1), color='blue', label='Cluster 1')
plt.scatter(*zip(*X2), color='red', label='Cluster 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly Separable Clusters')
plt.legend()
plt.show()
