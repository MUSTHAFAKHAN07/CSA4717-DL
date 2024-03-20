import numpy as np
from sklearn.metrics import confusion_matrix

# Actual labels
actual = np.array([1, 0, 1, 0, 1, 0, 1, 0])
# Predicted labels
predicted = np.array([1, 1, 0, 0, 1, 1, 1, 0])

# Calculate confusion matrix
cm = confusion_matrix(actual, predicted)
print("Confusion Matrix:")
print(cm)
