from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()

# Initialize logistic regression model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, data.data, data.target, cv=5)

# Print mean score
print("Mean Accuracy:", scores.mean())

