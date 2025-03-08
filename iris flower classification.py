import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better handling
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Take a smaller, stratified subset for better generalization
small_df = df.groupby('target', group_keys=False).apply(lambda x: x.sample(n=15, random_state=42))

# Add slight noise to avoid overfitting
np.random.seed(42)
small_df.iloc[:, :-1] += np.random.normal(0, 0.1, small_df.iloc[:, :-1].shape)

# Separate features and target
X_small = small_df.drop(columns=['target'])
y_small = small_df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.4, random_state=42, stratify=y_small)

# Initialize and train a simplified Random Forest classifier
model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Perform cross-validation to validate generalization
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Generate a classification report
class_report = classification_report(y_test, y_pred, target_names=iris.target_names, digits=4)
print("Classification Report:\n", class_report)

# Generate and plot a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Iris Classification")
plt.show()

# Plot feature importance
feature_importances = model.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(iris.feature_names, feature_importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.title("Actual vs Predicted Labels")
plt.legend()
plt.show()