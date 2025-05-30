from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import wandb
import matplotlib.pyplot as plt


mnist  = fetch_openml('mnist_784')
data = mnist.data
target = mnist.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
clf_hinge = SGDClassifier(loss='hinge',  max_iter=1000)
# Train the model using the training data
clf_hinge.fit(X_train, y_train)
# Predict the labels for the test data
y_pred = clf_hinge.predict(X_test)
# Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

print('\033[1m' + "Higne Loss Classifier" + '\033[0m')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred, average='weighted'))
print("Precision Score:", precision_score(y_test, y_pred, average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, y_pred))