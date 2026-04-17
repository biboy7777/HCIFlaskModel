import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import pickle

# === Load CSV ===
path = "HCI PURE NUMERICAL with header BUT IN WEKA 7 clusters no user column.csv"
headernames = ['object', 'age', 'view','rating', 'gender', 'glasses', 'handedness', 'cluster']

# Read CSV with header
dataset = pd.read_csv(path, names=headernames, header=0)

# Split features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

# === Split into Train, Validation, and Test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=1)

print("\nValidation Set has:", len(X_valid))
print("Training Set has:", len(X_train))
print("Test Set has:", len(X_test))

# === Train Gaussian Naive Bayes ===
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# === Predict on Test Set ===
y_pred = classifier.predict(X_test)
print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))

# === Predict on Validation Set ===
y_pred_validation = classifier.predict(X_valid)
print("Accuracy on Validation Set:", accuracy_score(y_valid, y_pred_validation))

# === Plot Confusion Matrix ===
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
cm_display.plot()
plt.show()

# === Save model ===
filename = 'ILANO_90VALIDATION_finalized_model_HCI_no_user_7clusters GAUSSIAN NB VIA VSCODE.pkl'
pickle.dump(classifier, open(filename, 'wb'))
print("\nModel saved as:", filename)
