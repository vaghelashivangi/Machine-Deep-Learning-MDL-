//MUTLILAYER PERCEPTRON
import numpy as np
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier

# Load Iris dataset
X, y = datasets.load_iris(return_X_y=True)
print(X)
print(y)

# Train-test split (using the same logic as in your original code)
X_train = X[range(0, 150, 2), :]   
y_train = y[range(0, 150, 2)]

X_test = X[range(1, 150, 2), :]
y_test = y[range(1, 150, 2)]

# Create an instance of MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(65), max_iter=200, solver='sgd')

# Train the MLP classifier
clf.fit(X_train, y_train)

# Make predictions
prediction = clf.predict(X_test)

# Display predictions and evaluation metrics
print("############### Predictions #################")
print(prediction)
print("#############################################")

print("Accuracy:", metrics.accuracy_score(y_test, prediction, normalize=True))

print(metrics.classification_report(y_test,prediction))
print(metrics.confusion_matrix(y_test,prediction))
