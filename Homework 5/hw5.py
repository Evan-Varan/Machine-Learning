import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
#load the dataset
iris = datasets.load_iris()
#print the shape of the dataset
print(iris.data.shape)
X = iris.data
y = iris.target
# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42) # Reduce to 2 dimensions
X_tsne = tsne.fit_transform(X)
# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('t-SNE Visualization of Iris Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()
plt.show()
plt.savefig("SVMtsne_iris_plot.png")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Standardize the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Apply PCA to the standardized training data
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
# Transform the test data using the fitted PCA model
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
# After applying PCA as described previously
num_features = X_train_pca.shape[1]
print(f"Number of features after PCA: {num_features}")
#print the shape of dataset after applying pca
print(X_train_pca.shape)
# Initialize the SVM model with RBF kernel
svm_model = SVC(kernel='rbf')
# Train the model
svm_model.fit(X_train_pca, y_train)
# Make predictions on the test set
y_pred_test = svm_model.predict(X_test_pca)
# Make predictions on the training set
y_pred_train = svm_model.predict(X_train_pca)
# Calculate and print the metrics
accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test, average='weighted')
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
test_accuracy = accuracy_score(y_test, y_pred_test)
training_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision Score: {precision}")
print(f"Recall Score: {recall}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Training Accuracy: {training_accuracy}")

#Name of the text file
with open("SVM_iris_accuracies.txt", "w") as f:
    f.write('Training Accuracy: %.2f' % accuracy_score(y_train,
    y_pred_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % accuracy_score(y_test,
    y_pred_test))
    f.write('\n')