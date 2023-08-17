import numpy as np
import matplotlib.pyplot as plt

# Simulate data matrix X (n x k)
n = int(input("Enter number of rows for data matrix: "))  # Number of rows
k = int(input("Enter number of columns for data matrix: "))   # Number of columns


np.random.seed(0)
X = np.random.rand(n, k)  # Simulated data matrix

# (i) Compute the covariance matrix for X
covariance_matrix = np.cov(X, rowvar=False)  # Rowvar is set to False as each row is a variable

# (ii) Compute top 3 principal components
_, _, Vt = np.linalg.svd(X, full_matrices=False)
top_components = Vt[:3, :]  # Top 3 principal components

# Calculate the amount of variance explained by the top 3 components
total_variance = np.sum(np.var(X, axis=0))
explained_variance = np.sum(np.var(np.dot(X, top_components.T), axis=0))
variance_ratio = explained_variance / total_variance

print("Covariance Matrix:")
print(covariance_matrix)

print("\nAmount of Variance Explained by Top 3 Components:", variance_ratio)

print("\nTop 3 Principal Components:")
print(top_components)

# (iii) Report top 3 principal component loading vectors
loading_vectors = top_components.T
print("\nTop 3 Principal Component Loading Vectors:")
print(loading_vectors)

# Plot the top 3 principal components
plt.figure(figsize=(10, 6))
plt.title("Top 3 Principal Components")
plt.xlabel("Feature Index")
plt.ylabel("Principal Component Value")
for i in range(3):
    plt.plot(top_components[i], label=f"PC{i+1}")
plt.legend()
plt.grid(True)
plt.show()