Sure! Here is an example of K-means clustering code using Python along with explanations for each step.

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic dataset
n_samples = 1500
n_features = 2
n_clusters = 3
X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=0.60, random_state=0)

# Plot the synthetic dataset
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Initialize and fit K-means model
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

### Explanation

1. **Import necessary libraries**:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from sklearn.cluster import KMeans
   ```
   - `numpy`: For numerical operations.
   - `matplotlib.pyplot`: For plotting.
   - `sklearn.datasets.make_blobs`: For generating a synthetic dataset.
   - `sklearn.cluster.KMeans`: For K-means clustering.

2. **Generate synthetic dataset**:
   ```python
   n_samples = 1500
   n_features = 2
   n_clusters = 3
   X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=0.60, random_state=0)
   ```
   - `n_samples`: Number of data points.
   - `n_features`: Number of features.
   - `n_clusters`: Number of clusters.
   - `make_blobs`: Generates a synthetic dataset.

3. **Plot the synthetic dataset**:
   ```python
   plt.scatter(X[:, 0], X[:, 1], s=50)
   plt.title("Generated Data")
   plt.xlabel("Feature 1")
   plt.ylabel("Feature 2")
   plt.show()
   ```
   - This code plots the generated data points.

4. **Initialize and fit K-means model**:
   ```python
   kmeans = KMeans(n_clusters=n_clusters)
   kmeans.fit(X)
   ```
   - `KMeans`: Initializes the K-means model.
   - `fit`: Fits the model to the data.

5. **Get the cluster centers and labels**:
   ```python
   centers = kmeans.cluster_centers_
   labels = kmeans.labels_
   ```
   - `cluster_centers_`: Coordinates of the cluster centers.
   - `labels_`: Labels of each point.

6. **Plot the clustered data**:
   ```python
   plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
   plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
   plt.title("K-means Clustering")
   plt.xlabel("Feature 1")
   plt.ylabel("Feature 2")
   plt.show()
   ```
   - This code plots the clustered data points and the cluster centers.

Feel free to ask if you have any questions or need further explanation!
