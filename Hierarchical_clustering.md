this is the complete code for both agglomerative and divisive hierarchical clustering, explaining each part in detail.

### Full Code with Detailed Explanation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

# Generate sample data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)
```
- **Imports and Data Generation:** 
  - `numpy` and `matplotlib.pyplot` for numerical operations and plotting.
  - `make_blobs` to generate synthetic data with 3 centers.
  - `linkage` and `dendrogram` from `scipy.cluster.hierarchy` for hierarchical clustering.
  - `AgglomerativeClustering` and `KMeans` from `sklearn` for clustering algorithms.

```python
# Compute the linkage matrix for Agglomerative Clustering
Z = linkage(X, method='ward')
```
- **Linkage Matrix:** 
  - `linkage` function computes the hierarchical clustering. `method='ward'` minimizes the variance of merged clusters.

```python
# Plot the dendrogram for visualization
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Agglomerative Clustering")
plt.xlabel("Sample index")
plt.ylabel("Distance")
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()
```
- **Dendrogram Plot:**
  - Creates a figure with a specified size.
  - Sets the title, x-label, and y-label.
  - `dendrogram` function visualizes the tree structure of clusters.
  - `leaf_rotation=90.` and `leaf_font_size=8.` adjust the appearance of leaf labels.

```python
# Apply Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
```
- **Agglomerative Clustering:**
  - Creates an instance of `AgglomerativeClustering` with 3 clusters.
  - `affinity='euclidean'` specifies the distance metric.
  - `linkage='ward'` uses the Ward variance minimization method.
  - `fit_predict` fits the model and predicts cluster labels.

```python
# Visualize the clusters obtained from Agglomerative Clustering
plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='rainbow')
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
- **Cluster Visualization:**
  - `scatter` plots the data points with colors corresponding to cluster labels.
  - `c=y_hc` colors the points based on the cluster labels.
  - Sets the title, x-label, and y-label for the plot.

### Divisive Clustering Implementation

```python
def divisive_clustering(X, n_clusters):
    clusters = [(X, np.arange(X.shape[0]))]
    while len(clusters) < n_clusters:
        new_clusters = []
        for cluster_data, cluster_indices in clusters:
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(cluster_data)
            labels = kmeans.labels_
            new_clusters.append((cluster_data[labels == 0], cluster_indices[labels == 0]))
            new_clusters.append((cluster_data[labels == 1], cluster_indices[labels == 1]))
        clusters = new_clusters
    return np.concatenate([indices for _, indices in clusters])
```
- **Divisive Clustering Function:**
  - `divisive_clustering` function recursively splits the data into clusters.
  - **Initialization:** Starts with the entire dataset.
  - **Loop:** Continues splitting until the desired number of clusters is reached.
  - **KMeans Clustering:** For each cluster, `KMeans` with 2 clusters is applied.
  - **Collecting Results:** Appends new clusters to the list.

```python
# Generate sample data (reused)
X, y = make_blobs(n_samples=100, centers=3, random_state=42)
```
- **Data Generation:** Generates the same sample data for consistency.

```python
# Apply Divisive Clustering
y_dc = np.zeros(X.shape[0])
cluster_indices = divisive_clustering(X, 3)
for cluster_id, indices in enumerate(cluster_indices.reshape(-1, X.shape[0] // 3)):
    y_dc[indices] = cluster_id
```
- **Divisive Clustering Execution:**
  - `y_dc` initializes an array to store cluster labels.
  - `divisive_clustering(X, 3)` computes the cluster indices.
  - Reshapes `cluster_indices` and assigns cluster IDs to each point in `y_dc`.

```python
# Visualize the clusters obtained from Divisive Clustering
plt.scatter(X[:, 0], X[:, 1], c=y_dc, cmap='rainbow')
plt.title("Divisive Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
- **Cluster Visualization:**
  - Plots the data points with colors based on divisive clustering labels.
  - Sets the plot title, x-label, and y-label.

### Summary

- **Agglomerative Clustering** is implemented using `scikit-learn`'s `AgglomerativeClustering`.
- **Divisive Clustering** is implemented using recursive KMeans splitting.
- Both methods are visualized with scatter plots and dendrograms for comparison.
