import pandas as pd
import numpy as np

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = []
        for centroid in centroids:
            distances.append(euclidean_distance(point, centroid))
        clusters.append(distances.index(min(distances)))
    return clusters

def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = []
        for j in range(len(data)):
            if clusters[j] == i:
                cluster_points.append(data[j])
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(data[i])
    return new_centroids

def kmeans(data, k, max_iterations=100):
    centroids = data[:k]
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
numeric_df = df.select_dtypes(include=["int64", "float64"])
numeric_df = numeric_df.fillna(numeric_df.mean())
data = numeric_df.values.tolist()
k = int(input("Enter number of clusters: "))
clusters, centroids = kmeans(data, k)
print("\nCluster Assigned to Each Data Point")
print(clusters)
print("\nFinal Centroids")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}:")
    print(centroid)