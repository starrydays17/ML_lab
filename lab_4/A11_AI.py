# =============================================================================
# Lab 04 - A1 (portion for Lab03-A11: K-means from scratch)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A11.py's from-scratch k-means implementation
#   using numpy broadcasting to vectorize both the cluster-assignment step
#   (distance-matrix computation) and the centroid-update step, instead of
#   nested Python for-loops. Keep the same algorithm structure from
#   Pang-Ning Tan et al., Section 8.2 (Algorithm 8.1)."
# Human review: verified convergence criterion (centroid stability) and
#   fixed empty-cluster handling (reinitializes to a random data point,
#   same fallback idea as the Lab03 version but avoids the index-out-of-
#   range bug when k > i for the i-th empty cluster).
# =============================================================================
import numpy as np
import pandas as pd


def assign_clusters(data, centroids):
    """
    Assign every point in `data` to its nearest centroid.

    Vectorized: computes the full (n_points x k_centroids) distance
    matrix in one shot via numpy broadcasting instead of a double loop.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_points, n_features)
    centroids : numpy.ndarray, shape (k, n_features)

    Returns
    -------
    numpy.ndarray, shape (n_points,) : cluster index of each point
    """
    # (n_points, 1, n_features) - (1, k, n_features) -> (n_points, k, n_features)
    diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_points, k)
    return np.argmin(distances, axis=1)


def update_centroids(data, clusters, k, rng):
    """
    Recompute each cluster's centroid as the mean of its assigned points.

    If a cluster ends up with no points assigned, it is re-seeded to a
    random data point (keeps the algorithm well-defined instead of
    crashing or silently freezing an empty cluster).

    Parameters
    ----------
    data : numpy.ndarray, shape (n_points, n_features)
    clusters : numpy.ndarray, shape (n_points,)
    k : int
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (k, n_features)
    """
    n_features = data.shape[1]
    new_centroids = np.empty((k, n_features))
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids[cluster_id] = cluster_points.mean(axis=0)
        else:
            new_centroids[cluster_id] = data[rng.integers(0, len(data))]
    return new_centroids


def kmeans(data, k, max_iterations=100, random_state=42):
    """
    Basic K-means clustering (Tan, Steinbach & Kumar, Section 8.2,
    Algorithm 8.1): pick k initial centroids, alternate between assigning
    points to their nearest centroid and recomputing centroids, until the
    centroids stop changing (or max_iterations is reached).

    Parameters
    ----------
    data : array-like, shape (n_points, n_features)
    k : int
    max_iterations : int
    random_state : int, for reproducible initial centroid selection

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) : (cluster assignment per point, final centroids)
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(random_state)

    centroids = data[:k].copy()
    clusters = assign_clusters(data, centroids)

    for _ in range(max_iterations):
        new_centroids = update_centroids(data, clusters, k, rng)
        new_clusters = assign_clusters(data, new_centroids)
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            clusters = new_clusters
            break
        centroids, clusters = new_centroids, new_clusters

    return clusters, centroids


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.fillna(numeric_df.mean())
    data = numeric_df.to_numpy()

    k = int(input("Enter number of clusters: "))
    clusters, centroids = kmeans(data, k)

    print("\nCluster Assigned to Each Data Point")
    print(clusters.tolist())

    print("\nFinal Centroids")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i+1}:")
        print(centroid)
