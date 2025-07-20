import numpy as np

def generate_biased_clusters(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_features: int = 50,
    cluster_std: float = 0.5
) -> np.ndarray:
    """
    クラスタ中心をランダムに生成し、各クラスタからガウス分布で点群を生成する。
    テストや可視化のための合成データ。

    Generate synthetic multi-cluster data for testing or visualization.

    Parameters:
        n_clusters : int
            クラスタ数 / Number of clusters
        n_samples_per_cluster : int
            各クラスタに含まれるサンプル数 / Samples per cluster
        n_features : int
            特徴次元数 / Number of dimensions
        cluster_std : float
            クラスタの広がり / Standard deviation of clusters

    Returns:
        data : np.ndarray of shape (n_clusters * n_samples_per_cluster, n_features)
    """
    centers = np.random.randn(n_clusters, n_features) * 5  # ランダムに離れた中心
    data = []
    for c in centers:
        samples = np.random.randn(n_samples_per_cluster, n_features) * cluster_std + c
        data.append(samples)
    return np.vstack(data)
