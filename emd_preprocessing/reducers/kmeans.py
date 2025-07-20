import numpy as np
from ..base import BaseReducer

class KMeansReducer(BaseReducer):
    """
    KMeans++ によるクラスタリングを用いて、
    特徴ベクトルを代表点と質量に要約するリデューサ。

    Reducer using KMeans++ clustering to reduce feature vectors
    into support points and their associated masses (weights).
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4):
        """
        Parameters:
        ----------
        n_clusters : int
            生成する代表点（クラスタ）数 / Number of output clusters.
        max_iter : int
            最大反復回数 / Maximum number of Lloyd's iterations.
        tol : float
            収束判定用の許容誤差 / Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def reduce(self, features: np.ndarray):
        """
        特徴ベクトル群を KMeans++ によってクラスタリングし、
        代表点と質量に変換する。

        Reduce input features using KMeans++ and compute
        support points and normalized cluster masses.

        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)

        Returns:
        -------
        support_points : np.ndarray of shape (n_clusters, n_features)
            各クラスタの重心 / Cluster centers.
        masses : np.ndarray of shape (n_clusters,)
            各クラスタの重み（正規化済み）/ Normalized cluster weights.
        """
        features = np.array(features)
        n_samples, _ = features.shape

        # ----- KMeans++ 初期化 / Initialization -----
        centers = [features[np.random.choice(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min([np.linalg.norm(features - c, axis=1) for c in centers], axis=0)
            
            # ゼロ距離に対処するために、非常に小さな値を追加
            dists = np.where(dists == 0, 1e-10, dists)
            
            # 確率分布を計算
            sum_dists = np.sum(dists)
            if sum_dists == 0:
                prob = np.full_like(dists, 1 / len(dists))
            else:
                prob = dists / sum_dists
            
            next_center = features[np.random.choice(n_samples, p=prob)]
            centers.append(next_center)
        centers = np.array(centers)

        # ----- Lloyd's algorithm -----
        for _ in range(self.max_iter):
            # 各ベクトルを最近接クラスタに割り当てる
            dists = np.linalg.norm(features[:, None] - centers[None, :], axis=2)  # (n_samples, n_clusters)
            labels = np.argmin(dists, axis=1)

            # 各クラスタの新しい重心を計算
            new_centers = np.array([
                features[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                for i in range(self.n_clusters)
            ])

            # 収束チェック
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers

        # ----- クラスタごとの重みを計算 -----
        masses = np.zeros(self.n_clusters) 
        for i in range(self.n_clusters):
            masses[i] = np.sum(labels == i) / n_samples
        
        masses /= np.sum(masses)
        return centers, masses

class KMeansInitReducer(BaseReducer):
    """
    KMeans++ の初期化のみを使ってクラスタ代表点と重みを構成する軽量リデューサ。

    Lightweight reducer using only KMeans++ initialization to extract
    support points and compute cluster weights.
    """

    def __init__(self, n_clusters: int):
        """
        Parameters:
        ----------
        n_clusters : int
            クラスタ（代表点）数 / Number of output support points
        """
        self.n_clusters = n_clusters

    def reduce(self, features: np.ndarray):
        """
        特徴ベクトル群を KMeans++ 初期化によりクラスタ分割し、
        各クラスタの重心と重みを計算する。

        Perform only KMeans++ initialization, assign points to nearest centers,
        and compute cluster centers and normalized weights.

        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)

        Returns:
        -------
        support_points : np.ndarray of shape (n_clusters, n_features)
            各クラスタの重心 / Cluster centers

        masses : np.ndarray of shape (n_clusters,)
            各クラスタの質量（正規化）/ Normalized weights
        """
        features = np.array(features)
        n_samples = len(features)

        # --- KMeans++ 初期化 ---
        centers = [features[np.random.choice(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min([np.linalg.norm(features - c, axis=1) for c in centers], axis=0)
            dists = np.where(dists == 0, 1e-10, dists)
            sum_dists = np.sum(dists)
            if sum_dists == 0:
                prob = np.full_like(dists, 1 / len(dists))
            else:
                prob = dists / sum_dists
            next_center = features[np.random.choice(n_samples, p=prob)]
            centers.append(next_center)
        centers = np.array(centers)

        # --- 一度だけ割り当て ---
        dists = np.linalg.norm(features[:, None] - centers[None, :], axis=2)  # (n_samples, n_clusters)
        labels = np.argmin(dists, axis=1)

        # --- 重心と質量を計算 ---
        support_points = np.array([
            features[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(self.n_clusters)
        ])
        masses = np.array([np.sum(labels == i) for i in range(self.n_clusters)], dtype=np.float32)
        masses /= np.sum(masses)

        return support_points, masses
