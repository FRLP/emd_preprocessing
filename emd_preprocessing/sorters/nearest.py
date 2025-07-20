import numpy as np
from scipy.spatial import cKDTree
import hnswlib
from ..base import BaseSorter

class KDTreeGreedySorter(BaseSorter):
    """
    KDTree を用いて、最近傍を1点ずつ辿ることで順序を構築するソーター。

    Sorter that greedily walks through nearest neighbors using KDTree.
    """

    def __init__(self, features: np.ndarray):
        """
        Initialize the sorter with feature vectors and build KDTree.
        
        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)
            Feature vectors to be sorted.
        """
        super().__init__(features)
        # KDTreeをコンストラクタで構築
        self.tree = cKDTree(self.features)

    def sort(self) -> np.ndarray:
        """
        Sort the vectors based on nearest neighbor traversal.
        
        Returns:
        -------
        sorted_features : np.ndarray of shape (n_samples, n_features)
            Feature vectors sorted by nearest neighbor traversal.
        """
        n = len(self.features)
        visited = [False] * n
        order = []
        
        # Start from the first vector
        current_index = 0
        order.append(current_index)
        visited[current_index] = True

        for _ in range(n - 1):
            # Query the nearest unvisited neighbor with parallel processing
            _, nearest_indices = self.tree.query(
                self.features[current_index], k=n, workers=-1
            )
            
            # Find the first unvisited neighbor
            for index in nearest_indices:
                if not visited[index]:
                    order.append(index)
                    visited[index] = True
                    current_index = index
                    break

        # Return the vectors in the nearest neighbor order
        return self.features[order]

class HNSWGreedySorter(BaseSorter):
    """
    HNSW インデックスを用いて、Greedyに最近傍を1点ずつ辿ることで
    特徴ベクトル群の順序を構築するソーター。

    Greedy sorter using HNSW index for nearest neighbor traversal.
    """

    def __init__(self, features: np.ndarray, ef_construction: int = 100, M: int = 16):
        """
        Initialize the sorter with feature vectors and build HNSW index.

        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)
            Feature vectors to be sorted.

        ef_construction : int
            Construction quality parameter. Higher = better recall, slower indexing.

        M : int
            Maximum number of neighbors for each node in the HNSW graph.
        """
        super().__init__(features)
        self.n, self.dim = self.features.shape

        # HNSWインデックスの作成
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(max_elements=self.n, ef_construction=ef_construction, M=M)
        self.index.add_items(self.features, np.arange(self.n))
        self.index.set_ef(50)  # 検索時の探索範囲

    def sort(self) -> np.ndarray:
        """
        Greedyに最も近い未訪問の点を辿ることで順序を構築する。

        Returns:
        -------
        sorted_features : np.ndarray of shape (n_samples, n_features)
            Feature vectors sorted by nearest neighbor traversal using HNSW.
        """
        visited = np.zeros(self.n, dtype=bool)
        order = []

        # 最初の点からスタート（ID=0）
        current_id = 0
        order.append(current_id)
        visited[current_id] = True

        for _ in range(self.n - 1):
            # 現在の点から最近傍候補を多数取得（高確率で未訪問点を含む）
            neighbor_ids, _ = self.index.knn_query(self.features[current_id], k=20)

            # 最も近くて未訪問の点を選ぶ
            for neighbor_id in neighbor_ids[0]:
                if not visited[neighbor_id]:
                    order.append(neighbor_id)
                    visited[neighbor_id] = True
                    current_id = neighbor_id
                    break
            else:
                # fallback: すべて訪問済みのときは未訪問のどこかへジャンプ
                unvisited_indices = np.where(~visited)[0]
                current_id = int(unvisited_indices[0])
                order.append(current_id)
                visited[current_id] = True

        # インデックス順に並べて返す
        return self.features[np.array(order, dtype=int)]
