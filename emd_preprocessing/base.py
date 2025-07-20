import numpy as np
from abc import ABC, abstractmethod

# ================================
# BaseReducer
# ================================
class BaseReducer(ABC):
    """
    特徴ベクトルを「代表点（support points）＋質量（mass）」に要約する抽象基底クラス。

    Abstract base class for reducing a set of feature vectors
    into support points and their associated masses (weights).
    """

    @abstractmethod
    def reduce(self, features: np.ndarray):
        """
        特徴ベクトル群を代表点と質量に変換する。

        Reduce the input feature vectors into support points and masses.

        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)
            入力ベクトル群 / Input feature vectors.

        Returns:
        -------
        support_points : np.ndarray of shape (n_support, n_features)
            要約された代表ベクトル / Extracted support points.

        masses : np.ndarray of shape (n_support,)
            各代表点の質量（正規化された重み）/ Normalized weights (sum to 1).
        """
        pass


# ================================
# BaseSorter
# ================================
class BaseSorter(ABC):
    """
    特徴ベクトル群を意味のある順序に並べ替えるための抽象基底クラス。

    Abstract base class for sorting feature vectors into a meaningful order.
    """

    def __init__(self, features: np.ndarray):
        """
        特徴ベクトルを受け取って保持する。

        Store input feature vectors.

        Parameters:
        ----------
        features : np.ndarray of shape (n_samples, n_features)
            並べ替え対象の特徴ベクトル / Feature vectors to be sorted.
        """
        self.features = np.array(features)

    @abstractmethod
    def sort(self) -> np.ndarray:
        """
        特徴ベクトルを並び替えた順で返す。

        Return feature vectors in sorted order.

        Returns:
        -------
        sorted_features : np.ndarray of shape (n_samples, n_features)
            並び替え後のベクトル群 / Reordered feature vectors.
        """
        pass
