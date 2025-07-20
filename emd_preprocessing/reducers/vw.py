import numpy as np
import heapq
from ..base import BaseReducer


class VWReducer(BaseReducer):
    """
    Visvalingam-Whyatt（VW）法に基づく間引きアルゴリズム。
    高次元空間でも外積を使わず、1/2ab*sinθ 相当の面積で点の重要度を評価。

    Standard VW reduction using triangle importance based on ab*sinθ.
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def reduce(self, points: np.ndarray, show_progress: bool = False):
        points = np.asarray(points)
        kept_indices, removed_points, removed_indices = self._vw_reduce(points, show_progress)

        assigned = {idx: [points[idx]] for idx in kept_indices}
        for pt, idx in zip(removed_points, removed_indices):
            prevs = [i for i in kept_indices if i < idx]
            nexts = [i for i in kept_indices if i > idx]
            prev_idx = prevs[-1] if prevs else kept_indices[0]
            next_idx = nexts[0] if nexts else kept_indices[-1]

            nearest = prev_idx if np.linalg.norm(pt - points[prev_idx]) <= np.linalg.norm(pt - points[next_idx]) else next_idx
            assigned[nearest].append(pt)

        centers = np.array([np.mean(assigned[idx], axis=0) for idx in kept_indices])
        weights = np.array([len(assigned[idx]) for idx in kept_indices], dtype=np.float32)
        weights /= weights.sum()
        return centers, weights

    def _vw_reduce(self, points: np.ndarray, show_progress: bool):
        if len(points) < 3:
            return [0, len(points) - 1], np.empty((0, points.shape[1])), []

        max_area = 0
        best_idx = 0
        iterator = range(1, len(points) - 1)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="VW Reduction", unit="pt")

        for i in iterator:
            area = self._triangle_importance(points[0], points[-1], points[i])
            if area > max_area:
                max_area = area
                best_idx = i

        if max_area > self.epsilon:
            left_k, left_rp, left_ri = self._vw_reduce(points[:best_idx + 1], show_progress)
            right_k, right_rp, right_ri = self._vw_reduce(points[best_idx:], show_progress)

            removed_combined = np.vstack([arr for arr in [left_rp, right_rp] if arr.size > 0]) \
                if any(arr.size > 0 for arr in [left_rp, right_rp]) else np.empty((0, points.shape[1]))
            removed_indices = left_ri + [i + best_idx for i in right_ri]

            return (
                left_k[:-1] + [i + best_idx for i in right_k],
                removed_combined,
                removed_indices
            )
        else:
            return [0, len(points) - 1], points[1:-1], list(range(1, len(points) - 1))

    def _triangle_importance(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        vec1 = p2 - p1
        vec2 = p3 - p1
        angle = self._angle_between(vec1, vec2)
        return np.linalg.norm(vec1) * np.linalg.norm(vec2) * np.abs(np.sin(angle))

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_theta)


class VWRawReducer(BaseReducer):
    """
    間引き後の点をそのまま代表点として使用する軽量版VW。

    VW reduction returning raw kept points as support, with uniform weights.
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def reduce(self, points: np.ndarray, show_progress: bool = False):
        points = np.asarray(points)
        kept_indices, _, _ = VWReducer(self.epsilon)._vw_reduce(points, show_progress)
        supports = points[kept_indices]
        weights = np.ones(len(supports), dtype=np.float32)
        weights /= weights.sum()
        return supports, weights


class ReverseVWReducer(BaseReducer):
    """
    指定数に達するまで最も重要な点を順に追加する「逆VW法」。

    Reverse VW algorithm:
    - Starts with endpoints
    - Iteratively inserts the most important point (max triangle area)
    - Continues until desired number of support points is reached
    """

    def __init__(self, n_support_points: int):
        self.n = n_support_points

    def reduce(self, features: np.ndarray):
        features = np.asarray(features)
        n = len(features)
        if self.n >= n:
            return features.copy(), np.ones(n, dtype=np.float32) / n

        kept_indices = [0, n - 1]
        candidates = []
        self._push_most_important(features, 0, n - 1, candidates)

        while len(kept_indices) < self.n and candidates:
            _, idx, a, b = heapq.heappop(candidates)
            if idx in kept_indices:
                continue
            kept_indices.append(idx)
            kept_indices.sort()
            i = kept_indices.index(idx)
            if idx - kept_indices[i - 1] > 1:
                self._push_most_important(features, kept_indices[i - 1], idx, candidates)
            if kept_indices[i + 1] - idx > 1:
                self._push_most_important(features, idx, kept_indices[i + 1], candidates)

        # 再割り当てと重心・重みの計算
        assigned = {i: [] for i in kept_indices}
        for i in range(n):
            if i in kept_indices:
                assigned[i].append(features[i])
            else:
                nearest = min(kept_indices, key=lambda k: np.linalg.norm(features[i] - features[k]))
                assigned[nearest].append(features[i])

        support_points = np.array([np.mean(assigned[i], axis=0) for i in kept_indices])
        masses = np.array([len(assigned[i]) for i in kept_indices], dtype=np.float32)
        masses /= np.sum(masses)
        return support_points, masses

    def _push_most_important(self, points, a, b, heap):
        if b - a < 2:
            return
        max_score = -1
        idx = -1
        for i in range(a + 1, b):
            score = self._triangle_area(points[a], points[b], points[i])
            if score > max_score:
                max_score = score
                idx = i
        if idx != -1:
            heapq.heappush(heap, (-max_score, idx, a, b))

    def _triangle_area(self, p1, p2, p3):
        v1, v2 = p2 - p1, p3 - p1
        cross = np.cross(v1, v2) if p1.shape[0] == 3 else np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(self._angle(v1, v2))
        return 0.5 * np.linalg.norm(cross) if isinstance(cross, np.ndarray) else 0.5 * cross

    def _angle(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_theta)

