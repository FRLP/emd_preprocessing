import numpy as np
import heapq
from ..base import BaseReducer

class RDPReducer(BaseReducer):
    """
    Douglas–Peucker アルゴリズムを使って重要な点を残し、
    他の点は代表点に重みとして集約するリデューサ。

    Reduce points using Douglas–Peucker and assign removed points
    to nearby remaining ones to form weighted support points.
    """

    def __init__(self, epsilon: float = 1.0):
        """
        Parameters:
        ----------
        epsilon : float
            削除許容誤差 / Approximation tolerance
        """
        self.epsilon = epsilon

    def reduce(self, features: np.ndarray):
        features = np.array(features)
        _, kept_indices, removed_points, removed_indices = self._rdp_recursive(features, 0)

        # --- 割当辞書 / Assignment dict ---
        assigned = {idx: [features[idx]] for idx in kept_indices}
        for point, idx in zip(removed_points, removed_indices):
            prev_candidates = [i for i in kept_indices if i < idx]
            next_candidates = [i for i in kept_indices if i > idx]
            prev_idx = prev_candidates[-1] if prev_candidates else kept_indices[0]
            next_idx = next_candidates[0] if next_candidates else kept_indices[-1]
            
            dist_prev = np.linalg.norm(point - features[prev_idx])
            dist_next = np.linalg.norm(point - features[next_idx])
            target = prev_idx if dist_prev <= dist_next else next_idx
            assigned[target].append(point)

        support_points = np.array([np.mean(assigned[i], axis=0) for i in kept_indices])
        masses = np.array([len(assigned[i]) for i in kept_indices], dtype=np.float32)
        masses /= masses.sum()
        return support_points, masses

    def _rdp_recursive(self, pts, start_idx):
        if len(pts) < 3:
            return pts, [start_idx, start_idx + len(pts) - 1], np.empty((0, pts.shape[1])), []

        dmax, index = 0, 0
        for i in range(1, len(pts) - 1):
            d = self._perpendicular_distance(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax, index = d, i

        if dmax > self.epsilon:
            l, li, lr, lri = self._rdp_recursive(pts[:index+1], start_idx)
            r, ri, rr, rri = self._rdp_recursive(pts[index:], start_idx + index)
            
            combined_removed = (
                np.vstack([arr for arr in [lr, rr] if arr.size > 0])
                if any(arr.size > 0 for arr in [lr, rr])
                else np.empty((0, pts.shape[1]))
            )
            combined_removed_idx = lri + rri
            
            return (
                np.vstack((l[:-1], r)),
                li[:-1] + ri,
                combined_removed,
                combined_removed_idx
            )
        else:
            return np.array([pts[0], pts[-1]]), [start_idx, start_idx + len(pts) - 1], pts[1:-1], list(range(start_idx + 1, start_idx + len(pts) - 1))

    def _perpendicular_distance(self, point, start, end):
        if np.array_equal(start, end):
            return np.linalg.norm(point - start)
        line = end - start
        proj = np.dot(point - start, line) / np.dot(line, line) * line
        return np.linalg.norm((point - start) - proj)


class RDPRawReducer(BaseReducer):
    """
    Douglas–Peucker による代表点のみ抽出（重みは均等）。

    Simplified Douglas–Peucker reducer returning only support points
    with uniform weights.
    """
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def reduce(self, features: np.ndarray):
        features = np.array(features)
        _, kept_indices, _, _ = self._rdp_recursive(features, 0)
        support_points = features[np.array(kept_indices)]
        masses = np.ones(len(support_points), dtype=np.float32)
        masses /= np.sum(masses)
        return support_points, masses

    def _rdp_recursive(self, pts, start_idx):
        if len(pts) < 3:
            return pts, [start_idx, start_idx + len(pts) - 1], np.empty((0, pts.shape[1])), []

        dmax, index = 0, 0
        for i in range(1, len(pts) - 1):
            d = self._perpendicular_distance(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax, index = d, i

        if dmax > self.epsilon:
            l, li, lr, lri = self._rdp_recursive(pts[:index+1], start_idx)
            r, ri, rr, rri = self._rdp_recursive(pts[index:], start_idx + index)
            
            combined_removed = (
                np.vstack([arr for arr in [lr, rr] if arr.size > 0])
                if any(arr.size > 0 for arr in [lr, rr])
                else np.empty((0, pts.shape[1]))
            )
            combined_removed_idx = lri + rri
            
            return (
                np.vstack((l[:-1], r)),
                li[:-1] + ri,
                combined_removed,
                combined_removed_idx
            )
        else:
            return np.array([pts[0], pts[-1]]), [start_idx, start_idx + len(pts) - 1], pts[1:-1], list(range(start_idx + 1, start_idx + len(pts) - 1))

    def _perpendicular_distance(self, point, start, end):
        if np.array_equal(start, end):
            return np.linalg.norm(point - start)
        line = end - start
        proj = np.dot(point - start, line) / np.dot(line, line) * line
        return np.linalg.norm((point - start) - proj)


class ReverseRDPReducer(BaseReducer):
    """
    必要な代表点数に達するまで、遠い点を優先的に保持する「逆RDP」。

    Recursively adds points with highest perpendicular error
    until the desired number of support points is reached.
    """

    def __init__(self, n_support_points: int):
        self.n = n_support_points

    def reduce(self, features: np.ndarray):
        features = np.array(features)
        n = len(features)
        if self.n >= n:
            return features.copy(), np.ones(n, dtype=np.float32) / n

        kept_indices = [0, n - 1]
        candidates = []
        self._push_farthest(features, 0, n - 1, candidates)

        while len(kept_indices) < self.n and candidates:
            _, idx, a, b = heapq.heappop(candidates)
            if idx in kept_indices:
                continue
            kept_indices.append(idx)
            kept_indices.sort()
            i = kept_indices.index(idx)
            if idx - kept_indices[i - 1] > 1:
                self._push_farthest(features, kept_indices[i - 1], idx, candidates)
            if kept_indices[i + 1] - idx > 1:
                self._push_farthest(features, idx, kept_indices[i + 1], candidates)

        # --- 各点を最寄り代表点に割当て / Assign to nearest ---
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

    def _push_farthest(self, points, a, b, heap):
        if b - a < 2:
            return
        max_dist, idx = -1, -1
        for i in range(a + 1, b):
            d = self._perpendicular_distance(points[i], points[a], points[b])
            if d > max_dist:
                max_dist, idx = d, i
        if idx != -1:
            heapq.heappush(heap, (-max_dist, idx, a, b))

    def _perpendicular_distance(self, point, start, end):
        if np.array_equal(start, end):
            return np.linalg.norm(point - start)
        line = end - start
        proj = np.dot(point - start, line) / np.dot(line, line) * line
        return np.linalg.norm((point - start) - proj)