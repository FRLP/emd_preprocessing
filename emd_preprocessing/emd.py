import numpy as np
import cvxpy as cp

def calculate_emd(vectors1: np.ndarray, vectors2: np.ndarray,
                  weights1: np.ndarray, weights2: np.ndarray) -> float | None:
    """
    Earth Mover's Distance (EMD) を最小輸送コストとして計算する関数。
    もしソルバーが失敗したら None を返す。

    Parameters
    ----------
    vectors1 : np.ndarray of shape (n1, d)
        第1分布のサポートポイント（特徴ベクトル）
    vectors2 : np.ndarray of shape (n2, d)
        第2分布のサポートポイント（特徴ベクトル）
    weights1 : np.ndarray of shape (n1,)
        第1分布の質量（合計1に正規化）
    weights2 : np.ndarray of shape (n2,)
        第2分布の質量（合計1に正規化）

    Returns
    -------
    float | None
        最小輸送コスト（EMD）. 解けなかった場合は None。
    """
    n, m = len(vectors1), len(vectors2)

    # --- 距離行列 D(i,j) を計算 ---
    dist_matrix = np.linalg.norm(vectors1[:, None] - vectors2[None, :], axis=2)  # shape (n, m)

    # --- 輸送量変数 T(i,j) ---
    flow = cp.Variable((n, m), nonneg=True)

    # --- 制約条件 ---
    constraints = [
        cp.sum(flow, axis=1) == weights1,
        cp.sum(flow, axis=0) == weights2
    ]

    # --- 最小化問題の定義 ---
    objective = cp.Minimize(cp.sum(cp.multiply(dist_matrix, flow)))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as e:
        # ソルバー実行時にエラーが発生した場合は None
        print(f"[WARNING] Solver failed with exception: {e}")
        return None

    # 最適化に失敗した場合（infeasible/unbounded など）
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[WARNING] Solver status: {problem.status}")
        return None

    return problem.value
