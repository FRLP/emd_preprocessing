import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CIE標準白色点（D65）
Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

def f_inv(t):
    """CIE-LAB → XYZ の非線形逆関数 / Inverse nonlinear function for LAB → XYZ"""
    delta = 6 / 29
    return np.where(t > delta, t ** 3, 3 * delta**2 * (t - 4 / 29))


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """
    CIE-LAB → XYZ 変換 / Convert LAB to XYZ color space (D65)

    Parameters:
        lab : np.ndarray of shape (N, 3)

    Returns:
        xyz : np.ndarray of shape (N, 3)
    """
    lab = np.asarray(lab)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    x = f_inv(fx) * Xn
    y = f_inv(fy) * Yn
    z = f_inv(fz) * Zn
    return np.stack([x, y, z], axis=1)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """
    XYZ → sRGB 変換（ガンマ補正付き）/ Convert XYZ to sRGB (with gamma correction)

    Parameters:
        xyz : np.ndarray of shape (N, 3)

    Returns:
        srgb : np.ndarray of shape (N, 3), values in [0, 1]
    """
    M = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])
    rgb_lin = xyz @ M.T
    threshold = 0.0031308
    rgb = np.where(
        rgb_lin <= threshold,
        12.92 * rgb_lin,
        1.055 * np.power(rgb_lin, 1 / 2.4) - 0.055
    )
    return np.clip(rgb, 0, 1)

def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """CIE-LAB → sRGB 変換（XYZを経由）/ Convert LAB to sRGB via XYZ"""
    return xyz_to_srgb(lab_to_xyz(lab))


def draw_lab_with_weights(lab_points: np.ndarray, weights: np.ndarray, ratio: float = 5000):
    """
    LAB空間のベクトルと重みから、以下を可視化：
    - 色帯グラフ（重みの大きい順）
    - 3D散布図（バブルサイズ = 重み）

    Visualize color distribution using LAB + weights.

    Parameters:
        lab_points : np.ndarray of shape (N, 3)
        weights : np.ndarray of shape (N,)
        ratio : float, バブルサイズ調整スケーリング

    Returns:
        plt : matplotlib.pyplot object (call .show())
    """
    idx = np.argsort(weights)[::-1]
    lab_points = lab_points[idx]
    weights = weights[idx]
    rgb_points = lab_to_srgb(lab_points)

    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])

    # 色帯グラフ
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.set_title("Color Band", pad=10)
    total_weight = np.sum(weights)
    start = 0.0
    for i in range(len(lab_points)):
        w_frac = weights[i] / total_weight
        ax_bar.add_patch(plt.Rectangle((start, 0), w_frac, 1, color=rgb_points[i]))
        start += w_frac
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')

    # 3D散布図
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    ax_3d.set_title("CIE-LAB 3D Scatter Plot", pad=10)
    ax_3d.scatter(
        lab_points[:, 0], lab_points[:, 1], lab_points[:, 2],
        s=weights * ratio,
        c=rgb_points,
        edgecolors='k',
        alpha=0.6
    )
    ax_3d.set_xlabel("L*")
    ax_3d.set_ylabel("a*")
    ax_3d.set_zlabel("b*")
    plt.tight_layout()
    return fig
