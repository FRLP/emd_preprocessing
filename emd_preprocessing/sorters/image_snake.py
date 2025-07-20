import numpy as np
import cv2
from ..base import BaseSorter

class ImageSnakeSorter(BaseSorter):
    """
    画像を CIE-LAB 空間に変換し、蛇腹（ジグザグ）順で並び替えるソーター。

    Sorter that reads an image, converts it to CIE-LAB color space,
    and outputs pixel vectors ordered in snake (zigzag) scan.
    """

    def __init__(self, image_path: str):
        """
        Parameters:
        ----------
        image_path : str
            入力画像のファイルパス / Path to input image (BGR format assumed by OpenCV).
        """
        self.image_path = image_path
        self.features = self._convert_to_lab_snake()

    def _convert_to_lab_snake(self) -> np.ndarray:
        """
        画像を読み込み、LABに変換し、ジグザグ順に並べてベクトル配列にする。

        Read image and convert to CIE-LAB pixels in zigzag order.

        Returns:
        -------
        features : np.ndarray of shape (H*W, 3)
        """
        img_bgr = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        H, W = img_lab.shape[:2]
        pixels = []

        for i in range(H):
            row = img_lab[i] if i % 2 == 0 else img_lab[i][::-1]
            pixels.extend(row)

        return np.array(pixels)

    def sort(self) -> np.ndarray:
        """
        蛇腹順に並んだ LAB ピクセル値ベクトルを返す。

        Returns:
        -------
        sorted_features : np.ndarray of shape (n_pixels, 3)
        """
        return self.features
