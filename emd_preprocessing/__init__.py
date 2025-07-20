"""
emd_preprocessing: EMD/WMD のためのベクトル前処理・並べ替え・可視化ツールキット

A unified toolkit for preprocessing vectors for EMD/WMD applications,
including reducers, sorters, vectorizers, and visual tools.
"""

from .emd import calculate_emd
from .vectorize import JaSentenceVectorizer, EnSentenceVectorizer
from .image import (
    lab_to_xyz, xyz_to_srgb, lab_to_srgb, draw_lab_with_weights
)
from .synthetic import generate_biased_clusters
from . import sorters 
from . import reducers