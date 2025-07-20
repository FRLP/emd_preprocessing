"""
Reducer modules for downsampling feature vectors into support points.
"""

from .kmeans import KMeansReducer, KMeansInitReducer
from .rdp import RDPReducer, RDPRawReducer, ReverseRDPReducer
from .vw import VWReducer, VWRawReducer, ReverseVWReducer
