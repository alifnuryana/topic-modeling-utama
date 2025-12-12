"""
Topic Modeling with LDA - Core Module

This package contains the core functionality for topic modeling
using Latent Dirichlet Allocation (LDA) on academic paper metadata.
"""

from src.config import Settings, get_settings
from src.harvester import SafeSickle, OAIPMHHarvester
from src.preprocessor import IndonesianPreprocessor
from src.lda_model import LDATopicModel
from src.analysis import TopicAnalyzer
from src.visualizations import TopicVisualizer

__all__ = [
    "Settings",
    "get_settings",
    "SafeSickle",
    "OAIPMHHarvester",
    "IndonesianPreprocessor",
    "LDATopicModel",
    "TopicAnalyzer",
    "TopicVisualizer",
]

__version__ = "0.1.0"
