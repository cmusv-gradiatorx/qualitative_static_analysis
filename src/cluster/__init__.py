"""
Clean Clustering System for Automated Grading

A modular clustering system for grouping similar student submissions
using various embedding techniques and clustering algorithms.

Author: Auto-generated
"""

__version__ = "1.0.0"
__author__ = "Auto-generated"

from .embedders import EmbedderFactory
from .processors import SubmissionProcessor, Submission
from .clustering import ClusterManager, ClusteringResult

__all__ = [
    "EmbedderFactory",
    "SubmissionProcessor", 
    "Submission",
    "ClusterManager",
    "ClusteringResult"
] 