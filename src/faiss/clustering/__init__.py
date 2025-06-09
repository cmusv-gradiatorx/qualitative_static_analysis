"""
Clustering-Based Grading System

This module provides clustering-based alternatives to FAISS similarity search for
automated grading context generation. Uses sklearn clustering algorithms with
StarCoder2 embeddings to categorize submissions by score and common issues.

Components:
- ClusteringManager: Main clustering interface (uses shared utilities from utils/)
- Training and evaluation scripts for clustering models

Shared utilities (from utils/):
- IssueExtractor: Extract issue categories from feedback text
- ClusteringEvaluator: Evaluate clustering performance
- GradeMappingManager: Handle grade mapping and student matching

Author: Auto-generated
"""

from .cluster_manager import ClusteringManager

__all__ = [
    'ClusteringManager'
]

__version__ = "1.0.0" 