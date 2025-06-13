"""
Repomix Cluster Manager

Handles k-means clustering of repomix embeddings for student submissions.
Groups similar submissions together based on their code embeddings.

Author: Auto-generated
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
import json

from src.faiss.repomix_embedder import RepomixEmbedder
from src.utils.logger import get_logger


class RepomixClusterManager:
    """
    Manages clustering of repomix embeddings for student submissions.
    Uses k-means clustering to group similar submissions.
    """
    
    def __init__(self, 
                 model_name: str = "starCoder2:3b",
                 max_tokens: int = 128000,
                 use_compression: bool = True,
                 remove_comments: bool = False,
                 ignore_patterns: Optional[List[str]] = None,
                 keep_patterns: Optional[List[str]] = None,
                 max_file_size: Optional[int] = None,
                 ollama_base_url: Optional[str] = None):
        """
        Initialize the RepomixClusterManager.
        
        Args:
            model_name: Model name for Ollama (default: starCoder2:3b)
            max_tokens: Maximum token limit for repomix processing
            use_compression: Whether to use compression in repomix
            remove_comments: Whether to remove comments in repomix
            ignore_patterns: List of file/directory patterns to ignore
            keep_patterns: List of file patterns to keep
            max_file_size: Maximum file size in bytes for filtering
            ollama_base_url: Base URL for Ollama API
        """
        self.logger = get_logger(__name__)
        
        # Initialize embedder
        self.embedder = RepomixEmbedder(
            model_name=model_name,
            max_tokens=max_tokens,
            use_compression=use_compression,
            remove_comments=remove_comments,
            ignore_patterns=ignore_patterns,
            keep_patterns=keep_patterns,
            max_file_size=max_file_size,
            ollama_base_url=ollama_base_url
        )
        
        # Initialize clustering model
        self.kmeans = None
        self.embeddings = None
        self.metadata_list = None
        
    def process_submissions(self, zip_paths: List[Path]) -> None:
        """
        Process all submission ZIP files and generate embeddings.
        
        Args:
            zip_paths: List of paths to submission ZIP files
        """
        self.logger.info(f"Processing {len(zip_paths)} submissions...")
        
        # Process all submissions
        results = self.embedder.process_and_embed_batch(zip_paths)
        
        # Extract embeddings and metadata
        self.embeddings = np.array([r[0] for r in results])
        self.metadata_list = [r[1] for r in results]
        
        self.logger.info(f"Successfully processed {len(self.embeddings)} submissions")
    
    def train_clustering(self, n_clusters: int, random_state: int = 42) -> Dict[str, Any]:
        """
        Train k-means clustering on the embeddings.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing clustering results and statistics
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available. Call process_submissions first.")
        
        self.logger.info(f"Training k-means clustering with {n_clusters} clusters...")
        
        # Initialize and train k-means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10  # Run multiple times and pick best result
        )
        
        # Fit the model
        cluster_labels = self.kmeans.fit_predict(self.embeddings)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(cluster_labels)
        
        # Create cluster assignments
        cluster_assignments = self._create_cluster_assignments(cluster_labels)
        
        return {
            'cluster_assignments': cluster_assignments,
            'cluster_statistics': cluster_stats,
            'model_info': {
                'n_clusters': n_clusters,
                'n_samples': len(self.embeddings),
                'inertia': self.kmeans.inertia_,
                'n_iterations': self.kmeans.n_iter_
            }
        }
    
    def _calculate_cluster_statistics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for each cluster.
        
        Args:
            cluster_labels: Array of cluster assignments
            
        Returns:
            Dictionary containing cluster statistics
        """
        n_clusters = len(np.unique(cluster_labels))
        stats = {}
        
        for i in range(n_clusters):
            # Get indices of samples in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            # Calculate cluster center
            cluster_center = self.kmeans.cluster_centers_[i]
            
            # Calculate average distance to center
            distances = np.linalg.norm(
                self.embeddings[cluster_indices] - cluster_center,
                axis=1
            )
            avg_distance = np.mean(distances)
            
            stats[f'cluster_{i}'] = {
                'size': len(cluster_indices),
                'avg_distance_to_center': float(avg_distance),
                'student_names': [
                    self.metadata_list[idx]['student_name']
                    for idx in cluster_indices
                ]
            }
        
        return stats
    
    def _create_cluster_assignments(self, cluster_labels: np.ndarray) -> Dict[str, List[str]]:
        """
        Create a mapping of clusters to student names.
        
        Args:
            cluster_labels: Array of cluster assignments
            
        Returns:
            Dictionary mapping cluster IDs to lists of student names
        """
        assignments = {}
        
        for i in range(len(np.unique(cluster_labels))):
            # Get indices of samples in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            # Get student names for this cluster
            student_names = [
                self.metadata_list[idx]['student_name']
                for idx in cluster_indices
            ]
            
            assignments[f'cluster_{i}'] = student_names
        
        return assignments
    
    def save_clustering_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save clustering results to a JSON file.
        
        Args:
            results: Clustering results dictionary
            output_path: Path to save the results
        """
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results to JSON-serializable format
        json_results = json.loads(
            json.dumps(results, default=convert_numpy)
        )
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Saved clustering results to {output_path}")
    
    def get_cluster_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of clustering results.
        
        Args:
            results: Clustering results dictionary
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("Clustering Results Summary:")
        summary.append("-" * 30)
        
        # Add model info
        model_info = results['model_info']
        summary.append(f"Number of clusters: {model_info['n_clusters']}")
        summary.append(f"Total submissions: {model_info['n_samples']}")
        summary.append(f"Clustering inertia: {model_info['inertia']:.2f}")
        summary.append(f"Number of iterations: {model_info['n_iterations']}")
        summary.append("")
        
        # Add cluster statistics
        stats = results['cluster_statistics']
        for cluster_id, cluster_info in stats.items():
            summary.append(f"{cluster_id}:")
            summary.append(f"  Size: {cluster_info['size']} submissions")
            summary.append(f"  Average distance to center: {cluster_info['avg_distance_to_center']:.4f}")
            summary.append(f"  Students: {', '.join(cluster_info['student_names'])}")
            summary.append("")
        
        return "\n".join(summary) 