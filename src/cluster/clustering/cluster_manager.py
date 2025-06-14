"""
Cluster Manager

Manages clustering of student submissions using various algorithms.
Algorithm-agnostic design that works with any embedder implementation.

Author: Auto-generated
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

from ..embedders.base_embedder import BaseEmbedder
from ..processors.submission_processor import Submission
from ...utils.logger import get_logger


@dataclass
class ClusteringResult:
    """Results from clustering operation"""
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    n_clusters: int
    algorithm: str
    silhouette_score: float
    calinski_harabasz_score: float
    inertia: Optional[float]
    metadata: Dict[str, Any]


class ClusterManager:
    """
    Manages clustering of student submissions.
    
    Algorithm-agnostic design that works with any embedder implementation.
    Supports multiple clustering algorithms and evaluation metrics.
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize cluster manager.
        
        Args:
            embedder: Embedder instance for generating embeddings
        """
        self.embedder = embedder
        self.logger = get_logger(__name__)
        
        # Clustering state
        self.submissions = []
        self.embeddings = None
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.clustering_result = None
        
        self.logger.info(f"Initialized ClusterManager with {embedder.__class__.__name__}")
    
    def fit(self, 
            submissions: List[Submission],
            algorithm: str = "kmeans",
            n_clusters: int = 5,
            random_state: int = 42,
            **algorithm_kwargs) -> ClusteringResult:
        """
        Fit clustering model on submissions.
        
        Args:
            submissions: List of submission objects
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            n_clusters: Number of clusters (ignored for DBSCAN)
            random_state: Random seed for reproducibility
            **algorithm_kwargs: Additional algorithm-specific parameters
            
        Returns:
            ClusteringResult object
        """
        self.submissions = submissions
        self.logger.info(f"Fitting {algorithm} clustering on {len(submissions)} submissions")
        
        # Generate embeddings
        self._generate_embeddings()
        
        # Scale embeddings
        scaled_embeddings = self.scaler.fit_transform(self.embeddings)
        
        # Create clustering model
        self.clustering_model = self._create_clusterer(
            algorithm, n_clusters, random_state, **algorithm_kwargs
        )
        
        # Fit clustering
        if hasattr(self.clustering_model, 'fit_predict'):
            cluster_labels = self.clustering_model.fit_predict(scaled_embeddings)
        else:
            self.clustering_model.fit(scaled_embeddings)
            cluster_labels = self.clustering_model.labels_
        
        # Get cluster centers if available
        cluster_centers = None
        if hasattr(self.clustering_model, 'cluster_centers_'):
            cluster_centers = self.clustering_model.cluster_centers_
        elif hasattr(self.clustering_model, 'means_'):
            cluster_centers = self.clustering_model.means_
        
        # Calculate metrics
        n_clusters_found = len(np.unique(cluster_labels))
        if n_clusters_found > 1:
            silhouette = silhouette_score(scaled_embeddings, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(scaled_embeddings, cluster_labels)
        else:
            silhouette = 0.0
            calinski_harabasz = 0.0
        
        # Get inertia if available
        inertia = getattr(self.clustering_model, 'inertia_', None)
        
        # Create metadata
        metadata = {
            'embedder_info': self.embedder.get_embedding_info(),
            'submission_count': len(submissions),
            'embedding_dimension': self.embeddings.shape[1],
            'algorithm_kwargs': algorithm_kwargs,
            'fit_timestamp': datetime.now().isoformat(),
            'student_names': [sub.student_name for sub in submissions]
        }
        
        # Store result
        self.clustering_result = ClusteringResult(
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            n_clusters=n_clusters_found,
            algorithm=algorithm,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            inertia=inertia,
            metadata=metadata
        )
        
        self.logger.info(f"Clustering complete: {n_clusters_found} clusters, silhouette={silhouette:.3f}")
        return self.clustering_result
    
    def predict(self, submission: Submission) -> int:
        """
        Predict cluster for a new submission.
        
        Args:
            submission: New submission to classify
            
        Returns:
            Cluster ID
        """
        if self.clustering_model is None:
            raise ValueError("Must fit clustering model first")
        
        # Convert submission to format expected by embedder based on embedder type
        embedder_class_name = self.embedder.__class__.__name__
        
        if embedder_class_name == 'JavaEmbedder':
            # Java embedder format
            submission_data = {
                'student_name': submission.student_name,
                'java_files': getattr(submission, 'java_files', {})
            }
        elif embedder_class_name == 'RepomixEmbedder':
            # Repomix embedder format
            submission_data = {
                'student_name': submission.student_name,
                'zip_path': str(getattr(submission, 'zip_path', ''))
            }
        elif embedder_class_name == 'IssueEmbedder':
            # Issue embedder format - only needs student name
            submission_data = {
                'student_name': submission.student_name
            }
        else:
            # Fallback: try to include available attributes
            submission_data = {
                'student_name': submission.student_name
            }
            
            # Add optional attributes if they exist
            if hasattr(submission, 'zip_path'):
                submission_data['zip_path'] = str(submission.zip_path)
            if hasattr(submission, 'java_files'):
                submission_data['java_files'] = submission.java_files
        
        # Generate embedding
        embedding = self.embedder.embed_submission(submission_data)
        
        # Scale embedding
        scaled_embedding = self.scaler.transform(embedding.reshape(1, -1))
        
        # Predict cluster
        if hasattr(self.clustering_model, 'predict'):
            cluster_id = self.clustering_model.predict(scaled_embedding)[0]
        else:
            # For DBSCAN, assign to closest cluster center
            cluster_id = self._assign_to_closest_cluster(scaled_embedding[0])
        
        return int(cluster_id)
    
    def get_cluster_analysis(self, submissions: Optional[List[Submission]] = None) -> Dict[str, Any]:
        """
        Get detailed analysis of clusters.
        
        Args:
            submissions: Optional list of submissions for detailed analysis.
                        If not provided, uses metadata from clustering result.
        
        Returns:
            Dictionary containing cluster analysis
        """
        if self.clustering_result is None:
            raise ValueError("Must fit clustering model first")
        
        analysis = {}
        
        # Use provided submissions or try to use existing ones
        submissions_to_use = submissions or self.submissions
        
        for cluster_id in np.unique(self.clustering_result.cluster_labels):
            # Get submissions in this cluster
            cluster_mask = self.clustering_result.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Get student names from metadata if submissions not available
            if not submissions_to_use or len(submissions_to_use) != len(self.clustering_result.cluster_labels):
                # Use student names from metadata
                all_student_names = self.clustering_result.metadata.get('student_names', [])
                if len(all_student_names) == len(self.clustering_result.cluster_labels):
                    cluster_student_names = [all_student_names[i] for i in cluster_indices]
                else:
                    cluster_student_names = [f"student_{i}" for i in cluster_indices]
                
                # Set default values for file counts
                java_files_avg = 0.0
                total_files_avg = 0.0
            else:
                # Use actual submissions
                cluster_submissions = [submissions_to_use[i] for i in cluster_indices]
                cluster_student_names = [sub.student_name for sub in cluster_submissions]
                java_files_avg = np.mean([sub.get_java_file_count() for sub in cluster_submissions])
                total_files_avg = np.mean([sub.get_file_count() for sub in cluster_submissions])
            
            # Calculate cluster statistics from embeddings
            cluster_embeddings = self.embeddings[cluster_mask]
            center = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            
            analysis[f"cluster_{cluster_id}"] = {
                'size': int(np.sum(cluster_mask)),
                'student_names': cluster_student_names,
                'avg_distance_to_center': float(np.mean(distances)),
                'max_distance_to_center': float(np.max(distances)),
                'compactness': float(np.std(distances)),
                'java_files_avg': java_files_avg,
                'total_files_avg': total_files_avg
            }
        
        return analysis
    
    def save_model(self, save_path: str) -> None:
        """
        Save trained clustering model and metadata.
        
        Args:
            save_path: Directory to save model files
        """
        if self.clustering_result is None:
            raise ValueError("Must fit clustering model first")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn model
        model_path = save_dir / "clustering_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.clustering_model, f)
        
        # Save scaler
        scaler_path = save_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save embeddings
        embeddings_path = save_dir / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        
        # Save clustering result (convert to JSON-serializable format)
        result_data = {
            'cluster_labels': self.clustering_result.cluster_labels.tolist(),
            'n_clusters': self.clustering_result.n_clusters,
            'algorithm': self.clustering_result.algorithm,
            'silhouette_score': self.clustering_result.silhouette_score,
            'calinski_harabasz_score': self.clustering_result.calinski_harabasz_score,
            'inertia': self.clustering_result.inertia,
            'metadata': self.clustering_result.metadata
        }
        
        if self.clustering_result.cluster_centers is not None:
            result_data['cluster_centers'] = self.clustering_result.cluster_centers.tolist()
        
        result_path = save_dir / "clustering_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Save cluster analysis
        analysis_path = save_dir / "cluster_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.get_cluster_analysis(), f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load trained clustering model.
        
        Args:
            load_path: Directory containing model files
        """
        load_dir = Path(load_path)
        
        # Load sklearn model
        model_path = load_dir / "clustering_model.pkl"
        with open(model_path, 'rb') as f:
            self.clustering_model = pickle.load(f)
        
        # Load scaler
        scaler_path = load_dir / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load embeddings
        embeddings_path = load_dir / "embeddings.npy"
        self.embeddings = np.load(embeddings_path)
        
        # Load clustering result
        result_path = load_dir / "clustering_result.json"
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Reconstruct clustering result
        cluster_centers = None
        if 'cluster_centers' in result_data:
            cluster_centers = np.array(result_data['cluster_centers'])
        
        self.clustering_result = ClusteringResult(
            cluster_labels=np.array(result_data['cluster_labels']),
            cluster_centers=cluster_centers,
            n_clusters=result_data['n_clusters'],
            algorithm=result_data['algorithm'],
            silhouette_score=result_data['silhouette_score'],
            calinski_harabasz_score=result_data['calinski_harabasz_score'],
            inertia=result_data.get('inertia'),
            metadata=result_data['metadata']
        )
        
        self.logger.info(f"Model loaded from {load_dir}")
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all submissions"""
        self.logger.info("Generating embeddings...")
        
        submission_data_list = []
        
        for submission in self.submissions:
            # Convert submission to format expected by embedder based on embedder type
            embedder_class_name = self.embedder.__class__.__name__
            
            if embedder_class_name == 'JavaEmbedder':
                # Java embedder format
                submission_data = {
                    'student_name': submission.student_name,
                    'java_files': getattr(submission, 'java_files', {})
                }
            elif embedder_class_name == 'RepomixEmbedder':
                # Repomix embedder format
                submission_data = {
                    'student_name': submission.student_name,
                    'zip_path': str(getattr(submission, 'zip_path', ''))
                }
            elif embedder_class_name == 'IssueEmbedder':
                # Issue embedder format - only needs student name
                submission_data = {
                    'student_name': submission.student_name
                }
            else:
                # Fallback: try to include available attributes
                submission_data = {
                    'student_name': submission.student_name
                }
                
                # Add optional attributes if they exist
                if hasattr(submission, 'zip_path'):
                    submission_data['zip_path'] = str(submission.zip_path)
                if hasattr(submission, 'java_files'):
                    submission_data['java_files'] = submission.java_files
            
            submission_data_list.append(submission_data)
        
        # Generate embeddings in batch
        embeddings = self.embedder.embed_batch(submission_data_list)
        self.embeddings = np.vstack(embeddings)
        
        self.logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
    
    def _create_clusterer(self, algorithm: str, n_clusters: int, 
                         random_state: int, **kwargs):
        """Create clustering algorithm instance"""
        algorithm = algorithm.lower()
        
        if algorithm == "kmeans":
            return KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                n_init=10,
                **kwargs
            )
        elif algorithm == "dbscan":
            return DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 2),
                **{k: v for k, v in kwargs.items() if k not in ['eps', 'min_samples']}
            )
        elif algorithm == "hierarchical":
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                **kwargs
            )
        elif algorithm == "gmm":
            return GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                **kwargs
            )
        else:
            available = ["kmeans", "dbscan", "hierarchical", "gmm"]
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    def _assign_to_closest_cluster(self, embedding: np.ndarray) -> int:
        """Assign embedding to closest cluster (for algorithms without predict method)"""
        if self.clustering_result.cluster_centers is None:
            # For DBSCAN without clear centers, return most common cluster
            unique, counts = np.unique(self.clustering_result.cluster_labels, return_counts=True)
            return unique[np.argmax(counts)]
        
        # Calculate distances to all cluster centers
        distances = np.linalg.norm(self.clustering_result.cluster_centers - embedding, axis=1)
        return np.argmin(distances)
    
    def evaluate_clustering(self, true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering performance.
        
        Args:
            true_labels: Optional true cluster labels for supervised evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.clustering_result is None:
            raise ValueError("Must fit clustering model first")
        
        metrics = {
            'silhouette_score': self.clustering_result.silhouette_score,
            'calinski_harabasz_score': self.clustering_result.calinski_harabasz_score,
            'n_clusters': self.clustering_result.n_clusters,
            'n_samples': len(self.clustering_result.cluster_labels)
        }
        
        if self.clustering_result.inertia is not None:
            metrics['inertia'] = self.clustering_result.inertia
        
        # Add supervised metrics if true labels provided
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(
                true_labels, self.clustering_result.cluster_labels
            )
        
        return metrics 