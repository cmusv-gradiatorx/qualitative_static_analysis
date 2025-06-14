"""
Issue Embedder

Semantic embeddings for student issues using sentence transformers.
Groups students based on similar issue patterns and problem types.

Author: Auto-generated
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from .base_embedder import BaseEmbedder, EmbedderConfig

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class IssueConfig(EmbedderConfig):
    """Extended configuration for Issue embedder"""
    sentence_model: str = "all-MiniLM-L6-v2"  # Lightweight, fast model
    issues_file: Optional[str] = None
    max_issues_per_student: int = 50
    min_issue_frequency: int = 1  # Minimum times an issue must appear to be included
    use_issue_clustering: bool = True  # Whether to cluster similar issues first
    similarity_threshold: float = 0.8  # Threshold for considering issues similar
    
    def __post_init__(self):
        super().__post_init__()
        if self.issues_file is None:
            self.issues_file = "src/cluster/data/student_issues.json"


class IssueEmbedder(BaseEmbedder):
    """
    Issue embedder that creates semantic embeddings based on student issues.
    Uses sentence transformers to understand semantic similarity between issues.
    """
    
    def __init__(self, config: IssueConfig = None):
        """Initialize the Issue embedder"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for IssueEmbedder. "
                "Install with: pip install sentence-transformers"
            )
        
        self.issue_config = config or IssueConfig()
        super().__init__(self.issue_config)
        
        # Initialize sentence transformer model
        self.logger.info(f"Loading sentence transformer model: {self.issue_config.sentence_model}")
        self.sentence_model = SentenceTransformer(self.issue_config.sentence_model)
        
        # Load and process issues data
        self.issues_data = self._load_issues_data()
        self.issue_clusters = self._cluster_similar_issues() if self.issue_config.use_issue_clustering else {}
        self.issue_vocabulary = self._build_issue_vocabulary()
        
        self.logger.info(f"Loaded {len(self.issue_vocabulary)} unique issue types")
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on sentence transformer model + profile dimension"""
        model_dimensions = {
            'all-MiniLM-L6-v2': 384,
            'all-MiniLM-L12-v2': 384,
            'all-mpnet-base-v2': 768,
            'paraphrase-MiniLM-L6-v2': 384,
            'paraphrase-mpnet-base-v2': 768,
            'distilbert-base-nli-stsb-mean-tokens': 768,
            'sentence-transformers/all-MiniLM-L6-v2': 384,
            'sentence-transformers/all-mpnet-base-v2': 768
        }
        
        # Get semantic embedding dimension
        semantic_dim = 384  # Default
        
        # Check exact match first
        if self.issue_config.sentence_model in model_dimensions:
            semantic_dim = model_dimensions[self.issue_config.sentence_model]
        else:
            # Check for partial matches
            for model_key, dim in model_dimensions.items():
                if model_key in self.issue_config.sentence_model:
                    semantic_dim = dim
                    break
        
        # If sentence model is not loaded yet, we need to estimate
        if not hasattr(self, 'sentence_model') or self.sentence_model is None:
            # During initialization, use estimated dimensions
            profile_dim = len(self.issue_vocabulary) if hasattr(self, 'issue_vocabulary') else 0
        else:
            # After initialization, get actual dimensions
            try:
                test_embedding = self.sentence_model.encode(["test"])
                semantic_dim = test_embedding.shape[1]
            except Exception as e:
                self.logger.warning(f"Could not determine embedding dimension: {e}")
            
            profile_dim = len(self.issue_vocabulary) if hasattr(self, 'issue_vocabulary') else 0
        
        return semantic_dim + profile_dim
    
    def _load_issues_data(self) -> Dict[str, Any]:
        """Load issues data from JSON file"""
        issues_path = Path(self.issue_config.issues_file)
        
        if not issues_path.exists():
            self.logger.error(f"Issues file not found: {issues_path}")
            raise FileNotFoundError(f"Issues file not found: {issues_path}")
        
        try:
            with open(issues_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded issues data for {data['metadata']['total_students']} students")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to load issues data: {e}")
            raise
    
    def _cluster_similar_issues(self) -> Dict[str, List[str]]:
        """Cluster semantically similar issues together"""
        if not self.issues_data:
            return {}
        
        # Get all unique issues
        all_issues = set()
        for student_issues in self.issues_data['student_issues'].values():
            all_issues.update(student_issues)
        
        all_issues = list(all_issues)
        
        if len(all_issues) < 2:
            return {}
        
        # Generate embeddings for all issues
        self.logger.info(f"Generating embeddings for {len(all_issues)} unique issues...")
        issue_embeddings = self.sentence_model.encode(all_issues)
        
        # Find similar issues using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(issue_embeddings)
        
        # Group similar issues
        issue_clusters = {}
        processed_issues = set()
        cluster_id = 0
        
        for i, issue in enumerate(all_issues):
            if issue in processed_issues:
                continue
            
            # Find all issues similar to this one
            similar_indices = np.where(similarity_matrix[i] >= self.issue_config.similarity_threshold)[0]
            similar_issues = [all_issues[j] for j in similar_indices]
            
            if len(similar_issues) > 1:
                cluster_key = f"issue_cluster_{cluster_id}"
                issue_clusters[cluster_key] = similar_issues
                processed_issues.update(similar_issues)
                cluster_id += 1
                
                self.logger.debug(f"Created issue cluster: {similar_issues}")
        
        self.logger.info(f"Created {len(issue_clusters)} issue clusters")
        return issue_clusters
    
    def _build_issue_vocabulary(self) -> List[str]:
        """Build vocabulary of issue types for embedding"""
        if self.issue_config.use_issue_clustering and self.issue_clusters:
            # Use clustered issues as vocabulary
            vocabulary = list(self.issue_clusters.keys())
        else:
            # Use individual issues as vocabulary
            issue_counts = defaultdict(int)
            for student_issues in self.issues_data['student_issues'].values():
                for issue in student_issues:
                    issue_counts[issue] += 1
            
            # Filter by minimum frequency
            vocabulary = [
                issue for issue, count in issue_counts.items()
                if count >= self.issue_config.min_issue_frequency
            ]
        
        return sorted(vocabulary)
    
    def _prepare_content(self, submission_data: Dict[str, Any]) -> str:
        """
        Prepare issues content for embedding.
        
        Args:
            submission_data: Dictionary containing submission data with 'student_name' key
            
        Returns:
            Processed issues string
        """
        student_name = submission_data.get('student_name', 'unknown')
        
        # Get student issues
        student_issues = self.issues_data['student_issues'].get(student_name, [])
        
        if not student_issues:
            self.logger.warning(f"No issues found for {student_name}")
            return "No issues found"
        
        # Limit number of issues
        if len(student_issues) > self.issue_config.max_issues_per_student:
            student_issues = student_issues[:self.issue_config.max_issues_per_student]
        
        # Join issues into a single text
        issues_text = " | ".join(student_issues)
        
        return issues_text
    
    def generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate embedding using sentence transformer.
        
        Args:
            content: Issues content to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Generate embedding using sentence transformer
            embedding = self.sentence_model.encode(content)
            
            # Ensure it's a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            self.logger.debug(f"Generated embedding with shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return self._get_zero_embedding()
    
    def create_issue_profile_embedding(self, submission_data: Dict[str, Any]) -> np.ndarray:
        """
        Create an embedding based on issue profile (presence/absence of issue types).
        
        Args:
            submission_data: Dictionary containing submission data
            
        Returns:
            Binary/weighted embedding vector
        """
        student_name = submission_data.get('student_name', 'unknown')
        student_issues = self.issues_data['student_issues'].get(student_name, [])
        
        # Create binary vector based on issue vocabulary
        issue_vector = np.zeros(len(self.issue_vocabulary), dtype=np.float32)
        
        if self.issue_config.use_issue_clustering and self.issue_clusters:
            # Use clustered approach
            for i, cluster_key in enumerate(self.issue_vocabulary):
                cluster_issues = self.issue_clusters.get(cluster_key, [])
                # Check if any issue in this cluster is present
                if any(issue in student_issues for issue in cluster_issues):
                    issue_vector[i] = 1.0
        else:
            # Use individual issues
            for i, issue_type in enumerate(self.issue_vocabulary):
                if issue_type in student_issues:
                    issue_vector[i] = 1.0
        
        return issue_vector
    
    def embed_submission(self, submission_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a submission using both semantic and profile-based approaches.
        
        Args:
            submission_data: Dictionary containing submission information
            
        Returns:
            Combined embedding vector
        """
        student_name = submission_data.get('student_name', 'unknown')
        
        # Check cache first
        if self.config.cache_embeddings:
            cached_embedding = self._load_cached_embedding(student_name)
            if cached_embedding is not None:
                self.logger.debug(f"Loaded cached embedding for {student_name}")
                return cached_embedding
        
        # Generate semantic embedding
        content = self._prepare_content(submission_data)
        semantic_embedding = self.generate_embedding(content)
        
        # Generate profile embedding
        profile_embedding = self.create_issue_profile_embedding(submission_data)
        
        # Combine embeddings (you can experiment with different combination strategies)
        if len(profile_embedding) > 0:
            # Weighted combination
            combined_embedding = np.concatenate([
                semantic_embedding * 0.7,  # Semantic features
                profile_embedding * 0.3    # Profile features
            ])
        else:
            combined_embedding = semantic_embedding
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            combined_embedding = self._normalize_embedding(combined_embedding)
        
        # Cache the embedding
        if self.config.cache_embeddings:
            self._save_cached_embedding(student_name, combined_embedding)
        
        return combined_embedding
    

    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about this embedder"""
        base_info = super().get_embedding_info()
        base_info.update({
            'sentence_model': self.issue_config.sentence_model,
            'issues_file': self.issue_config.issues_file,
            'issue_vocabulary_size': len(self.issue_vocabulary),
            'use_issue_clustering': self.issue_config.use_issue_clustering,
            'issue_clusters_count': len(self.issue_clusters),
            'total_students': self.issues_data.get('metadata', {}).get('total_students', 0)
        })
        return base_info
    
    def analyze_issue_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of issues across students"""
        issue_counts = defaultdict(int)
        student_issue_counts = []
        
        for student, issues in self.issues_data['student_issues'].items():
            student_issue_counts.append(len(issues))
            for issue in issues:
                issue_counts[issue] += 1
        
        return {
            'total_students': len(self.issues_data['student_issues']),
            'avg_issues_per_student': np.mean(student_issue_counts),
            'median_issues_per_student': np.median(student_issue_counts),
            'max_issues_per_student': np.max(student_issue_counts),
            'min_issues_per_student': np.min(student_issue_counts),
            'most_common_issues': dict(sorted(issue_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]),
            'total_unique_issues': len(issue_counts)
        } 