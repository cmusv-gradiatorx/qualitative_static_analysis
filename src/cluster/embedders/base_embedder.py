"""
Base Embedder Abstract Class

Defines the interface that all embedders must implement for the clustering system.
Provides common utilities and configuration management.

Author: Auto-generated
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

from ...utils.logger import get_logger

# Load environment variables from config.env
config_path = Path(__file__).parent.parent.parent.parent / 'config.env'
if config_path.exists():
    load_dotenv(config_path)

@dataclass
class EmbedderConfig:
    """Configuration for embedders"""
    model_name: str = "starCoder2:3b"
    ollama_base_url: str = "http://localhost:11434"
    max_length: int = 8192
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_dir: str = ".cache/cluster"
    timeout: int = 60
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', self.ollama_base_url)
        self.cache_dir = os.environ.get('CLUSTER_CACHE_DIR', self.cache_dir)
        
        # Create cache directory
        if self.cache_embeddings:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.
    
    All embedders must implement the generate_embedding method.
    Provides common utilities for caching, normalization, and logging.
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """
        Initialize the base embedder.
        
        Args:
            config: Embedder configuration (uses default if None)
        """
        self.config = config or EmbedderConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Set embedding dimension (to be overridden by subclasses)
        self.embedding_dim = self._get_embedding_dimension()
        
        self.logger.info(f"Initialized {self.__class__.__name__} with model: {self.config.model_name}")
    
    @abstractmethod
    def generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate embedding for the given content.
        
        Args:
            content: Text content to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate_embedding method")
    
    @abstractmethod
    def _get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for this embedder.
        
        Returns:
            Dimension of embeddings produced by this embedder
        """
        raise NotImplementedError("Subclasses must implement _get_embedding_dimension method")
    
    def embed_submission(self, submission_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a submission, handling caching and normalization.
        
        Args:
            submission_data: Dictionary containing submission information
                            Must include 'student_name' and content fields
            
        Returns:
            Embedding vector
        """
        student_name = submission_data.get('student_name', 'unknown')
        
        # Check cache first
        if self.config.cache_embeddings:
            cached_embedding = self._load_cached_embedding(student_name)
            if cached_embedding is not None:
                self.logger.debug(f"Loaded cached embedding for {student_name}")
                return cached_embedding
        
        # Generate new embedding
        content = self._prepare_content(submission_data)
        embedding = self.generate_embedding(content)
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
        
        # Cache the embedding
        if self.config.cache_embeddings:
            self._save_cached_embedding(student_name, embedding)
        
        return embedding
    
    def embed_batch(self, submission_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Embed multiple submissions.
        
        Args:
            submission_list: List of submission data dictionaries
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, submission in enumerate(submission_list):
            self.logger.debug(f"Processing submission {i+1}/{len(submission_list)}")
            try:
                embedding = self.embed_submission(submission)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Failed to embed submission {submission.get('student_name', i)}: {e}")
                # Add zero embedding as fallback
                embeddings.append(self._get_zero_embedding())
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Ensure embeddings are normalized
        embedding1 = self._normalize_embedding(embedding1)
        embedding2 = self._normalize_embedding(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Clamp to [0, 1] range
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about this embedder"""
        return {
            'embedder_type': self.__class__.__name__,
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'ollama_base_url': self.config.ollama_base_url,
            'max_length': self.config.max_length,
            'normalize_embeddings': self.config.normalize_embeddings,
            'cache_embeddings': self.config.cache_embeddings
        }
    
    @abstractmethod
    def _prepare_content(self, submission_data: Dict[str, Any]) -> str:
        """
        Prepare content for embedding (to be implemented by subclasses).
        
        Args:
            submission_data: Submission data dictionary
            
        Returns:
            Prepared content string
        """
        raise NotImplementedError("Subclasses must implement _prepare_content method")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _get_zero_embedding(self) -> np.ndarray:
        """Return zero embedding as fallback"""
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _get_cache_path(self, student_name: str) -> Path:
        """Get cache file path for a student"""
        cache_dir = Path(self.config.cache_dir) / self.__class__.__name__
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from student name and model
        filename = f"{student_name}_{self.config.model_name.replace(':', '_')}.npy"
        return cache_dir / filename
    
    def _load_cached_embedding(self, student_name: str) -> Optional[np.ndarray]:
        """Load cached embedding if it exists"""
        try:
            cache_path = self._get_cache_path(student_name)
            if cache_path.exists():
                return np.load(cache_path)
        except Exception as e:
            self.logger.warning(f"Failed to load cached embedding for {student_name}: {e}")
        return None
    
    def _save_cached_embedding(self, student_name: str, embedding: np.ndarray) -> None:
        """Save embedding to cache"""
        try:
            cache_path = self._get_cache_path(student_name)
            np.save(cache_path, embedding)
            self.logger.debug(f"Cached embedding for {student_name}")
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding for {student_name}: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings for this embedder"""
        try:
            cache_dir = Path(self.config.cache_dir) / self.__class__.__name__
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                self.logger.info(f"Cleared cache for {self.__class__.__name__}")
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}") 