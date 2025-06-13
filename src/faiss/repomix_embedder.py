"""
Repomix Embedder

Combines RepomixProcessor with direct StarCoder2 embedding for processed codebases.
First processes codebases using repomix, then vectorizes the processed output using StarCoder2.

Author: Auto-generated
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import tempfile
import requests
import json
from dotenv import load_dotenv

from .processor import RepomixProcessor
from ..utils.logger import get_logger

# Load configuration on module import
config_path = Path(__file__).parent.parent.parent / 'config.env'
if config_path.exists():
    load_dotenv(config_path)


class RepomixEmbedder:
    """
    Combines RepomixProcessor with direct StarCoder2 embedding for processed codebases.
    First processes codebases using repomix, then vectorizes the processed output.
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
        Initialize the RepomixEmbedder.
        
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
        
        # Initialize repomix processor
        self.repomix_processor = RepomixProcessor(
            max_tokens=max_tokens,
            use_compression=use_compression,
            remove_comments=remove_comments,
            ignore_patterns=ignore_patterns,
            keep_patterns=keep_patterns,
            max_file_size=max_file_size
        )
        
        # Set up Ollama configuration
        self.model_name = model_name
        # Load from environment variable if available
        self.ollama_base_url = ollama_base_url or os.environ.get('OLLAMA_BASE_URL', "http://localhost:11434")
        
        # Set embedding dimension based on model
        self.embedding_dim = self._get_embedding_dimension()
        
        # Test connection
        self._test_ollama_connection()
        
        self.logger.info(f"RepomixEmbedder initialized with model: {model_name}")
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model"""
        model_dimensions = {
            'starCoder2:3b': 2048,
            'starCoder2:7b': 4096, 
            'starCoder2:15b': 6144,
            'starcoder:3b': 2048,
            'starcoder:7b': 4096,
            'qwen2.5-coder:3b': 2048,
            'qwen2.5-coder:7b': 4096,
            'codellama:7b': 4096,
            'codellama:13b': 5120
        }
        
        # Check exact match first
        if self.model_name in model_dimensions:
            return model_dimensions[self.model_name]
        
        # Check for partial matches
        for model_key, dim in model_dimensions.items():
            if model_key.split(':')[0] in self.model_name:
                return dim
        
        # Default fallback
        self.logger.warning(f"Unknown model {self.model_name}, using default dimension 2048")
        return 2048
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check if our model is available
                if not any(self.model_name in name for name in model_names):
                    self.logger.warning(f"Model {self.model_name} not found in Ollama. Available: {model_names}")
                    self.logger.info(f"To pull the model, run: ollama pull {self.model_name}")
                else:
                    self.logger.info(f"Model {self.model_name} available in Ollama")
            else:
                raise requests.RequestException(f"HTTP {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"Cannot connect to Ollama at {self.ollama_base_url}: {e}")
            self.logger.info("Make sure Ollama is running and accessible")
    
    def _extract_student_name(self, zip_path: Path) -> str:
        """
        Extract student name from ZIP filename (format: studentName_otherthings....zip)
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Student name extracted from filename
        """
        # Get filename without extension
        filename = zip_path.stem
        
        # Split by first underscore and take first part
        student_name = filename.split('_')[0]
        
        return student_name
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using Ollama API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            # Create a prompt that encourages semantic understanding
            prompt = f"""Please analyze this code and understand its semantic meaning:

{text}

This code represents a complete implementation."""
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": prompt
                },
                timeout=60  # Increased timeout for larger models
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['embedding'], dtype=np.float32)
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
                
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._get_zero_embedding()
                
        except requests.RequestException as e:
            self.logger.error(f"Error connecting to Ollama: {str(e)}")
            return self._get_zero_embedding()
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return self._get_zero_embedding()
    
    def _get_zero_embedding(self) -> np.ndarray:
        """Return zero embedding as fallback"""
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def process_and_embed_codebase(self, zip_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a codebase ZIP file using repomix and create an embedding.
        
        Args:
            zip_path: Path to the ZIP file containing the codebase
            
        Returns:
            Tuple of (embedding vector, metadata dictionary)
            
        Raises:
            Exception: If processing or embedding fails
        """
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Process with repomix
                repomix_result = self.repomix_processor.process_codebase(zip_path, temp_path)
                
                # Create embedding from processed content
                embedding = self._generate_embedding(repomix_result['content'])
                
                # Extract student name
                student_name = self._extract_student_name(zip_path)
                
                # Prepare metadata
                metadata = {
                    'student_name': student_name,
                    'token_count': repomix_result['token_count'],
                    'compressed': repomix_result['compressed'],
                    'within_limit': repomix_result['within_limit'],
                    'project_name': repomix_result['project_name'],
                    'filtered_files_count': repomix_result['filtered_files_count']
                }
                
                return embedding, metadata
                
        except Exception as e:
            self.logger.error(f"Failed to process and embed codebase {zip_path}: {str(e)}")
            raise
    
    def process_and_embed_batch(self, zip_paths: List[Path]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process and embed multiple codebase ZIP files.
        
        Args:
            zip_paths: List of paths to ZIP files
            
        Returns:
            List of (embedding vector, metadata) tuples
            
        Raises:
            Exception: If processing or embedding fails
        """
        results = []
        
        for i, zip_path in enumerate(zip_paths):
            self.logger.info(f"Processing codebase {i+1}/{len(zip_paths)}: {zip_path}")
            try:
                result = self.process_and_embed_codebase(zip_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path}: {str(e)}")
                # Add zero embedding as fallback
                student_name = self._extract_student_name(zip_path)
                results.append((self._get_zero_embedding(), {
                    'student_name': student_name,
                    'error': str(e),
                    'zip_path': str(zip_path)
                }))
        
        return results
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedder configuration"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'ollama_base_url': self.ollama_base_url,
            'max_tokens': self.repomix_processor.max_tokens,
            'use_compression': self.repomix_processor.use_compression,
            'remove_comments': self.repomix_processor.remove_comments,
            'ignore_patterns': self.repomix_processor.ignore_patterns,
            'keep_patterns': self.repomix_processor.keep_patterns,
            'max_file_size': self.repomix_processor.max_file_size
        } 