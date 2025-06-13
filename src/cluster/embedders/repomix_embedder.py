"""
Repomix Code Embedder

Processes entire codebases using repomix and generates embeddings.
Good for holistic codebase analysis across multiple file types.

Author: Auto-generated
"""

import subprocess
import tempfile
import zipfile
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field

from .base_embedder import BaseEmbedder, EmbedderConfig


@dataclass
class RepomixConfig(EmbedderConfig):
    """Extended configuration for Repomix embedder"""
    max_tokens: int = 128000
    use_compression: bool = True
    remove_comments: bool = False
    ignore_patterns: Optional[List[str]] = field(default=None)
    keep_patterns: Optional[List[str]] = field(default=None)
    max_file_size: int = 1000000  # 1MB default
    
    def __post_init__(self):
        super().__post_init__()
        if self.ignore_patterns is None:
            self.ignore_patterns = ["*.pyc", "__pycache__", ".git", "*.log", "node_modules"]
        if self.keep_patterns is None:
            self.keep_patterns = ["*.java", "*.py", "*.js", "*.ts", "*.cpp", "*.h", "*.md", "*.txt"]


class RepomixEmbedder(BaseEmbedder):
    """
    Repomix embedder that processes entire codebases using repomix tool.
    Handles multiple file types and provides holistic codebase analysis.
    """
    
    def __init__(self, config: RepomixConfig = None):
        """Initialize the Repomix embedder"""
        self.repomix_config = config or RepomixConfig()
        super().__init__(self.repomix_config)
        self._check_repomix_availability()
        self._test_ollama_connection()
    
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
        if self.config.model_name in model_dimensions:
            return model_dimensions[self.config.model_name]
        
        # Check for partial matches
        for model_key, dim in model_dimensions.items():
            if model_key.split(':')[0] in self.config.model_name:
                return dim
        
        # Default fallback
        self.logger.warning(f"Unknown model {self.config.model_name}, using default dimension 2048")
        return 2048
    
    def _check_repomix_availability(self):
        """Check if repomix is available via npx"""
        try:
            result = subprocess.run(
                ['npx', 'repomix', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.logger.info(f"Repomix available: {result.stdout.strip()}")
            else:
                self.logger.warning("Repomix not found, will attempt to install on first use")
        except Exception as e:
            self.logger.warning(f"Could not check repomix availability: {str(e)}")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check if our model is available
                if not any(self.config.model_name in name for name in model_names):
                    self.logger.warning(f"Model {self.config.model_name} not found in Ollama. Available: {model_names}")
                    self.logger.info(f"To pull the model, run: ollama pull {self.config.model_name}")
                else:
                    self.logger.info(f"Model {self.config.model_name} available in Ollama")
            else:
                raise requests.RequestException(f"HTTP {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"Cannot connect to Ollama at {self.config.ollama_base_url}: {e}")
            self.logger.info("Make sure Ollama is running and accessible")
    
    def _prepare_content(self, submission_data: Dict[str, Any]) -> str:
        """
        Prepare codebase content using repomix processing.
        
        Args:
            submission_data: Dictionary containing submission data with 'zip_path' key
            
        Returns:
            Processed codebase content string
        """
        zip_path = submission_data.get('zip_path')
        if not zip_path:
            self.logger.error(f"No zip_path found for {submission_data.get('student_name', 'unknown')}")
            return "// No codebase found"
        
        try:
            # Process with repomix
            processed_content = self._process_with_repomix(Path(zip_path))
            return processed_content
        except Exception as e:
            self.logger.error(f"Failed to process codebase with repomix: {e}")
            return f"// Error processing codebase: {str(e)}"
    
    def generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate embedding using Ollama API.
        
        Args:
            content: Processed codebase content to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Create a prompt that encourages holistic understanding
            prompt = f"""Analyze this entire codebase and understand its overall structure and implementation:

{content}

Focus on the complete system architecture, file organization, module interactions, and overall design approach."""
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.config.ollama_base_url}/api/embeddings",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['embedding'], dtype=np.float32)
                
                self.logger.debug(f"Generated embedding with shape: {embedding.shape}")
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
    
    def _process_with_repomix(self, zip_path: Path) -> str:
        """
        Process a ZIP file using repomix.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Processed content string
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract ZIP file
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the main project directory (skip __MACOSX and similar)
            project_dirs = [d for d in extract_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('__')]
            
            if not project_dirs:
                # No subdirectories, use extract_dir directly
                project_dir = extract_dir
            elif len(project_dirs) == 1:
                # Single project directory
                project_dir = project_dirs[0]
            else:
                # Multiple directories, use extract_dir
                project_dir = extract_dir
            
            # Run repomix on the project directory
            output_file = temp_path / "repomix_output.txt"
            self._run_repomix(project_dir, output_file)
            
            # Read the processed content
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content
            else:
                raise RuntimeError("Repomix did not generate output file")
    
    def _run_repomix(self, project_dir: Path, output_file: Path) -> None:
        """
        Run repomix command on the project directory.
        
        Args:
            project_dir: Directory containing the project files
            output_file: Output file for repomix
        """
        try:
            # Build repomix command
            cmd = ['npx', 'repomix', str(project_dir), '--output', str(output_file)]
            
            # Add compression options
            if self.repomix_config.use_compression:
                cmd.append('--compress')
                cmd.append('--remove-empty-lines')
            
            if self.repomix_config.remove_comments:
                cmd.append('--remove-comments')
            
            # Add token counting with encoding
            cmd.extend(['--token-count-encoding', 'cl100k_base'])
            
            # Add ignore patterns
            if self.repomix_config.ignore_patterns:
                ignore_patterns_str = ','.join(self.repomix_config.ignore_patterns)
                cmd.extend(['--ignore', ignore_patterns_str])
            
            # Add keep patterns (include patterns)
            if self.repomix_config.keep_patterns:
                include_patterns_str = ','.join(self.repomix_config.keep_patterns)
                cmd.extend(['--include', include_patterns_str])
            
            # Run repomix
            self.logger.debug(f"Running repomix command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Repomix failed: {result.stderr}")
                raise RuntimeError(f"Repomix execution failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("Repomix execution timed out")
            raise RuntimeError("Repomix execution timed out")
        except Exception as e:
            self.logger.error(f"Error running repomix: {str(e)}")
            raise
    
    def get_repomix_metadata(self, zip_path: Path) -> Dict[str, Any]:
        """
        Get metadata about repomix processing of a codebase.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Dictionary containing processing metadata
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract and process
                extract_dir = temp_path / "extracted"
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Count files and estimate metrics
                total_files = 0
                code_files = 0
                total_size = 0
                
                for root, dirs, files in extract_dir.rglob('*'):
                    if root.is_file():
                        total_files += 1
                        total_size += root.stat().st_size
                        
                        if any(root.suffix in ext for ext in self.repomix_config.keep_patterns):
                            code_files += 1
                
                return {
                    'total_files': total_files,
                    'code_files': code_files,
                    'total_size_bytes': total_size,
                    'estimated_tokens': total_size // 4,  # Rough estimate
                    'zip_name': zip_path.name
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get repomix metadata: {e}")
            return {
                'error': str(e),
                'zip_name': zip_path.name
            }
    
    def calculate_codebase_similarity(self, zip_path1: Path, zip_path2: Path) -> float:
        """
        Calculate similarity between two codebases.
        
        Args:
            zip_path1: First codebase ZIP
            zip_path2: Second codebase ZIP
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Create submission data format
        submission1 = {'zip_path': str(zip_path1), 'student_name': 'temp1'}
        submission2 = {'zip_path': str(zip_path2), 'student_name': 'temp2'}
        
        # Generate embeddings
        embedding1 = self.embed_submission(submission1)
        embedding2 = self.embed_submission(submission2)
        
        # Calculate similarity
        return self.calculate_similarity(embedding1, embedding2) 