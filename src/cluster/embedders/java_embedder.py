"""
Java Code Embedder

Semantic embeddings for Java code using StarCoder2 models via Ollama.
Focuses on understanding Java code structure and semantics.

Author: Auto-generated
"""

import requests
import json
import re
from typing import Dict, Any
import numpy as np

from .base_embedder import BaseEmbedder, EmbedderConfig


class JavaEmbedder(BaseEmbedder):
    """
    Java code embedder using StarCoder2 models via Ollama.
    Specializes in understanding Java code structure and semantics.
    """
    
    def __init__(self, config: EmbedderConfig = None):
        """Initialize the Java embedder"""
        super().__init__(config)
        self._test_ollama_connection()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on StarCoder2 model"""
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
        Prepare Java code content for embedding.
        
        Args:
            submission_data: Dictionary containing submission data with 'java_files' key
            
        Returns:
            Processed Java code string
        """
        java_files = submission_data.get('java_files', {})
        
        if not java_files:
            self.logger.warning(f"No Java files found for {submission_data.get('student_name', 'unknown')}")
            return "// No Java code found"
        
        # Process and combine Java files
        code_parts = []
        
        for filename, content in java_files.items():
            # Clean the Java code
            cleaned_content = self._clean_java_code(content)
            
            # Add file marker and content
            code_parts.append(f"// FILE: {filename}\n{cleaned_content}")
        
        # Combine all files
        combined_code = "\n\n// ===== NEXT FILE =====\n\n".join(code_parts)
        
        # Truncate if too long
        if len(combined_code) > self.config.max_length * 4:  # Rough character estimate
            self.logger.debug(f"Code too long for {submission_data.get('student_name', 'unknown')}, truncating...")
            combined_code = self._smart_truncate(combined_code)
        
        return combined_code
    
    def generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate embedding using Ollama API.
        
        Args:
            content: Java code content to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Create a prompt that encourages semantic understanding
            prompt = f"""Analyze this Java code and understand its semantic meaning and structure:

{content}

Focus on the programming concepts, algorithms, design patterns, and overall code structure."""
            
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
    
    def _clean_java_code(self, content: str) -> str:
        """Clean Java code content while preserving structure"""
        # Remove excessive whitespace while preserving structure
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Skip completely empty lines in sequence
            if not line.strip():
                if not cleaned_lines or cleaned_lines[-1].strip():
                    cleaned_lines.append('')
            else:
                cleaned_lines.append(line)
        
        # Remove excessive blank lines
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result
    
    def _smart_truncate(self, code: str) -> str:
        """Smart truncation that preserves Java code structure"""
        target_length = self.config.max_length * 3  # Conservative estimate
        
        if len(code) <= target_length:
            return code
        
        # Try to truncate at natural Java boundaries
        boundaries = [
            '\n\n// ===== NEXT FILE =====\n\n',  # File boundaries
            '\n    }\n}',  # End of class
            '\n    }',     # End of method
            '\n}',         # Any closing brace
            '\n'           # Line boundary
        ]
        
        for boundary in boundaries:
            last_boundary = code.rfind(boundary, 0, target_length)
            if last_boundary > target_length * 0.7:  # At least 70% of target
                return code[:last_boundary + len(boundary)]
        
        # Fallback: truncate at word boundary
        truncated = code[:target_length]
        last_space = truncated.rfind(' ')
        if last_space > target_length * 0.9:
            truncated = truncated[:last_space]
        
        return truncated + "\n// ... (code truncated)"
    
    def calculate_semantic_similarity(self, java_files1: Dict[str, str], 
                                    java_files2: Dict[str, str]) -> float:
        """
        Calculate semantic similarity between two Java codebases.
        
        Args:
            java_files1: First codebase Java files
            java_files2: Second codebase Java files
            
        Returns:
            Semantic similarity score between 0.0 and 1.0
        """
        # Create submission data format
        submission1 = {'java_files': java_files1, 'student_name': 'temp1'}
        submission2 = {'java_files': java_files2, 'student_name': 'temp2'}
        
        # Generate embeddings
        embedding1 = self.embed_submission(submission1)
        embedding2 = self.embed_submission(submission2)
        
        # Calculate similarity
        return self.calculate_similarity(embedding1, embedding2)
    
    def extract_java_features(self, java_content: str) -> Dict[str, Any]:
        """
        Extract high-level features from Java code for additional analysis.
        
        Args:
            java_content: Java code content
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'num_classes': len(re.findall(r'\bclass\s+\w+', java_content)),
            'num_interfaces': len(re.findall(r'\binterface\s+\w+', java_content)),
            'num_methods': len(re.findall(r'\b(public|private|protected|static|\s)+\w+\s+\w+\s*\([^)]*\)\s*\{', java_content)),
            'num_imports': len(re.findall(r'^\s*import\s+', java_content, re.MULTILINE)),
            'lines_of_code': len([line for line in java_content.split('\n') if line.strip()]),
            'cyclomatic_complexity': self._estimate_complexity(java_content),
            'design_patterns': self._detect_patterns(java_content)
        }
        
        return features
    
    def _estimate_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity of Java code"""
        complexity = 1  # Base complexity
        
        # Decision points
        complexity += len(re.findall(r'\bif\s*\(', content))
        complexity += len(re.findall(r'\belse\b', content))
        complexity += len(re.findall(r'\bwhile\s*\(', content))
        complexity += len(re.findall(r'\bfor\s*\(', content))
        complexity += len(re.findall(r'\bcase\s+', content))
        complexity += len(re.findall(r'\bcatch\s*\(', content))
        complexity += len(re.findall(r'\b&&\b', content))
        complexity += len(re.findall(r'\b\|\|\b', content))
        
        return complexity
    
    def _detect_patterns(self, content: str) -> Dict[str, int]:
        """Detect common design patterns in Java code"""
        patterns = {
            'singleton': len(re.findall(r'private\s+static\s+.*\s+instance', content)),
            'factory': len(re.findall(r'create\w*\(', content)),
            'builder': len(re.findall(r'\.build\(\)', content)),
            'observer': len(re.findall(r'(addListener|addEventListener|notify)', content)),
            'strategy': len(re.findall(r'Strategy\b', content)),
            'decorator': len(re.findall(r'Decorator\b', content)),
            'adapter': len(re.findall(r'Adapter\b', content))
        }
        
        return patterns 