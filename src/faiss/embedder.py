"""
Java Code Embedder using StarCoder2 or Ollama

A clean, simple code embedder focused on Java programming language using StarCoder2 model
or Ollama API endpoints. Removes all structural/dependency analysis and focuses purely 
on semantic code embeddings.

Author: Auto-generated
"""

import numpy as np
import torch
import os
import requests
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import re
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

from ..utils.logger import get_logger

# Import the enhanced AST analyzer
try:
    from .analyzers.ast_java_analyzer import ASTJavaAnalyzer
    AST_ANALYZER_AVAILABLE = True
except ImportError:
    AST_ANALYZER_AVAILABLE = False


@dataclass
class EmbedderConfig:
    """Configuration for the Java code embedder"""
    model_name: str = "starcoder2:3b"  # Default to Ollama StarCoder2
    use_ollama: bool = True  # Use Ollama API by default
    ollama_base_url: str = "http://localhost:11434"  # Default Ollama URL
    device: Optional[str] = None  # Auto-detect if None (for transformers)
    max_length: int = 8192  # Maximum sequence length (Ollama models support more)
    java_only: bool = True  # Only process Java files
    normalize_embeddings: bool = True  # L2 normalize embeddings
    batch_size: int = 1  # Batch size for processing


class OllamaJavaCodeEmbedder:
    """Fast Java code embedder using Ollama API"""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """
        Initialize the Ollama Java code embedder.
        
        Args:
            config: Embedder configuration (uses default if None)
        """
        self.config = config or EmbedderConfig()
        self.logger = get_logger(__name__)
        
        # Load from environment if available
        if 'OLLAMA_BASE_URL' in os.environ:
            self.config.ollama_base_url = os.environ['OLLAMA_BASE_URL']
        
        # Set embedding dimension based on model
        self.embedding_dim = self._get_embedding_dimension()
        
        # Initialize AST analyzer for enhanced code analysis
        if AST_ANALYZER_AVAILABLE:
            self.ast_analyzer = ASTJavaAnalyzer()
            self.logger.info("AST-based code analysis enabled")
        else:
            self.ast_analyzer = None
            self.logger.warning("AST analyzer not available - using basic analysis")
        
        # Test connection
        self._test_ollama_connection()
        
        self.logger.info(f"Ollama Java code embedder initialized with {self.config.model_name}")
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model"""
        model_dimensions = {
            'starcoder2:3b': 2048,
            'starcoder2:7b': 4096, 
            'starcoder2:15b': 6144,
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
        
        # Check for partial matches (e.g., "starcoder2" in model name)
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

    def embed_codebase(self, code_files: Dict[str, str]) -> np.ndarray:
        """
        Generate embeddings for Java codebase using Ollama with enhanced AST analysis.
        
        Args:
            code_files: Dictionary mapping filenames to content
            
        Returns:
            Numpy array containing the enhanced codebase embedding
        """
        # Filter to Java files only
        java_files = self._filter_java_files(code_files)
        
        if not java_files:
            self.logger.warning("No Java files found in codebase")
            return self._get_zero_embedding()
        
        # Enhanced code preparation with AST analysis
        processed_code = self._prepare_java_code_enhanced(java_files)
        
        # Generate embedding using Ollama
        embedding = self._generate_ollama_embedding(processed_code)
        
        return embedding
    
    def _filter_java_files(self, code_files: Dict[str, str]) -> Dict[str, str]:
        """Filter dictionary to only include Java files"""
        java_files = {}
        
        for filename, content in code_files.items():
            if filename.lower().endswith('.java'):
                # Skip empty files
                if content.strip():
                    java_files[filename] = content
        
        self.logger.debug(f"Filtered to {len(java_files)} Java files from {len(code_files)} total files")
        return java_files
    
    def _prepare_java_code(self, java_files: Dict[str, str]) -> str:
        """
        Prepare Java code for embedding by cleaning and combining files.
        
        Args:
            java_files: Dictionary of Java files
            
        Returns:
            Processed code string ready for embedding
        """
        code_parts = []
            
        for filename, content in java_files.items():
            # Clean the code content
            cleaned_content = self._clean_java_code(content)
            
            # Add file marker and content
            code_parts.append(f"// FILE: {filename}\n{cleaned_content}")
        
        # Combine all files
        combined_code = "\n\n// ===== NEXT FILE =====\n\n".join(code_parts)
        
        # Truncate if too long (Ollama models support longer context)
        if len(combined_code) > self.config.max_length * 4:  # Rough character estimate
            self.logger.debug("Code too long, truncating...")
            combined_code = self._smart_truncate(combined_code)
        
        return combined_code
    
    def _prepare_java_code_enhanced(self, java_files: Dict[str, str]) -> str:
        """
        Enhanced Java code preparation with AST analysis and structural insights.
        
        Args:
            java_files: Dictionary of Java files
            
        Returns:
            Enhanced processed code string with structural context
        """
        if not self.ast_analyzer:
            # Fallback to regular preparation if AST analyzer unavailable
            return self._prepare_java_code(java_files)
        
        # Analyze all files for structural insights
        structural_summary = self._extract_structural_summary(java_files)
        
        code_parts = []
        
        # Add structural summary at the beginning
        if structural_summary:
            code_parts.append(f"// CODEBASE STRUCTURAL SUMMARY:\n{structural_summary}\n")
        
        # Process files with enhanced context
        for filename, content in java_files.items():
            try:
                # Perform AST analysis on this file
                file_analysis = self.ast_analyzer.analyze_file(content, filename)
                
                # Create enhanced file header with structural info
                file_header = self._create_enhanced_file_header(filename, file_analysis)
                
                # Clean the code content
                cleaned_content = self._clean_java_code(content)
                
                # Combine enhanced header with code
                enhanced_file = f"{file_header}\n{cleaned_content}"
                code_parts.append(enhanced_file)
                
            except Exception as e:
                self.logger.warning(f"AST analysis failed for {filename}: {e}")
                # Fallback to regular processing for this file
                cleaned_content = self._clean_java_code(content)
                code_parts.append(f"// FILE: {filename}\n{cleaned_content}")
        
        # Combine all parts
        combined_code = "\n\n// ===== NEXT FILE =====\n\n".join(code_parts)
        
        # Truncate if too long (AST info might make it longer)
        if len(combined_code) > self.config.max_length * 4:
            self.logger.debug("Enhanced code too long, truncating...")
            combined_code = self._smart_truncate(combined_code)
        
        return combined_code
    
    def _extract_structural_summary(self, java_files: Dict[str, str]) -> str:
        """Extract high-level structural summary of the codebase"""
        if not self.ast_analyzer:
            return ""
        
        total_classes = 0
        total_methods = 0
        total_complexity = 0
        design_patterns = set()
        key_apis = set()
        
        for filename, content in java_files.items():
            try:
                analysis = self.ast_analyzer.analyze_file(content, filename)
                
                # Aggregate metrics
                total_classes += analysis.get('class_analysis', {}).get('total_classes', 0)
                total_methods += analysis.get('method_analysis', {}).get('total_methods', 0)
                total_complexity += analysis.get('complexity_indicators', {}).get('cyclomatic_complexity', 0)
                
                # Collect design patterns
                patterns = analysis.get('design_patterns', {})
                for pattern, count in patterns.items():
                    if count > 0:
                        design_patterns.add(pattern)
                
                # Collect key API calls
                api_calls = analysis.get('unique_elements', {}).get('api_calls', [])
                key_apis.update(api_calls[:5])  # Top 5 API calls per file
                
            except Exception:
                continue
        
        summary_parts = []
        summary_parts.append(f"Classes: {total_classes}, Methods: {total_methods}")
        summary_parts.append(f"Total Complexity: {total_complexity}")
        
        if design_patterns:
            summary_parts.append(f"Design Patterns: {', '.join(list(design_patterns)[:3])}")
        
        if key_apis:
            summary_parts.append(f"Key APIs: {', '.join(list(key_apis)[:5])}")
        
        return " | ".join(summary_parts)
    
    def _create_enhanced_file_header(self, filename: str, file_analysis: Dict[str, Any]) -> str:
        """Create enhanced file header with structural metadata"""
        header_parts = [f"// FILE: {filename}"]
        
        # Add structural fingerprints
        fingerprints = file_analysis.get('fingerprints', {})
        if fingerprints:
            header_parts.append(f"// STRUCTURAL_ID: {fingerprints.get('combined_hash', 'unknown')}")
        
        # Add key metrics
        class_info = file_analysis.get('class_analysis', {})
        method_info = file_analysis.get('method_analysis', {})
        
        if class_info.get('total_classes', 0) > 0:
            header_parts.append(f"// CLASSES: {class_info['total_classes']}")
        
        if method_info.get('total_methods', 0) > 0:
            header_parts.append(f"// METHODS: {method_info['total_methods']}")
        
        # Add complexity indicator
        complexity = file_analysis.get('complexity_indicators', {}).get('cyclomatic_complexity', 0)
        if complexity > 0:
            header_parts.append(f"// COMPLEXITY: {complexity}")
        
        # Add design patterns if present
        patterns = file_analysis.get('design_patterns', {})
        active_patterns = [p for p, c in patterns.items() if c > 0]
        if active_patterns:
            header_parts.append(f"// PATTERNS: {', '.join(active_patterns[:2])}")
        
        return "\n".join(header_parts)
    
    def _clean_java_code(self, content: str) -> str:
        """Clean Java code content"""
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
    
    def _generate_ollama_embedding(self, code: str) -> np.ndarray:
        """Generate embedding using Ollama API"""
        try:
            # Create a prompt that encourages semantic understanding
            prompt = f"""Please analyze this Java code and understand its semantic meaning:

{code}

This code represents a complete Java implementation."""
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.config.ollama_base_url}/api/embeddings",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt
                },
                timeout=60  # Increased timeout for larger models
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['embedding'], dtype=np.float32)
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
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
    
    def _get_zero_embedding(self) -> np.ndarray:
        """Return zero embedding as fallback"""
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_batch(self, code_files_list: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple codebases.
        
        Args:
            code_files_list: List of codebase dictionaries
            
        Returns:
            List of embedding arrays
        """
        embeddings = []
        
        for i, code_files in enumerate(code_files_list):
            self.logger.debug(f"Processing codebase {i+1}/{len(code_files_list)}")
            embedding = self.embed_codebase(code_files)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_info(self) -> Dict[str, any]:
        """Get information about the embedder configuration"""
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'use_ollama': True,
            'ollama_base_url': self.config.ollama_base_url,
            'max_length': self.config.max_length,
            'java_only': self.config.java_only,
            'normalize_embeddings': self.config.normalize_embeddings
        }
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate enhanced cosine similarity between two embeddings with AST-aware enhancements.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Enhanced similarity score between 0.0 and 1.0
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Clamp to [0, 1] range to handle floating point precision errors
        similarity = np.clip(similarity, 0.0, 1.0)
        
        # Enhanced penalty function that better spreads out similarities
        # Use a more aggressive power function to create more distinction
        similarity = similarity ** 4  # More aggressive penalty: 0.99^4 ≈ 0.96, 0.95^4 ≈ 0.81
        
        # Add controlled noise for very high similarities to avoid clustering
        if similarity > 0.9:
            noise = np.random.normal(0, 0.015)  # Reduced noise for AST-enhanced embeddings
            similarity = np.clip(similarity + noise, 0.0, 1.0)
        
        return float(similarity)
    
    def calculate_structural_similarity(self, code_files1: Dict[str, str], 
                                      code_files2: Dict[str, str]) -> float:
        """
        Calculate structural similarity using AST fingerprints (complementary to embedding similarity).
        
        Args:
            code_files1: First codebase files
            code_files2: Second codebase files
            
        Returns:
            Structural similarity score between 0.0 and 1.0
        """
        if not self.ast_analyzer:
            return 0.0  # No structural analysis available
        
        try:
            # Analyze both codebases
            fingerprints1 = self._extract_codebase_fingerprints(code_files1)
            fingerprints2 = self._extract_codebase_fingerprints(code_files2)
            
            # Compare structural elements
            structural_sim = self._compare_structural_fingerprints(fingerprints1, fingerprints2)
            
            return structural_sim
            
        except Exception as e:
            self.logger.warning(f"Structural similarity calculation failed: {e}")
            return 0.0
    
    def _extract_codebase_fingerprints(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Extract structural fingerprints from a codebase"""
        all_class_names = set()
        all_method_names = set()
        all_api_calls = set()
        all_fingerprints = []
        total_complexity = 0
        design_patterns = set()
        
        for filename, content in code_files.items():
            if not filename.lower().endswith('.java'):
                continue
                
            try:
                analysis = self.ast_analyzer.analyze_file(content, filename)
                
                # Collect unique elements
                unique_elements = analysis.get('unique_elements', {})
                all_class_names.update(unique_elements.get('class_names', []))
                all_method_names.update(unique_elements.get('method_names', []))
                all_api_calls.update(unique_elements.get('api_calls', []))
                
                # Collect fingerprints
                fingerprints = analysis.get('fingerprints', {})
                if 'combined_hash' in fingerprints:
                    all_fingerprints.append(fingerprints['combined_hash'])
                
                # Accumulate complexity
                total_complexity += analysis.get('complexity_indicators', {}).get('cyclomatic_complexity', 0)
                
                # Collect design patterns
                patterns = analysis.get('design_patterns', {})
                for pattern, count in patterns.items():
                    if count > 0:
                        design_patterns.add(pattern)
                        
            except Exception:
                continue
        
        return {
            'class_names': all_class_names,
            'method_names': all_method_names,
            'api_calls': all_api_calls,
            'fingerprints': all_fingerprints,
            'total_complexity': total_complexity,
            'design_patterns': design_patterns
        }
    
    def _compare_structural_fingerprints(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """Compare structural fingerprints to calculate similarity"""
        similarities = []
        
        # Class name similarity
        classes1, classes2 = fp1['class_names'], fp2['class_names']
        if classes1 or classes2:
            class_sim = len(classes1 & classes2) / max(len(classes1 | classes2), 1)
            similarities.append(class_sim * 0.3)  # 30% weight
        
        # Method name similarity
        methods1, methods2 = fp1['method_names'], fp2['method_names']
        if methods1 or methods2:
            method_sim = len(methods1 & methods2) / max(len(methods1 | methods2), 1)
            similarities.append(method_sim * 0.25)  # 25% weight
        
        # API usage similarity
        apis1, apis2 = fp1['api_calls'], fp2['api_calls']
        if apis1 or apis2:
            api_sim = len(apis1 & apis2) / max(len(apis1 | apis2), 1)
            similarities.append(api_sim * 0.25)  # 25% weight
        
        # Design pattern similarity
        patterns1, patterns2 = fp1['design_patterns'], fp2['design_patterns']
        if patterns1 or patterns2:
            pattern_sim = len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)
            similarities.append(pattern_sim * 0.2)  # 20% weight
        
        return sum(similarities) if similarities else 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        candidate_embeddings: List[np.ndarray]) -> List[float]:
        """
        Calculate similarities between query and multiple candidates.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for candidate in candidate_embeddings:
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append(similarity)
        
        return similarities


# Factory function for easy initialization with Ollama support
def create_java_embedder(model_name: str = "starcoder2:3b",
                        use_ollama: bool = True,
                        ollama_base_url: str = None,
                        device: Optional[str] = None,
                        max_length: int = 8192) -> OllamaJavaCodeEmbedder:
    """
    Factory function to create a Java code embedder with custom configuration.
    
    Args:
        model_name: Model name (Ollama format like "starcoder2:3b")
        use_ollama: Whether to use Ollama API (default: True)
        ollama_base_url: Ollama base URL (loads from env if None)
        device: Device to use for transformers (auto-detect if None)
        max_length: Maximum sequence length
            
    Returns:
        Configured Java code embedder instance (always OllamaJavaCodeEmbedder now)
    """
    # Load from environment
    if ollama_base_url is None and 'OLLAMA_BASE_URL' in os.environ:
        ollama_base_url = os.environ['OLLAMA_BASE_URL']
    
    config = EmbedderConfig(
        model_name=model_name,
        use_ollama=use_ollama,
        ollama_base_url=ollama_base_url or "http://localhost:11434",
        device=device,
        max_length=max_length
    )
    
    # Always return OllamaJavaCodeEmbedder since JavaCodeEmbedder was removed
    return OllamaJavaCodeEmbedder(config)


# Backward compatibility aliases
HybridCodeEmbedder = OllamaJavaCodeEmbedder
ImprovedCodeEmbedder = OllamaJavaCodeEmbedder
JavaCodeEmbedder = OllamaJavaCodeEmbedder  # For backward compatibility 