"""
Hybrid Code Embedding System

This module provides multi-layer code embeddings that combine:
1. Semantic embeddings (CodeBERT-based)
2. Structural metrics (AST-based)
3. Dependency graph features (NetworkX-based)

Author: Auto-generated
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

# Import multi-language analyzers
try:
    from .analyzers import MultiLanguageCodeAnalyzer, MultiLanguageDependencyGraphBuilder
    MULTI_LANG_AVAILABLE = True
except ImportError:
    # Fallback to original analyzer
    try:
        from .analyzer import CodeStructureAnalyzer, DependencyGraphBuilder
        MULTI_LANG_AVAILABLE = False
    except ImportError:
        # Neither available
        MULTI_LANG_AVAILABLE = None

from ..utils.logger import get_logger


class HybridCodeEmbedder:
    """Hybrid embedding system combining multiple approaches"""
    
    def __init__(self, model_name: str = 'microsoft/codebert-base', device: Optional[str] = None):
        """
        Initialize the hybrid code embedder.
        
        Args:
            model_name: Name of the code embedding model to use
            device: Device to use for computation ('cuda', 'cpu', or None for auto)
        """
        self.logger = get_logger(__name__)
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model
        self.model_name = model_name
        self.code_model = None
        self.tokenizer = None
        
        self._initialize_models()
        
        # Initialize analyzers
        if MULTI_LANG_AVAILABLE == True:
            self.structure_analyzer = MultiLanguageCodeAnalyzer()
            self.graph_builder = MultiLanguageDependencyGraphBuilder()
            self.logger.info("Using multi-language code analyzers")
        elif MULTI_LANG_AVAILABLE == False:
            self.structure_analyzer = CodeStructureAnalyzer()
            self.graph_builder = DependencyGraphBuilder()
            self.logger.info("Using single-language code analyzers (fallback)")
        else:
            # Neither available - create minimal analyzers
            self.structure_analyzer = None
            self.graph_builder = None
            self.logger.warning("No code analyzers available - using minimal analysis")
        
        # Scalers for normalization (will be fitted during training)
        self.structure_scaler = StandardScaler()
        self.graph_scaler = StandardScaler()
        self.is_fitted = False
        
        # Embedding dimensions
        self.code_embedding_dim = 768  # Standard for CodeBERT
        self.structure_dim = 0  # Will be determined from first analysis
        self.graph_dim = 20 if MULTI_LANG_AVAILABLE else 15  # Multi-language has more features
        
        self.logger.info("Hybrid code embedder initialized successfully")
    
    def _initialize_models(self):
        """Initialize the code embedding models"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.code_model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info("SentenceTransformer model loaded successfully")
            
            elif TRANSFORMERS_AVAILABLE:
                self.logger.info(f"Loading Transformers model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.code_model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.code_model.eval()
                self.logger.info("Transformers model loaded successfully")
            
            else:
                self.logger.warning("No embedding models available. Using random embeddings for testing.")
                self.code_model = None
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            self.logger.warning("Falling back to random embeddings")
            self.code_model = None
    
    def embed_codebase(self, code_files: Dict[str, str]) -> np.ndarray:
        """
        Generate hybrid embedding for entire codebase.
        
        Args:
            code_files: Dictionary mapping filenames to content
            
        Returns:
            Combined embedding vector as numpy array
        """
        # 1. Generate semantic embeddings
        code_embedding = self._generate_semantic_embedding(code_files)
        
        # 2. Extract structural features
        structure_features = self._extract_structural_features(code_files)
        
        # 3. Extract dependency graph features
        graph_features = self._extract_graph_features(code_files)
        
        # Set dimensions on first embedding
        if self.structure_dim == 0:
            self.structure_dim = len(structure_features)
        
        # Combine all embeddings
        combined_embedding = np.concatenate([
            code_embedding,      # Semantic understanding
            structure_features,  # Structural metrics  
            graph_features      # Graph/dependency features
        ])
        
        return combined_embedding.astype(np.float32)
    
    def _generate_semantic_embedding(self, code_files: Dict[str, str]) -> np.ndarray:
        """Generate semantic embedding using CodeBERT or similar model"""
        try:
            # Filter and combine code files with language detection
            if MULTI_LANG_AVAILABLE and hasattr(self.structure_analyzer, 'get_supported_extensions'):
                supported_exts = self.structure_analyzer.get_supported_extensions()
                relevant_files = {fname: content for fname, content in code_files.items() 
                                if any(fname.endswith(ext) for ext in supported_exts)}
            else:
                # Fallback: include common programming files
                relevant_files = {fname: content for fname, content in code_files.items() 
                                if any(fname.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.c', '.h', '.cs'])}
            
            if not relevant_files:
                relevant_files = code_files  # Use all files if no programming files found
            
            # Combine all code files with language-aware formatting
            all_code_parts = []
            for filename, content in relevant_files.items():
                # Add filename as context for better semantic understanding
                lang_hint = ""
                if MULTI_LANG_AVAILABLE and hasattr(self.structure_analyzer, 'detect_language'):
                    detected_lang = self.structure_analyzer.detect_language(filename)
                    if detected_lang:
                        lang_hint = f" ({detected_lang})"
                
                all_code_parts.append(f"# File: {filename}{lang_hint}\n{content}")
            
            all_code = "\n\n".join(all_code_parts)
            
            # Truncate if too long (most models have token limits)
            if len(all_code) > 15000:  # Conservative limit
                all_code = all_code[:15000] + "\n# ... (truncated)"
            
            if self.code_model is None:
                # Return random embedding for testing when models not available
                return np.random.normal(0, 0.1, self.code_embedding_dim).astype(np.float32)
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self.code_model, SentenceTransformer):
                # Use SentenceTransformer
                embedding = self.code_model.encode([all_code], convert_to_numpy=True)[0]
                return embedding.astype(np.float32)
            
            elif TRANSFORMERS_AVAILABLE and self.tokenizer is not None:
                # Use raw transformers
                with torch.no_grad():
                    inputs = self.tokenizer(all_code, return_tensors='pt', 
                                          max_length=512, truncation=True, padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.code_model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    return embedding.astype(np.float32)
            
            else:
                # Fallback: random embedding
                return np.random.normal(0, 0.1, self.code_embedding_dim).astype(np.float32)
                
        except Exception as e:
            self.logger.warning(f"Error generating semantic embedding: {str(e)}")
            return np.random.normal(0, 0.1, self.code_embedding_dim).astype(np.float32)
    
    def _extract_structural_features(self, code_files: Dict[str, str]) -> np.ndarray:
        """Extract structural features from codebase"""
        try:
            if self.structure_analyzer is None:
                # Return minimal features when no analyzer available
                return np.zeros(10, dtype=np.float32)
                
            aggregated_metrics = self.structure_analyzer.analyze_codebase(code_files)
            # Convert to array, ensuring consistent ordering
            feature_keys = sorted(aggregated_metrics.keys())
            features = np.array([aggregated_metrics[key] for key in feature_keys], dtype=np.float32)
            return features
        except Exception as e:
            self.logger.warning(f"Error extracting structural features: {str(e)}")
            # Return zero features if analysis fails
            return np.zeros(50, dtype=np.float32)  # Reasonable default size
    
    def _extract_graph_features(self, code_files: Dict[str, str]) -> np.ndarray:
        """Extract dependency graph features from codebase"""
        try:
            if self.graph_builder is None:
                # Return minimal graph features when no builder available
                return np.zeros(self.graph_dim, dtype=np.float32)
                
            dependency_graph = self.graph_builder.build_from_files(code_files)
            graph_features = self.graph_builder.get_graph_features()
            return graph_features
        except Exception as e:
            self.logger.warning(f"Error extracting graph features: {str(e)}")
            return np.zeros(self.graph_dim, dtype=np.float32)
    
    def fit_scalers(self, all_embeddings: List[np.ndarray]):
        """
        Fit scalers on all embeddings for normalization.
        
        Args:
            all_embeddings: List of embedding vectors from training data
        """
        if not all_embeddings:
            self.logger.warning("No embeddings provided for scaler fitting")
            return
        
        self.logger.info(f"Fitting scalers on {len(all_embeddings)} embeddings")
        
        # Determine dimensions from first embedding
        first_embedding = all_embeddings[0]
        total_dim = len(first_embedding)
        
        if self.structure_dim == 0:
            # Estimate structure dimension (total - code - graph)
            self.structure_dim = total_dim - self.code_embedding_dim - self.graph_dim
        
        # Split embeddings into components
        structure_features = []
        graph_features = []
        
        for embedding in all_embeddings:
            if len(embedding) != total_dim:
                self.logger.warning(f"Inconsistent embedding dimension: {len(embedding)} vs {total_dim}")
                continue
                
            struct_start = self.code_embedding_dim
            struct_end = struct_start + self.structure_dim
            graph_start = struct_end
            
            structure_features.append(embedding[struct_start:struct_end])
            graph_features.append(embedding[graph_start:])
        
        # Fit scalers
        if structure_features:
            try:
                self.structure_scaler.fit(structure_features)
                self.logger.info("Structure scaler fitted successfully")
            except Exception as e:
                self.logger.warning(f"Failed to fit structure scaler: {str(e)}")
        
        if graph_features:
            try:
                self.graph_scaler.fit(graph_features)
                self.logger.info("Graph scaler fitted successfully")
            except Exception as e:
                self.logger.warning(f"Failed to fit graph scaler: {str(e)}")
        
        self.is_fitted = True
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding components using fitted scalers.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        if not self.is_fitted:
            self.logger.warning("Scalers not fitted. Returning unnormalized embedding.")
            return embedding
        
        try:
            # Split embedding into components
            struct_start = self.code_embedding_dim
            struct_end = struct_start + self.structure_dim
            graph_start = struct_end
            
            code_part = embedding[:self.code_embedding_dim]
            struct_part = embedding[struct_start:struct_end]
            graph_part = embedding[graph_start:]
            
            # Normalize structural and graph parts
            struct_normalized = self.structure_scaler.transform(struct_part.reshape(1, -1)).flatten()
            graph_normalized = self.graph_scaler.transform(graph_part.reshape(1, -1)).flatten()
            
            # Combine normalized components
            normalized = np.concatenate([code_part, struct_normalized, graph_normalized])
            return normalized.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error normalizing embedding: {str(e)}")
            return embedding
    
    def get_embedding_info(self) -> Dict[str, int]:
        """
        Get information about embedding dimensions.
        
        Returns:
            Dictionary with dimension information
        """
        return {
            'code_embedding_dim': self.code_embedding_dim,
            'structure_dim': self.structure_dim,
            'graph_dim': self.graph_dim,
            'total_dim': self.code_embedding_dim + self.structure_dim + self.graph_dim
        }
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file"""
        import pickle
        scaler_data = {
            'structure_scaler': self.structure_scaler,
            'graph_scaler': self.graph_scaler,
            'is_fitted': self.is_fitted,
            'embedding_info': self.get_embedding_info()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        self.logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file"""
        import pickle
        try:
            with open(filepath, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.structure_scaler = scaler_data['structure_scaler']
            self.graph_scaler = scaler_data['graph_scaler']
            self.is_fitted = scaler_data['is_fitted']
            
            # Update dimensions
            if 'embedding_info' in scaler_data:
                info = scaler_data['embedding_info']
                self.structure_dim = info.get('structure_dim', self.structure_dim)
                self.graph_dim = info.get('graph_dim', self.graph_dim)
                self.code_embedding_dim = info.get('code_embedding_dim', self.code_embedding_dim)
            
            self.logger.info(f"Scalers loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load scalers: {str(e)}")
            raise 