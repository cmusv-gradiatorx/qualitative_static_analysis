"""
Hybrid Code Embedding System

This module provides multi-layer code embeddings that combine:
1. Semantic embeddings (CodeBERT-based)
2. Structural metrics (AST-based)
3. Dependency graph features (NetworkX-based)
4. Enhanced similarity calculation with component weighting

Author: Auto-generated
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import re
import hashlib

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

# Import enhanced similarity calculator
from .similarity_enhancer import SimilarityEnhancer
from ..utils.logger import get_logger


class ImprovedCodeEmbedder:
    """Improved embedding system with enhanced similarity detection and component weighting"""
    
    def __init__(self, model_name: str = 'microsoft/codebert-base', device: Optional[str] = None):
        """
        Initialize the improved code embedder with enhanced similarity.
        
        Args:
            model_name: Name of the model to use for semantic embeddings
            device: Device to run models on ('cpu', 'cuda', or None for auto)
        """
        self.logger = get_logger(__name__)
        
        # Choose better default models for code
        if model_name == 'microsoft/unixcoder-base':
            # Use a better model for code similarity
            model_name = 'microsoft/codebert-base'
            self.logger.info("Switching to CodeBERT for better code understanding")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.code_model = None
        self.tokenizer = None
        self._initialize_models()
        
        # Initialize enhanced similarity calculator
        self.similarity_enhancer = SimilarityEnhancer()
        self.logger.info("Enhanced similarity calculator initialized")
        
        # Initialize analyzers
        if MULTI_LANG_AVAILABLE:
            self.structure_analyzer = MultiLanguageCodeAnalyzer()
            self.graph_builder = MultiLanguageDependencyGraphBuilder()
        elif MULTI_LANG_AVAILABLE is False:
            from .analyzer import CodeStructureAnalyzer, DependencyGraphBuilder
            self.structure_analyzer = CodeStructureAnalyzer()
            self.graph_builder = DependencyGraphBuilder()
        else:
            self.structure_analyzer = None
            self.graph_builder = None
            self.logger.warning("No code analyzers available")
        
        # Initialize scalers
        self.structure_scaler = StandardScaler()
        self.graph_scaler = StandardScaler() 
        self.is_fitted = False
        
        # Embedding dimensions - updated for better balance
        self.code_embedding_dim = 768  # Standard for CodeBERT
        self.structure_dim = 0  # Will be determined from first analysis
        self.graph_dim = 20 if MULTI_LANG_AVAILABLE else 15
        self.code_pattern_dim = 50  # New: pattern-based features
        
        self.logger.info("Improved code embedder initialized successfully with enhanced similarity")
    
    def _initialize_models(self):
        """Initialize the code embedding models with better model selection"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                
                try:
                    # Try specific code models first
                    if 'codebert' in self.model_name.lower():
                        self.logger.info("Using CodeBERT-based model for better code understanding")
                        self.code_model = SentenceTransformer(self.model_name, device=self.device)
                    else:
                        # Try to use sentence-transformers wrapper
                        self.code_model = SentenceTransformer(self.model_name, device=self.device)
                    
                    self.logger.info("SentenceTransformer model loaded successfully")
                except Exception as st_error:
                    self.logger.warning(f"SentenceTransformer failed: {st_error}")
                    # Fall back to transformers
                    if TRANSFORMERS_AVAILABLE:
                        self.logger.info("Falling back to raw transformers")
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        self.code_model = AutoModel.from_pretrained(self.model_name).to(self.device)
                        self.code_model.eval()
                    else:
                        self.code_model = None
            
            elif TRANSFORMERS_AVAILABLE:
                self.logger.info(f"Loading Transformers model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.code_model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.code_model.eval()
                self.logger.info("Transformers model loaded successfully")
            
            else:
                self.logger.warning("No embedding models available. Using pattern-based embeddings only.")
                self.code_model = None
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            self.logger.warning("Falling back to pattern-based embeddings")
            self.code_model = None

    def embed_codebase(self, code_files: Dict[str, str]) -> np.ndarray:
        """
        Generate improved hybrid embedding for entire codebase.
        
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
        
        # 4. Extract code pattern features
        pattern_features = self._extract_code_patterns(code_files)
        
        # Set dimensions on first embedding
        if self.structure_dim == 0:
            self.structure_dim = len(structure_features)
        
        # Combine all embeddings with proper handling
        try:
            combined_embedding = np.concatenate([
                code_embedding,      # Semantic understanding (768)
                structure_features,  # Structural metrics (variable)
                graph_features,      # Graph/dependency features (15-20)
                pattern_features     # Code patterns (50)
            ])
            
            self.logger.debug(f"Combined embedding dimensions: {len(combined_embedding)} "
                            f"(code: {len(code_embedding)}, struct: {len(structure_features)}, "
                            f"graph: {len(graph_features)}, patterns: {len(pattern_features)})")
            
            return combined_embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error combining embeddings: {str(e)}")
            # Return a safe fallback embedding
            fallback_dim = self.code_embedding_dim + 100  # Safe fallback size
            return np.random.normal(0, 0.01, fallback_dim).astype(np.float32)

    def _generate_semantic_embedding(self, code_files: Dict[str, str]) -> np.ndarray:
        """Generate improved semantic embedding with better code preprocessing"""
        try:
            # Enhanced code preprocessing
            processed_code = self._preprocess_code_for_embedding(code_files)
            
            if self.code_model is None:
                # Better fallback: pattern-based embeddings
                return self._generate_pattern_based_embedding(processed_code)
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self.code_model, SentenceTransformer):
                # Use SentenceTransformer with better chunking
                embedding = self._encode_with_chunking(processed_code)
                return embedding.astype(np.float32)
            
            elif TRANSFORMERS_AVAILABLE and self.tokenizer is not None:
                # Use raw transformers with improved pooling
                with torch.no_grad():
                    inputs = self.tokenizer(processed_code, return_tensors='pt', 
                                          max_length=512, truncation=True, padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.code_model(**inputs)
                    # Use better pooling strategy
                    embedding = self._improved_pooling(outputs, inputs['attention_mask'])
                    return embedding.astype(np.float32)
            
            else:
                return self._generate_pattern_based_embedding(processed_code)
                
        except Exception as e:
            self.logger.warning(f"Error generating semantic embedding: {str(e)}")
            return self._generate_pattern_based_embedding("")
    
    def _preprocess_code_for_embedding(self, code_files: Dict[str, str]) -> str:
        """Enhanced code preprocessing for better semantic understanding"""
        relevant_files = {}
        
        # Filter programming files more intelligently
        programming_extensions = {
            '.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.c', '.h', '.cs',
            '.php', '.rb', '.go', '.rs', '.kt', '.scala', '.swift', '.m', '.mm'
        }
        
        for filename, content in code_files.items():
            # Skip binary/meta files
            if filename.startswith('__MACOSX') or filename.startswith('.'):
                continue
            
            # Include files with programming extensions
            if any(filename.lower().endswith(ext) for ext in programming_extensions):
                relevant_files[filename] = content
        
        if not relevant_files:
            relevant_files = code_files  # Fallback to all files
        
        # Enhanced code combining with structure preservation
        code_parts = []
        for filename, content in relevant_files.items():
            # Detect language and add context
            lang_hint = self._detect_language_from_extension(filename)
            
            # Clean and normalize code
            cleaned_content = self._clean_code_content(content)
            
            # Add with better formatting
            code_parts.append(f"// FILE: {filename} [{lang_hint}]\n{cleaned_content}")
        
        combined_code = "\n\n// ===== NEXT FILE =====\n\n".join(code_parts)
        
        # Smart truncation preserving structure
        if len(combined_code) > 12000:  # Conservative limit
            # Truncate but try to preserve complete functions/classes
            truncated = self._smart_truncate(combined_code, 12000)
            combined_code = truncated + "\n// ... (truncated)"
        
        return combined_code
    
    def _clean_code_content(self, content: str) -> str:
        """Clean code content for better embedding"""
        # Remove excessive whitespace but preserve structure
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            # Skip empty lines but keep one for structure
            if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(line)
        
        # Remove comments that are just noise
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Smart truncation that preserves code structure"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at natural boundaries
        boundaries = ['\n\n// ===== NEXT FILE =====\n\n', '\nclass ', '\ndef ', '\nfunction ', '\npublic ']
        
        for boundary in boundaries:
            last_boundary = text.rfind(boundary, 0, max_length)
            if last_boundary > max_length * 0.7:  # At least 70% of content
                return text[:last_boundary]
        
        # Fallback: truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.9:
            truncated = truncated[:last_space]
        
        return truncated
    
    def _detect_language_from_extension(self, filename: str) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python', '.java': 'java', '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript', '.cpp': 'cpp', '.c': 'c',
            '.h': 'c', '.cs': 'csharp', '.php': 'php', '.rb': 'ruby',
            '.go': 'go', '.rs': 'rust', '.kt': 'kotlin', '.scala': 'scala'
        }
        
        for ext, lang in ext_to_lang.items():
            if filename.lower().endswith(ext):
                return lang
        return 'unknown'
    
    def _encode_with_chunking(self, text: str) -> np.ndarray:
        """Encode long text with intelligent chunking"""
        # If text is short enough, encode directly
        if len(text) <= 8000:
            return self.code_model.encode([text], convert_to_numpy=True)[0]
        
        # Split into chunks at natural boundaries
        chunks = self._split_into_chunks(text, 6000)
        chunk_embeddings = []
        
        for chunk in chunks:
            embedding = self.code_model.encode([chunk], convert_to_numpy=True)[0]
            chunk_embeddings.append(embedding)
        
        # Combine chunks using weighted average (give more weight to earlier chunks)
        weights = np.exp(-0.1 * np.arange(len(chunk_embeddings)))  # Exponential decay
        weights = weights / weights.sum()
        
        combined = np.average(chunk_embeddings, axis=0, weights=weights)
        return combined
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at natural boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = current_pos + chunk_size
            
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            
            # Find a good splitting point
            chunk_text = text[current_pos:end_pos]
            
            # Try to split at natural boundaries
            for boundary in ['\n\nclass ', '\n\ndef ', '\n\nfunction ', '\n\n']:
                last_boundary = chunk_text.rfind(boundary)
                if last_boundary > chunk_size * 0.5:  # At least half the chunk
                    end_pos = current_pos + last_boundary
                    break
            
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos
        
        return chunks
    
    def _improved_pooling(self, outputs, attention_mask) -> np.ndarray:
        """Improved pooling strategy for better code representation"""
        last_hidden_states = outputs.last_hidden_state
        
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Also get CLS token representation
        cls_token = last_hidden_states[:, 0, :]
        
        # Combine CLS and mean pooling (weighted average)
        combined = 0.3 * cls_token + 0.7 * mean_pooled
        
        return combined.cpu().numpy()[0]
    
    def _generate_pattern_based_embedding(self, code: str) -> np.ndarray:
        """Generate embeddings based on code patterns when models aren't available"""
        features = np.zeros(self.code_embedding_dim, dtype=np.float32)
        
        if not code:
            return features
        
        # Pattern-based features
        patterns = {
            'class_declarations': len(re.findall(r'\bclass\s+\w+', code, re.IGNORECASE)),
            'function_definitions': len(re.findall(r'\b(def|function|void|int|String)\s+\w+\s*\(', code)),
            'imports': len(re.findall(r'\b(import|include|using|from)\s+', code)),
            'loops': len(re.findall(r'\b(for|while)\s*\(', code)),
            'conditionals': len(re.findall(r'\bif\s*\(', code)),
            'try_catch': len(re.findall(r'\b(try|catch|except)\b', code)),
            'comments': len(re.findall(r'//.*|/\*.*?\*/|#.*', code)),
            'strings': len(re.findall(r'"[^"]*"|\'[^\']*\'', code)),
            'numbers': len(re.findall(r'\b\d+\b', code))
        }
        
        # Normalize and encode patterns
        total_lines = max(len(code.split('\n')), 1)
        for i, (pattern, count) in enumerate(patterns.items()):
            if i < len(features):
                features[i] = min(count / total_lines, 1.0)  # Normalize by lines
        
        # Add code complexity indicators
        complexity_indicators = [
            len(set(re.findall(r'\b[a-zA-Z_]\w*\b', code))) / max(len(code.split()), 1),  # Vocabulary diversity
            len(re.findall(r'[{}()]', code)) / max(len(code), 1),  # Structural complexity
            len(re.findall(r'\b(public|private|static|final|const)\b', code)) / max(total_lines, 1)  # Modifiers
        ]
        
        for i, indicator in enumerate(complexity_indicators):
            idx = len(patterns) + i
            if idx < len(features):
                features[idx] = min(indicator, 1.0)
        
        return features
    
    def _extract_code_patterns(self, code_files: Dict[str, str]) -> np.ndarray:
        """Extract code pattern features for better similarity detection"""
        features = np.zeros(self.code_pattern_dim, dtype=np.float32)
        
        # Combine all code
        all_code = '\n'.join(code_files.values())
        
        if not all_code:
            return features
        
        # Extract various code patterns
        patterns = {
            # Structural patterns
            'avg_line_length': np.mean([len(line) for line in all_code.split('\n')]),
            'max_line_length': max([len(line) for line in all_code.split('\n')]),
            'indentation_consistency': self._measure_indentation_consistency(all_code),
            'brace_style': self._measure_brace_style(all_code),
            
            # Naming patterns
            'camel_case_usage': len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', all_code)),
            'snake_case_usage': len(re.findall(r'\b[a-z]+_[a-z_]+\b', all_code)),
            'constant_usage': len(re.findall(r'\b[A-Z][A-Z_]+\b', all_code)),
            
            # Design patterns
            'interface_usage': len(re.findall(r'\binterface\s+\w+', all_code, re.IGNORECASE)),
            'abstract_usage': len(re.findall(r'\babstract\s+', all_code, re.IGNORECASE)),
            'inheritance_depth': self._estimate_inheritance_depth(all_code),
            
            # Error handling patterns
            'exception_handling': len(re.findall(r'\b(try|catch|except|finally|throw|raise)\b', all_code)),
            'null_checks': len(re.findall(r'\b(null|None|nil)\s*[!=]=', all_code)),
            
            # Documentation patterns
            'docstring_usage': len(re.findall(r'""".*?"""|\'\'\'.*?\'\'\'', all_code, re.DOTALL)),
            'comment_density': len(re.findall(r'//.*|/\*.*?\*/|#.*', all_code)) / max(len(all_code.split('\n')), 1),
            
            # Complexity patterns
            'nested_structures': self._count_nested_structures(all_code),
            'method_chaining': len(re.findall(r'\.\w+\(\)[.\w()]*', all_code)),
            'lambda_usage': len(re.findall(r'\blambda\b|=>', all_code)),
            
            # Code organization
            'file_count': len(code_files),
            'avg_file_size': np.mean([len(content) for content in code_files.values()]),
            'package_structure_depth': max([filename.count('/') for filename in code_files.keys()], default=0)
        }
        
        # Normalize and store patterns
        total_lines = max(len(all_code.split('\n')), 1)
        total_chars = max(len(all_code), 1)
        
        feature_idx = 0
        for pattern_name, value in patterns.items():
            if feature_idx >= self.code_pattern_dim:
                break
                
            # Normalize different types of values
            if 'length' in pattern_name or 'size' in pattern_name:
                normalized_value = min(value / 100, 5.0)  # Normalize lengths
            elif 'count' in pattern_name or 'usage' in pattern_name:
                normalized_value = min(value / total_lines, 2.0)  # Normalize counts by lines
            elif 'density' in pattern_name or 'consistency' in pattern_name:
                normalized_value = min(value, 1.0)  # Already normalized values
            else:
                normalized_value = min(value / max(total_lines, 1), 2.0)  # General normalization
            
            features[feature_idx] = normalized_value
            feature_idx += 1
        
        return features
    
    def _measure_indentation_consistency(self, code: str) -> float:
        """Measure consistency of indentation style"""
        lines = [line for line in code.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        space_indents = sum(1 for line in lines if line.startswith('    '))
        tab_indents = sum(1 for line in lines if line.startswith('\t'))
        total_indented = sum(1 for line in lines if line.startswith((' ', '\t')))
        
        if total_indented == 0:
            return 1.0
        
        # Higher score for consistent style
        consistency = max(space_indents, tab_indents) / total_indented
        return consistency
    
    def _measure_brace_style(self, code: str) -> float:
        """Measure brace placement consistency"""
        # Count different brace styles
        same_line_braces = len(re.findall(r'\)\s*{', code))
        new_line_braces = len(re.findall(r'\)\s*\n\s*{', code))
        total_braces = same_line_braces + new_line_braces
        
        if total_braces == 0:
            return 0.0
        
        return max(same_line_braces, new_line_braces) / total_braces
    
    def _estimate_inheritance_depth(self, code: str) -> int:
        """Estimate maximum inheritance depth"""
        # Simple heuristic: count extends/implements chains
        inheritance_patterns = re.findall(r'\bextends\s+\w+|\bimplements\s+[\w,\s]+', code, re.IGNORECASE)
        return len(inheritance_patterns)
    
    def _count_nested_structures(self, code: str) -> int:
        """Count deeply nested structures"""
        max_nesting = 0
        current_nesting = 0
        
        for char in code:
            if char in '{([':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char in '})]':
                current_nesting = max(0, current_nesting - 1)
        
        return max_nesting
    
    def _extract_structural_features(self, code_files: Dict[str, str]) -> np.ndarray:
        """Extract structural features from codebase"""
        try:
            if self.structure_analyzer is None:
                # Return minimal features when no analyzer available
                return np.zeros(10, dtype=np.float32)
                
            aggregated_metrics = self.structure_analyzer.analyze_codebase(code_files)
            
            # Filter out non-numeric values and encode categorical ones
            numeric_features = []
            categorical_encodings = []
            
            # Define expected categorical features and their encodings
            language_encoding = {
                'python': 1.0,
                'java': 2.0, 
                'javascript': 3.0,
                'typescript': 4.0,
                'unknown': 0.0
            }
            
            for key in sorted(aggregated_metrics.keys()):
                value = aggregated_metrics[key]
                
                if key == 'primary_language':
                    # Encode primary language as numeric
                    encoded_value = language_encoding.get(str(value).lower(), 0.0)
                    categorical_encodings.append(encoded_value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Include numeric values directly
                    numeric_features.append(float(value))
                else:
                    # Skip other non-numeric values
                    self.logger.debug(f"Skipping non-numeric feature {key}: {value}")
            
            # Combine numeric and categorical features
            all_features = numeric_features + categorical_encodings
            features = np.array(all_features, dtype=np.float32)
            
            self.logger.debug(f"Extracted {len(features)} structural features")
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
            # Estimate structure dimension (total - code - graph - pattern)
            self.structure_dim = total_dim - self.code_embedding_dim - self.graph_dim - self.code_pattern_dim
            self.structure_dim = max(self.structure_dim, 0)  # Ensure non-negative
        
        # Split embeddings into components
        structure_features = []
        graph_features = []
        
        for embedding in all_embeddings:
            if len(embedding) != total_dim:
                self.logger.warning(f"Inconsistent embedding dimension: {len(embedding)} vs {total_dim}")
                continue
                
            struct_start = self.code_embedding_dim
            struct_end = struct_start + self.structure_dim
            graph_start = struct_end + self.code_pattern_dim  # Skip pattern features
            
            if struct_end > struct_start:
                structure_features.append(embedding[struct_start:struct_end])
            if graph_start < len(embedding):
                graph_features.append(embedding[graph_start:graph_start + self.graph_dim])
        
        # Fit scalers
        if structure_features and len(structure_features[0]) > 0:
            try:
                self.structure_scaler.fit(structure_features)
                self.logger.info("Structure scaler fitted successfully")
            except Exception as e:
                self.logger.warning(f"Failed to fit structure scaler: {str(e)}")
        
        if graph_features and len(graph_features[0]) > 0:
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
            pattern_start = struct_end
            pattern_end = pattern_start + self.code_pattern_dim
            graph_start = pattern_end
            
            code_part = embedding[:self.code_embedding_dim]
            pattern_part = embedding[pattern_start:pattern_end] if pattern_end <= len(embedding) else np.array([])
            
            # Normalize structural and graph parts if they exist
            if struct_end > struct_start and struct_end <= len(embedding):
                struct_part = embedding[struct_start:struct_end]
                struct_normalized = self.structure_scaler.transform(struct_part.reshape(1, -1)).flatten()
            else:
                struct_normalized = np.array([])
            
            if graph_start < len(embedding):
                graph_part = embedding[graph_start:graph_start + self.graph_dim]
                if len(graph_part) == self.graph_dim:
                    graph_normalized = self.graph_scaler.transform(graph_part.reshape(1, -1)).flatten()
                else:
                    graph_normalized = graph_part  # Use as-is if wrong dimension
            else:
                graph_normalized = np.array([])
            
            # Combine normalized components
            parts = [code_part]
            if len(struct_normalized) > 0:
                parts.append(struct_normalized)
            if len(pattern_part) > 0:
                parts.append(pattern_part)
            if len(graph_normalized) > 0:
                parts.append(graph_normalized)
            
            normalized = np.concatenate(parts)
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
            'code_pattern_dim': self.code_pattern_dim,
            'total_dim': self.code_embedding_dim + self.structure_dim + self.graph_dim + self.code_pattern_dim
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
                self.code_pattern_dim = info.get('code_pattern_dim', getattr(self, 'code_pattern_dim', 50))
            
            self.logger.info(f"Scalers loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load scalers: {str(e)}")
            raise

    def calculate_enhanced_similarity(self, query_embedding: np.ndarray, 
                                    candidate_embedding: np.ndarray,
                                    query_metadata: Optional[Dict] = None,
                                    candidate_metadata: Optional[Dict] = None) -> float:
        """
        Calculate enhanced similarity between two embeddings using component weighting.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            query_metadata: Optional metadata for query
            candidate_metadata: Optional metadata for candidate
            
        Returns:
            Enhanced similarity score (0-1)
        """
        embedding_info = self.get_embedding_info()
        
        return self.similarity_enhancer.enhanced_similarity(
            query_embedding, candidate_embedding, 
            query_metadata, candidate_metadata
        )
    
    def calculate_component_weighted_similarity(self, query_embedding: np.ndarray,
                                              candidate_embedding: np.ndarray) -> float:
        """
        Calculate similarity with different weights for embedding components.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            
        Returns:
            Component-weighted similarity score
        """
        embedding_info = self.get_embedding_info()
        
        return self.similarity_enhancer.component_weighted_similarity(
            query_embedding, candidate_embedding, embedding_info
        )
    
    def batch_enhanced_similarity(self, query_embedding: np.ndarray,
                                candidate_embeddings: List[np.ndarray],
                                query_metadata: Optional[Dict] = None,
                                candidate_metadata_list: Optional[List[Dict]] = None) -> List[float]:
        """
        Calculate enhanced similarity for batch of candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            query_metadata: Optional query metadata
            candidate_metadata_list: Optional list of candidate metadata
            
        Returns:
            List of enhanced similarity scores
        """
        embedding_info = self.get_embedding_info()
        
        return self.similarity_enhancer.batch_enhanced_similarity(
            query_embedding, candidate_embeddings, embedding_info,
            query_metadata, candidate_metadata_list
        )
    
    def adjust_similarity_weights(self, semantic_weight: float = 0.5,
                                structural_weight: float = 0.3,
                                pattern_weight: float = 0.15,
                                graph_weight: float = 0.05):
        """
        Adjust the weights for different embedding components in similarity calculation.
        
        Args:
            semantic_weight: Weight for semantic embedding component
            structural_weight: Weight for structural features component
            pattern_weight: Weight for code patterns component
            graph_weight: Weight for dependency graph component
        """
        self.similarity_enhancer.adjust_similarity_weights(
            semantic_weight, structural_weight, pattern_weight, graph_weight
        )
        self.logger.info(f"Updated similarity weights: semantic={semantic_weight}, "
                        f"structural={structural_weight}, pattern={pattern_weight}, graph={graph_weight}")
    
    def get_similarity_explanation(self, query_embedding: np.ndarray,
                                 candidate_embedding: np.ndarray,
                                 query_metadata: Optional[Dict] = None,
                                 candidate_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Get detailed explanation of similarity calculation.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            query_metadata: Optional query metadata
            candidate_metadata: Optional candidate metadata
            
        Returns:
            Dictionary with similarity breakdown and explanations
        """
        embedding_info = self.get_embedding_info()
        
        return self.similarity_enhancer.get_similarity_explanation(
            query_embedding, candidate_embedding, embedding_info,
            query_metadata, candidate_metadata
        )


# Backward compatibility alias
HybridCodeEmbedder = ImprovedCodeEmbedder 