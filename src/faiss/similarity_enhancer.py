"""
Similarity Enhancement Module

This module provides enhanced similarity calculation methods that go beyond
simple cosine similarity to better capture code similarity patterns.

Author: Auto-generated
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import logging

from ..utils.logger import get_logger


class SimilarityEnhancer:
    """Enhanced similarity calculator for code embeddings"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Weights for different embedding components (optimized for GraphCodeBERT)
        self.component_weights = {
            'semantic': 0.65,     # Increased for GraphCodeBERT's superior semantic understanding
            'structural': 0.20,   # Reduced since GraphCodeBERT captures some structural info
            'patterns': 0.12,     # Slightly reduced
            'graph': 0.03         # Reduced since GraphCodeBERT includes graph information
        }
        
        # Similarity method weights (optimized for better self-retrieval)
        self.similarity_weights = {
            'cosine': 0.8,        # Increased primary similarity metric
            'structural_match': 0.15,  # Reduced structural boost
            'pattern_match': 0.05   # Reduced pattern boost
        }
    
    def enhanced_similarity(self, query_embedding: np.ndarray, 
                          candidate_embedding: np.ndarray,
                          query_metadata: Optional[Dict] = None,
                          candidate_metadata: Optional[Dict] = None) -> float:
        """
        Calculate enhanced similarity between embeddings.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            query_metadata: Optional metadata for query (structural info, etc.)
            candidate_metadata: Optional metadata for candidate
            
        Returns:
            Enhanced similarity score (0-1)
        """
        # Basic cosine similarity
        cosine_sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0, 0]
        
        # Initialize enhanced score with cosine similarity
        enhanced_score = cosine_sim * self.similarity_weights['cosine']
        
        # Add structural similarity boost if metadata available
        if query_metadata and candidate_metadata:
            structural_boost = self._calculate_structural_similarity(
                query_metadata, candidate_metadata
            )
            enhanced_score += structural_boost * self.similarity_weights['structural_match']
            
            pattern_boost = self._calculate_pattern_similarity(
                query_metadata, candidate_metadata
            )
            enhanced_score += pattern_boost * self.similarity_weights['pattern_match']
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, enhanced_score))
    
    def component_weighted_similarity(self, query_embedding: np.ndarray,
                                    candidate_embedding: np.ndarray,
                                    embedding_info: Dict[str, int]) -> float:
        """
        Calculate similarity with different weights for embedding components.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            embedding_info: Dictionary with dimension information
            
        Returns:
            Weighted similarity score
        """
        # Extract embedding components
        query_components = self._split_embedding(query_embedding, embedding_info)
        candidate_components = self._split_embedding(candidate_embedding, embedding_info)
        
        # Calculate component-wise similarities
        component_similarities = {}
        
        for component in ['semantic', 'structural', 'patterns', 'graph']:
            if component in query_components and component in candidate_components:
                q_comp = query_components[component]
                c_comp = candidate_components[component]
                
                if len(q_comp) > 0 and len(c_comp) > 0:
                    sim = cosine_similarity(
                        q_comp.reshape(1, -1),
                        c_comp.reshape(1, -1)
                    )[0, 0]
                    component_similarities[component] = sim
                else:
                    component_similarities[component] = 0.0
        
        # Calculate weighted average
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, weight in self.component_weights.items():
            if component in component_similarities:
                weighted_score += component_similarities[component] * weight
                total_weight += weight
        
        # Normalize by total weight used
        if total_weight > 0:
            weighted_score /= total_weight
        
        return max(0.0, min(1.0, weighted_score))
    
    def _split_embedding(self, embedding: np.ndarray, 
                        embedding_info: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Split embedding into components based on dimension info"""
        components = {}
        current_pos = 0
        
        # Extract semantic component (code embedding)
        code_dim = embedding_info.get('code_embedding_dim', 768)
        if current_pos + code_dim <= len(embedding):
            components['semantic'] = embedding[current_pos:current_pos + code_dim]
            current_pos += code_dim
        
        # Extract structural component
        struct_dim = embedding_info.get('structure_dim', 0)
        if struct_dim > 0 and current_pos + struct_dim <= len(embedding):
            components['structural'] = embedding[current_pos:current_pos + struct_dim]
            current_pos += struct_dim
        
        # Extract pattern component
        pattern_dim = embedding_info.get('code_pattern_dim', 50)
        if pattern_dim > 0 and current_pos + pattern_dim <= len(embedding):
            components['patterns'] = embedding[current_pos:current_pos + pattern_dim]
            current_pos += pattern_dim
        
        # Extract graph component
        graph_dim = embedding_info.get('graph_dim', 20)
        if graph_dim > 0 and current_pos + graph_dim <= len(embedding):
            components['graph'] = embedding[current_pos:current_pos + graph_dim]
        
        return components
    
    def _calculate_structural_similarity(self, query_meta: Dict, 
                                       candidate_meta: Dict) -> float:
        """Calculate structural similarity from metadata"""
        try:
            structural_features = [
                'num_classes', 'num_functions', 'cyclomatic_complexity',
                'max_nesting_depth', 'num_imports', 'lines_of_code'
            ]
            
            similarities = []
            
            for feature in structural_features:
                q_val = query_meta.get(feature, 0)
                c_val = candidate_meta.get(feature, 0)
                
                # Calculate relative similarity for this feature
                if q_val == 0 and c_val == 0:
                    sim = 1.0
                elif q_val == 0 or c_val == 0:
                    sim = 0.0
                else:
                    # Use min/max ratio for similarity
                    sim = min(q_val, c_val) / max(q_val, c_val)
                
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_similarity(self, query_meta: Dict,
                                    candidate_meta: Dict) -> float:
        """Calculate pattern-based similarity from metadata"""
        try:
            # Language similarity boost
            q_lang = query_meta.get('primary_language', 'unknown')
            c_lang = candidate_meta.get('primary_language', 'unknown')
            
            if q_lang == c_lang and q_lang != 'unknown':
                language_boost = 0.2
            else:
                language_boost = 0.0
            
            # File structure similarity
            q_files = query_meta.get('num_files', 1)
            c_files = candidate_meta.get('num_files', 1)
            
            if q_files > 0 and c_files > 0:
                file_similarity = min(q_files, c_files) / max(q_files, c_files)
            else:
                file_similarity = 0.0
            
            # Complexity ratio similarity
            q_complexity = query_meta.get('complexity_per_function', 0)
            c_complexity = candidate_meta.get('complexity_per_function', 0)
            
            if q_complexity > 0 and c_complexity > 0:
                complexity_similarity = min(q_complexity, c_complexity) / max(q_complexity, c_complexity)
            else:
                complexity_similarity = 0.0
            
            # Combine pattern features
            pattern_similarity = (language_boost + file_similarity + complexity_similarity) / 3
            return pattern_similarity
            
        except Exception:
            return 0.0
    
    def batch_enhanced_similarity(self, query_embedding: np.ndarray,
                                candidate_embeddings: List[np.ndarray],
                                embedding_info: Dict[str, int],
                                query_metadata: Optional[Dict] = None,
                                candidate_metadata_list: Optional[List[Dict]] = None) -> List[float]:
        """
        Calculate enhanced similarity for batch of candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            embedding_info: Embedding dimension information
            query_metadata: Optional query metadata
            candidate_metadata_list: Optional list of candidate metadata
            
        Returns:
            List of enhanced similarity scores
        """
        similarities = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            candidate_metadata = None
            if candidate_metadata_list and i < len(candidate_metadata_list):
                candidate_metadata = candidate_metadata_list[i]
            
            # Use component-weighted similarity as base
            base_similarity = self.component_weighted_similarity(
                query_embedding, candidate_embedding, embedding_info
            )
            
            # Add metadata-based enhancements if available
            if query_metadata and candidate_metadata:
                enhanced_sim = self.enhanced_similarity(
                    query_embedding, candidate_embedding,
                    query_metadata, candidate_metadata
                )
                # Blend the two approaches
                final_similarity = 0.7 * enhanced_sim + 0.3 * base_similarity
            else:
                final_similarity = base_similarity
            
            similarities.append(final_similarity)
        
        return similarities
    
    def adjust_similarity_weights(self, semantic_weight: float = 0.5,
                                structural_weight: float = 0.3,
                                pattern_weight: float = 0.15,
                                graph_weight: float = 0.05):
        """Adjust the weights for different embedding components"""
        total_weight = semantic_weight + structural_weight + pattern_weight + graph_weight
        
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Component weights sum to {total_weight}, normalizing...")
            semantic_weight /= total_weight
            structural_weight /= total_weight
            pattern_weight /= total_weight
            graph_weight /= total_weight
        
        self.component_weights = {
            'semantic': semantic_weight,
            'structural': structural_weight,
            'patterns': pattern_weight,
            'graph': graph_weight
        }
        
        self.logger.info(f"Updated component weights: {self.component_weights}")
    
    def get_similarity_explanation(self, query_embedding: np.ndarray,
                                 candidate_embedding: np.ndarray,
                                 embedding_info: Dict[str, int],
                                 query_metadata: Optional[Dict] = None,
                                 candidate_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get detailed explanation of similarity calculation.
        
        Returns:
            Dictionary with similarity breakdown and explanations
        """
        explanation = {}
        
        # Component-wise similarities
        query_components = self._split_embedding(query_embedding, embedding_info)
        candidate_components = self._split_embedding(candidate_embedding, embedding_info)
        
        component_scores = {}
        for component in ['semantic', 'structural', 'patterns', 'graph']:
            if component in query_components and component in candidate_components:
                q_comp = query_components[component]
                c_comp = candidate_components[component]
                
                if len(q_comp) > 0 and len(c_comp) > 0:
                    sim = cosine_similarity(q_comp.reshape(1, -1), c_comp.reshape(1, -1))[0, 0]
                    component_scores[component] = {
                        'similarity': float(sim),
                        'weight': self.component_weights[component],
                        'weighted_contribution': float(sim * self.component_weights[component])
                    }
        
        explanation['component_scores'] = component_scores
        
        # Overall scores
        base_similarity = self.component_weighted_similarity(
            query_embedding, candidate_embedding, embedding_info
        )
        explanation['component_weighted_similarity'] = float(base_similarity)
        
        if query_metadata and candidate_metadata:
            enhanced_sim = self.enhanced_similarity(
                query_embedding, candidate_embedding,
                query_metadata, candidate_metadata
            )
            explanation['enhanced_similarity'] = float(enhanced_sim)
            
            # Metadata-based boosts
            explanation['structural_boost'] = float(
                self._calculate_structural_similarity(query_metadata, candidate_metadata)
            )
            explanation['pattern_boost'] = float(
                self._calculate_pattern_similarity(query_metadata, candidate_metadata)
            )
        
        return explanation 