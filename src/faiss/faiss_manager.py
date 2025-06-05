"""
FAISS Vector Database Manager

This module manages the FAISS vector database for similarity search
of historical code submissions with enhanced similarity calculation.

Author: Auto-generated
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from .processor import Submission
from .embedder import HybridCodeEmbedder
from ..utils.logger import get_logger


class FAISSManager:
    """Manage FAISS index for similarity search with enhanced similarity calculation"""
    
    def __init__(self, embedding_dim: Optional[int] = None, index_type: str = "flat", 
                 use_enhanced_similarity: bool = True):
        """
        Initialize FAISS manager.
        
        Args:
            embedding_dim: Dimension of embeddings (determined automatically if None)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            use_enhanced_similarity: Whether to use enhanced similarity for re-ranking
        """
        self.logger = get_logger(__name__)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not available. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.submissions_metadata = []
        self.is_trained = False
        self.use_enhanced_similarity = use_enhanced_similarity
        
        self.logger.info(f"FAISS manager initialized with index type: {index_type}, "
                        f"enhanced similarity: {use_enhanced_similarity}")
    
    def build_index(self, submissions: List[Submission], embedder: HybridCodeEmbedder, 
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build FAISS index from submissions.
        
        Args:
            submissions: List of submission objects
            embedder: Hybrid code embedder instance
            save_path: Optional path to save the index
            
        Returns:
            Dictionary with build statistics
        """
        self.logger.info(f"Building FAISS index from {len(submissions)} submissions")
        
        # Generate embeddings for all submissions
        embeddings = []
        valid_submissions = []
        failed_count = 0
        
        for i, submission in enumerate(submissions):
            try:
                self.logger.debug(f"Processing submission {i+1}/{len(submissions)}: {submission.file_name}")
                
                embedding = embedder.embed_codebase(submission.code_files)
                embeddings.append(embedding)
                valid_submissions.append(submission)
                
                # Store embedding in submission for future use
                submission.embedding = embedding
                
            except Exception as e:
                self.logger.warning(f"Failed to embed {submission.file_name}: {str(e)}")
                failed_count += 1
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        self.logger.info(f"Generated {len(embeddings)} embeddings, {failed_count} failed")
        
        # Fit scalers and normalize embeddings
        embedder.fit_scalers(embeddings)
        normalized_embeddings = [embedder.normalize_embedding(emb) for emb in embeddings]
        
        # Convert to numpy array
        embedding_matrix = np.vstack(normalized_embeddings).astype(np.float32)
        
        # Set embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = embedding_matrix.shape[1]
        
        # Build appropriate FAISS index
        self.index = self._create_index(embedding_matrix)
        
        # Store metadata
        self.submissions_metadata = valid_submissions
        self.is_trained = True
        
        # Save if path provided
        if save_path:
            self.save_index(save_path)
        
        build_stats = {
            'total_submissions': len(submissions),
            'valid_embeddings': len(embeddings),
            'failed_embeddings': failed_count,
            'embedding_dimension': embedding_matrix.shape[1],
            'index_type': self.index_type,
            'assignments': list(set(sub.assignment_id for sub in valid_submissions)),
            'build_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"FAISS index built successfully: {build_stats}")
        return build_stats
    
    def _create_index(self, embedding_matrix: np.ndarray) -> faiss.Index:
        """Create appropriate FAISS index based on configuration"""
        n_vectors, dim = embedding_matrix.shape
        
        if self.index_type == "flat":
            # Simple flat index for exact search
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            
        elif self.index_type == "ivf" and n_vectors > 100:
            # IVF index for faster approximate search
            n_centroids = min(int(np.sqrt(n_vectors)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_centroids)
            
            # Train the index
            self.logger.info(f"Training IVF index with {n_centroids} centroids")
            index.train(embedding_matrix)
            
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 connections per node
            index.hnsw.efConstruction = 40
            
        else:
            # Fallback to flat index
            self.logger.warning(f"Unsupported index type {self.index_type} or insufficient data, using flat index")
            index = faiss.IndexFlatIP(dim)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Add vectors to index
        index.add(embedding_matrix)
        
        self.logger.info(f"Created {type(index).__name__} with {index.ntotal} vectors")
        return index
    
    def search_similar(self, query_embedding: np.ndarray, assignment_id: Optional[str] = None, 
                      top_k: int = 5, score_threshold: float = 0.0, 
                      embedder: Optional[HybridCodeEmbedder] = None,
                      query_metadata: Optional[Dict] = None,
                      use_enhanced_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar submissions with enhanced similarity calculation.
        
        Args:
            query_embedding: Query embedding vector
            assignment_id: Filter by assignment ID (None for all assignments)
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            embedder: Code embedder for enhanced similarity (required for enhanced features)
            query_metadata: Optional metadata for enhanced similarity
            use_enhanced_reranking: Whether to use enhanced similarity for re-ranking
            
        Returns:
            List of similar submissions with metadata
        """
        if not self.is_trained or self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS - get more results for enhanced re-ranking
        if use_enhanced_reranking and embedder and self.use_enhanced_similarity:
            search_k = min(len(self.submissions_metadata), max(top_k * 5, 100))
        else:
            search_k = min(len(self.submissions_metadata), max(top_k * 3, 50))
            
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Process results
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.submissions_metadata) or idx < 0:
                continue
                
            submission = self.submissions_metadata[idx]
            
            # Apply assignment filter
            if assignment_id and submission.assignment_id != assignment_id:
                continue
                
            # Basic score threshold
            if float(score) < score_threshold:
                continue
                
            candidates.append({
                'submission': submission,
                'faiss_score': float(score),
                'index': int(idx)
            })
        
        # Enhanced re-ranking if enabled and embedder is provided
        if use_enhanced_reranking and embedder and self.use_enhanced_similarity and candidates:
            self.logger.debug(f"Re-ranking {len(candidates)} candidates with enhanced similarity")
            
            # Collect candidate embeddings and metadata
            candidate_embeddings = []
            candidate_metadata_list = []
            
            for candidate in candidates:
                submission = candidate['submission']
                if hasattr(submission, 'embedding') and submission.embedding is not None:
                    candidate_embeddings.append(submission.embedding)
                    
                    # Extract metadata for enhanced similarity
                    metadata = self._extract_submission_metadata(submission)
                    candidate_metadata_list.append(metadata)
                else:
                    # Skip candidates without embeddings
                    continue
            
            if candidate_embeddings:
                # Calculate enhanced similarities
                enhanced_scores = embedder.batch_enhanced_similarity(
                    query_embedding.flatten(),
                    candidate_embeddings,
                    query_metadata,
                    candidate_metadata_list
                )
                
                # Update scores and re-sort
                for i, enhanced_score in enumerate(enhanced_scores):
                    if i < len(candidates):
                        candidates[i]['enhanced_score'] = enhanced_score
                        # Combine FAISS score and enhanced score
                        candidates[i]['final_score'] = 0.3 * candidates[i]['faiss_score'] + 0.7 * enhanced_score
                
                # Sort by enhanced score
                candidates.sort(key=lambda x: x.get('final_score', x['faiss_score']), reverse=True)
                self.logger.debug("Enhanced re-ranking completed")
        
        # Return top results
        results = []
        for candidate in candidates[:top_k]:
            submission = candidate['submission']
            
            # Extract student ID from file name if possible
            student_id = submission.file_name
            # Try to extract a cleaner student ID from the file name
            if '_' in submission.file_name:
                parts = submission.file_name.split('_')
                if len(parts) >= 2:
                    student_id = parts[0]  # First part is usually student identifier
            
            result = {
                'submission_id': student_id,
                'assignment_id': submission.assignment_id,
                'file_name': submission.file_name,
                'score': submission.score,
                'similarity_score': candidate.get('final_score', candidate['faiss_score']),
                'faiss_score': candidate['faiss_score'],
                'submission': submission
            }
            
            # Add enhanced score if available
            if 'enhanced_score' in candidate:
                result['enhanced_score'] = candidate['enhanced_score']
                
            results.append(result)
        
        return results
    
    def _extract_submission_metadata(self, submission: Submission) -> Dict[str, Any]:
        """Extract metadata from submission for enhanced similarity"""
        try:
            # Extract basic metadata
            metadata = {
                'assignment_id': submission.assignment_id,
                'score': submission.score,
                'num_files': len(submission.code_files),
                'primary_language': 'unknown'
            }
            
            # Detect primary language from file extensions
            extensions = [Path(f).suffix.lower() for f in submission.code_files.keys()]
            if extensions:
                ext_counts = {}
                for ext in extensions:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                primary_ext = max(ext_counts.items(), key=lambda x: x[1])[0]
                
                # Map extension to language
                lang_map = {'.py': 'python', '.java': 'java', '.js': 'javascript', 
                           '.ts': 'typescript', '.cpp': 'cpp', '.c': 'c'}
                metadata['primary_language'] = lang_map.get(primary_ext, 'unknown')
            
            # Add more metadata if available from submission
            if hasattr(submission, 'additional_metrics'):
                metadata.update(submission.additional_metrics)
                
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error extracting submission metadata: {e}")
            return {'assignment_id': getattr(submission, 'assignment_id', 'unknown')}
    
    def get_assignment_submissions(self, assignment_id: str) -> List[Submission]:
        """Get all submissions for a specific assignment"""
        return [sub for sub in self.submissions_metadata if sub.assignment_id == assignment_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        # Assignment distribution
        assignment_counts = {}
        score_stats = {}
        
        for submission in self.submissions_metadata:
            assignment_id = submission.assignment_id
            assignment_counts[assignment_id] = assignment_counts.get(assignment_id, 0) + 1
            
            if assignment_id not in score_stats:
                score_stats[assignment_id] = []
            score_stats[assignment_id].append(submission.score)
        
        # Calculate score statistics per assignment
        assignment_score_stats = {}
        for assignment_id, scores in score_stats.items():
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                assignment_score_stats[assignment_id] = {
                    'count': len(valid_scores),
                    'avg_score': np.mean(valid_scores),
                    'min_score': min(valid_scores),
                    'max_score': max(valid_scores),
                    'std_score': np.std(valid_scores)
                }
        
        return {
            'status': 'trained',
            'total_submissions': len(self.submissions_metadata),
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'index_size': self.index.ntotal if self.index else 0,
            'assignments': list(assignment_counts.keys()),
            'assignment_counts': assignment_counts,
            'assignment_score_stats': assignment_score_stats
        }
    
    def save_index(self, save_path: str):
        """Save FAISS index and metadata to disk"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.submissions_metadata, f)
        
        # Save configuration
        config_path = save_dir / "config.json"
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'save_timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"FAISS index saved to {save_path}")
    
    def load_index(self, save_path: str):
        """Load FAISS index and metadata from disk"""
        save_dir = Path(save_path)
        
        if not save_dir.exists():
            raise FileNotFoundError(f"Save directory not found: {save_path}")
        
        # Load configuration
        config_path = save_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.embedding_dim = config['embedding_dim']
                self.index_type = config['index_type']
                self.is_trained = config['is_trained']
        
        # Load FAISS index
        index_path = save_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = save_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.submissions_metadata = pickle.load(f)
        
        self.logger.info(f"FAISS index loaded from {save_path}")
        self.logger.info(f"Loaded {len(self.submissions_metadata)} submissions")
    
    def add_submission(self, submission: Submission, embedder: HybridCodeEmbedder):
        """Add a new submission to the existing index"""
        if not self.is_trained or self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate embedding
        embedding = embedder.embed_codebase(submission.code_files)
        normalized_embedding = embedder.normalize_embedding(embedding)
        
        # Add to index
        embedding_matrix = normalized_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(embedding_matrix)
        self.index.add(embedding_matrix)
        
        # Add to metadata
        submission.embedding = embedding
        self.submissions_metadata.append(submission)
        
        self.logger.info(f"Added submission {submission.file_name} to index")
    
    def remove_submission(self, submission_id: str):
        """Remove a submission from the index (requires rebuilding)"""
        # Note: FAISS doesn't support efficient removal, so we remove from metadata
        # and mark for rebuilding
        original_count = len(self.submissions_metadata)
        self.submissions_metadata = [
            sub for sub in self.submissions_metadata 
            if sub.file_name != submission_id
        ]
        
        removed_count = original_count - len(self.submissions_metadata)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} submission(s) with ID {submission_id}")
            self.logger.warning("Index needs to be rebuilt for changes to take effect")
        else:
            self.logger.warning(f"No submission found with ID {submission_id}")
    
    def search_by_code_similarity(self, code_files: Dict[str, str], embedder: HybridCodeEmbedder,
                                 assignment_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar submissions by code content.
        
        Args:
            code_files: Dictionary of code files to search for
            embedder: Embedder instance
            assignment_id: Optional assignment filter
            top_k: Number of results to return
            
        Returns:
            List of similar submissions
        """
        # Generate embedding for the query code
        query_embedding = embedder.embed_codebase(code_files)
        normalized_query = embedder.normalize_embedding(query_embedding)
        
        return self.search_similar(normalized_query, assignment_id, top_k) 