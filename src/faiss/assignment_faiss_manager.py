"""
Assignment-Specific FAISS Manager

This module manages separate FAISS indices for each assignment to improve
efficiency and provide better contextual results. Each assignment gets its
own dedicated FAISS index using Java code embeddings.

Author: Auto-generated
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from .processor import Submission
from .embedder import JavaCodeEmbedder
from .faiss_manager import FAISSManager
from ..utils.logger import get_logger


class AssignmentFAISSManager:
    """Manages separate FAISS indices for each assignment"""
    
    def __init__(self, base_index_path: str, index_type: str = "flat"):
        """
        Initialize assignment-specific FAISS manager.
        
        Args:
            base_index_path: Base directory for storing assignment indices
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.logger = get_logger(__name__)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not available. Install with: pip install faiss-cpu")
        
        self.base_index_path = Path(base_index_path)
        self.index_type = index_type
        
        # Dictionary to store individual FAISS managers for each assignment
        self.assignment_managers: Dict[str, FAISSManager] = {}
        self.assignment_metadata: Dict[str, Dict] = {}
        
        self.logger.info(f"Assignment FAISS manager initialized with base path: {base_index_path}")
    
    def build_assignment_indices(self, submissions_by_assignment: Dict[str, List[Submission]], 
                                embedder: JavaCodeEmbedder) -> Dict[str, Dict[str, Any]]:
        """
        Build separate FAISS indices for each assignment.
        
        Args:
            submissions_by_assignment: Dictionary mapping assignment_id to list of submissions
            embedder: Java code embedder instance
            
        Returns:
            Dictionary with build statistics for each assignment
        """
        self.logger.info(f"Building FAISS indices for {len(submissions_by_assignment)} assignments")
        
        all_build_stats = {}
        
        for assignment_id, submissions in submissions_by_assignment.items():
            if not submissions:
                self.logger.warning(f"No submissions found for assignment: {assignment_id}")
                continue
            
            self.logger.info(f"Building index for assignment '{assignment_id}' with {len(submissions)} submissions")
            
            try:
                # Create assignment-specific directory
                assignment_path = self.base_index_path / assignment_id
                assignment_path.mkdir(parents=True, exist_ok=True)
                
                # Create FAISS manager for this assignment
                assignment_manager = FAISSManager(index_type=self.index_type)
                
                # Build index for this assignment
                build_stats = assignment_manager.build_index(
                    submissions=submissions,
                    embedder=embedder,
                    save_path=str(assignment_path)
                )
                
                # Store the manager
                self.assignment_managers[assignment_id] = assignment_manager
                
                # Store metadata
                self.assignment_metadata[assignment_id] = {
                    'num_submissions': len(submissions),
                    'build_timestamp': datetime.now().isoformat(),
                    'index_path': str(assignment_path),
                    'build_stats': build_stats
                }
                
                all_build_stats[assignment_id] = build_stats
                
                self.logger.info(f"Successfully built index for assignment '{assignment_id}'")
                
            except Exception as e:
                self.logger.error(f"Failed to build index for assignment '{assignment_id}': {str(e)}")
                continue
        
        # Save overall metadata
        self._save_assignment_metadata()
        
        self.logger.info(f"Completed building indices for {len(all_build_stats)} assignments")
        return all_build_stats
    
    def search_similar_in_assignment(self, assignment_id: str, query_embedding: np.ndarray,
                                   top_k: int = 5, score_threshold: float = 0.0,
                                   embedder: Optional[JavaCodeEmbedder] = None) -> List[Dict[str, Any]]:
        """
        Search for similar submissions within a specific assignment.
        
        Args:
            assignment_id: ID of the assignment to search in
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            embedder: Java code embedder for similarity calculation
            
        Returns:
            List of similar submissions from the assignment
        """
        if assignment_id not in self.assignment_managers:
            self.logger.warning(f"No index found for assignment: {assignment_id}")
            return []
        
        assignment_manager = self.assignment_managers[assignment_id]
        
        if not assignment_manager.is_trained:
            self.logger.warning(f"Index not trained for assignment: {assignment_id}")
            return []
        
        self.logger.debug(f"Searching in assignment '{assignment_id}'")
        
        return assignment_manager.search_similar(
            query_embedding=query_embedding,
            assignment_id=assignment_id,  # This is redundant but kept for consistency
            top_k=top_k,
            score_threshold=score_threshold,
            embedder=embedder
        )
    
    def search_across_assignments(self, query_embedding: np.ndarray,
                                assignment_ids: Optional[List[str]] = None,
                                top_k_per_assignment: int = 3,
                                overall_top_k: int = 10,
                                score_threshold: float = 0.0,
                                embedder: Optional[JavaCodeEmbedder] = None) -> List[Dict[str, Any]]:
        """
        Search across multiple assignments and aggregate results.
        
        Args:
            query_embedding: Query embedding vector
            assignment_ids: List of assignment IDs to search (None for all)
            top_k_per_assignment: Number of results to get from each assignment
            overall_top_k: Total number of results to return
            score_threshold: Minimum similarity score threshold
            embedder: Java code embedder for similarity calculation
            
        Returns:
            Aggregated list of similar submissions from multiple assignments
        """
        if assignment_ids is None:
            assignment_ids = list(self.assignment_managers.keys())
        
        self.logger.debug(f"Searching across {len(assignment_ids)} assignments")
        
        all_results = []
        
        for assignment_id in assignment_ids:
            if assignment_id not in self.assignment_managers:
                continue
            
            results = self.search_similar_in_assignment(
                assignment_id=assignment_id,
                query_embedding=query_embedding,
                top_k=top_k_per_assignment,
                score_threshold=score_threshold,
                embedder=embedder
            )
            
            # Add assignment context to results
            for result in results:
                result['source_assignment'] = assignment_id
            
            all_results.extend(results)
        
        # Sort all results by similarity score and return top k
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return all_results[:overall_top_k]
    
    def get_assignment_statistics(self, assignment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for assignments.
        
        Args:
            assignment_id: Specific assignment ID (None for all assignments)
            
        Returns:
            Statistics dictionary
        """
        if assignment_id:
            if assignment_id in self.assignment_metadata:
                return self.assignment_metadata[assignment_id]
            else:
                return {}
        
        # Return statistics for all assignments
        total_submissions = sum(
            meta['num_submissions'] for meta in self.assignment_metadata.values()
        )
        
        return {
            'total_assignments': len(self.assignment_metadata),
            'total_submissions': total_submissions,
            'assignments': list(self.assignment_metadata.keys()),
            'per_assignment_stats': self.assignment_metadata
        }
    
    def load_assignment_indices(self) -> bool:
        """
        Load existing assignment indices from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Load assignment metadata
            metadata_path = self.base_index_path / "assignment_metadata.json"
            if not metadata_path.exists():
                self.logger.warning("No assignment metadata found")
                return False
            
            with open(metadata_path, 'r') as f:
                self.assignment_metadata = json.load(f)
            
            # Load individual assignment indices
            loaded_count = 0
            for assignment_id, metadata in self.assignment_metadata.items():
                try:
                    assignment_path = Path(metadata['index_path'])
                    if assignment_path.exists():
                        assignment_manager = FAISSManager(index_type=self.index_type)
                        assignment_manager.load_index(str(assignment_path))
                        self.assignment_managers[assignment_id] = assignment_manager
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load index for assignment {assignment_id}: {e}")
                    continue
            
            self.logger.info(f"Loaded {loaded_count} assignment indices")
            return loaded_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load assignment indices: {e}")
            return False
    
    def _save_assignment_metadata(self):
        """Save assignment metadata to disk"""
        try:
            metadata_path = self.base_index_path / "assignment_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.assignment_metadata, f, indent=2, default=str)
            self.logger.info(f"Saved assignment metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save assignment metadata: {e}")
    
    def get_available_assignments(self) -> List[str]:
        """Get list of available assignment IDs"""
        return list(self.assignment_managers.keys())
    
    def has_assignment(self, assignment_id: str) -> bool:
        """Check if assignment index is available"""
        return assignment_id in self.assignment_managers
    
    def remove_assignment_index(self, assignment_id: str) -> bool:
        """
        Remove an assignment index.
        
        Args:
            assignment_id: Assignment ID to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            if assignment_id in self.assignment_managers:
                del self.assignment_managers[assignment_id]
            
            if assignment_id in self.assignment_metadata:
                # Remove files
                assignment_path = Path(self.assignment_metadata[assignment_id]['index_path'])
                if assignment_path.exists():
                    import shutil
                    shutil.rmtree(assignment_path)
                
                del self.assignment_metadata[assignment_id]
                self._save_assignment_metadata()
            
            self.logger.info(f"Removed assignment index: {assignment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove assignment index {assignment_id}: {e}")
            return False 