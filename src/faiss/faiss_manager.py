"""
FAISS Manager for Assignment-Separated Indices

This module manages separate FAISS indices for each assignment to improve
efficiency and provide better contextual results. Each assignment gets its
own dedicated FAISS index using Java code embeddings.

Author: Auto-generated
"""

import numpy as np
import pickle
import json
import pandas as pd
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
from .embedder import JavaCodeEmbedder  # Backward compatibility alias to OllamaJavaCodeEmbedder
# Individual assignment managers will be instances of a simple FAISS index class
# This is kept for backward compatibility in case any code expects it
FAISS_MANAGER_AVAILABLE = True
from ..utils.logger import get_logger


class SimpleFAISSIndex:
    """Simple FAISS index for individual assignments"""
    
    def __init__(self, index_type: str = "flat"):
        self.index_type = index_type
        self.index = None
        self.submissions = []
        self.embeddings = None
        self.is_trained = False
        self.logger = get_logger(__name__)
    
    def build_index(self, submissions: List[Submission], embedder, save_path: str) -> Dict[str, Any]:
        """Build FAISS index for this assignment"""
        self.submissions = submissions
        
        # Generate embeddings
        embeddings = []
        valid_submissions = []
        
        for submission in submissions:
            try:
                embedding = embedder.embed_codebase(submission.code_files)
                if embedding is not None and not np.isnan(embedding).any():
                    embeddings.append(embedding)
                    valid_submissions.append(submission)
            except Exception as e:
                self.logger.warning(f"Failed to embed {submission.file_name}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        self.embeddings = np.vstack(embeddings)
        self.submissions = valid_submissions
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            nlist = min(100, len(embeddings) // 4)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # Train index if needed
        if hasattr(self.index, 'train'):
            self.index.train(normalized_embeddings)
        
        # Add embeddings to index
        self.index.add(normalized_embeddings)
        self.is_trained = True
        
        # Save index and metadata
        save_path = Path(save_path)
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save submission metadata
        submission_metadata = []
        for i, submission in enumerate(self.submissions):
            submission_metadata.append({
                'index_id': i,
                'file_name': submission.file_name,
                'assignment_id': submission.assignment_id,
                'score': getattr(submission, 'score', None),
                'feedback': getattr(submission, 'feedback', None)
            })
        
        with open(save_path / "submissions.json", 'w') as f:
            json.dump(submission_metadata, f, indent=2, default=str)
        
        return {
            'valid_embeddings': len(valid_submissions),
            'total_submissions': len(submissions),
            'skipped_submissions': len(submissions) - len(valid_submissions),
            'index_type': self.index_type,
            'embedding_dimension': dimension
        }
    
    def load_index(self, path: str) -> bool:
        """Load FAISS index from disk"""
        try:
            index_path = Path(path)
            self.index = faiss.read_index(str(index_path / "index.faiss"))
            
            # Load submission metadata
            with open(index_path / "submissions.json", 'r') as f:
                submission_data = json.load(f)
            
            # Reconstruct submissions (minimal data for search)
            self.submissions = []
            for data in submission_data:
                # Create minimal submission objects for search results
                submission = type('Submission', (), {})()
                submission.file_name = data['file_name']
                submission.assignment_id = data['assignment_id']
                submission.score = data.get('score')
                submission.feedback = data.get('feedback')
                self.submissions.append(submission)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index from {path}: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, assignment_id: str, 
                      top_k: int = 5, score_threshold: float = 0.0, 
                      embedder=None) -> List[Dict[str, Any]]:
        """Search for similar submissions"""
        if not self.is_trained:
            return []
        
        # Normalize query embedding
        query_normalized = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        similarities, indices = self.index.search(query_normalized, min(top_k, len(self.submissions)))
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1 or similarity < score_threshold:
                continue
                
            submission = self.submissions[idx]
            results.append({
                'submission': submission,
                'similarity_score': float(similarity),
                'rank': i + 1
            })
        
        return results


class FAISSManager:
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
        self.assignment_managers: Dict[str, Any] = {}
        self.assignment_metadata: Dict[str, Dict] = {}
        
        self.logger.info(f"Assignment FAISS manager initialized with base path: {base_index_path}")
    
    def load_grades_for_assignment(self, assignment_id: str, submissions: List[Submission], 
                                 grade_mapping_csv: str) -> List[Submission]:
        """
        Load grade mapping and populate submissions with scores and feedback.
        
        Args:
            assignment_id: Assignment identifier
            submissions: List of submission objects to populate
            grade_mapping_csv: Path to CSV with student_name, grade, feedback
            
        Returns:
            List of submissions with populated scores and feedback
        """
        self.logger.info(f"Loading grades for assignment {assignment_id} from {grade_mapping_csv}")
        
        # Load grade mapping with proper CSV parsing settings
        try:
            grade_df = pd.read_csv(
                grade_mapping_csv,
                skipinitialspace=True,  # Skip whitespace after delimiter
                quotechar='"',          # Handle quoted strings properly
                escapechar='\\',        # Handle escape characters
                on_bad_lines='warn'     # Warn about bad lines but continue
            )
            grade_df.columns = grade_df.columns.str.strip()  # Remove whitespace
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {grade_mapping_csv}: {e}")
            # Try alternative parsing method
            try:
                self.logger.info("Trying alternative CSV parsing...")
                grade_df = pd.read_csv(
                    grade_mapping_csv,
                    sep=',',
                    skipinitialspace=True,
                    engine='python'  # Python engine is more forgiving
                )
                grade_df.columns = grade_df.columns.str.strip()
            except Exception as e2:
                self.logger.error(f"Alternative CSV parsing also failed: {e2}")
                raise ValueError(f"Cannot parse CSV file {grade_mapping_csv}. Please check the file format.")
        
        # Log the first few rows to understand the data structure
        self.logger.debug(f"CSV columns: {list(grade_df.columns)}")
        self.logger.debug(f"First 3 rows:\n{grade_df.head(3)}")
        
        # Create grade mapping dictionary
        grade_mapping = {}
        skipped_rows = 0
        
        for idx, row in grade_df.iterrows():
            try:
                student_name = str(row['student_name']).strip()
                
                # Handle various grade formats and validation
                grade_str = str(row['grade']).strip()
                
                # Skip empty or invalid rows
                if not student_name or student_name.lower() in ['nan', 'none', '']:
                    self.logger.debug(f"Skipping row {idx}: empty student name")
                    skipped_rows += 1
                    continue
                
                # Try to convert grade to float, with error handling
                try:
                    # Handle common non-numeric cases
                    if grade_str.lower() in ['nan', 'none', '', 'n/a']:
                        self.logger.warning(f"Skipping row {idx}: no grade for {student_name}")
                        skipped_rows += 1
                        continue
                    
                    # Remove any non-numeric characters and try conversion
                    grade_cleaned = ''.join(c for c in grade_str if c.isdigit() or c in '.-')
                    if not grade_cleaned:
                        raise ValueError(f"No numeric content in grade: '{grade_str}'")
                    
                    grade = float(grade_cleaned)
                    
                    # Validate grade range (assuming 0-1 scale)
                    if grade < 0 or grade > 1:
                        self.logger.warning(f"Grade {grade} for {student_name} outside expected range [0,1]. Using as-is.")
                    
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Cannot convert grade '{grade_str}' to float for student {student_name} (row {idx}): {e}")
                    skipped_rows += 1
                    continue
                
                feedback = str(row['feedback']).strip()
                
                # Handle missing feedback
                if feedback.lower() in ['nan', 'none', '']:
                    feedback = "No feedback provided"
                
                grade_mapping[student_name] = {'grade': grade, 'feedback': feedback}
                self.logger.debug(f"Loaded grade {grade} for student {student_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                self.logger.debug(f"Row data: {dict(row)}")
                skipped_rows += 1
                continue
        
        # Log statistics about grade loading
        self.logger.info(f"Loaded grades for {len(grade_mapping)} students")
        if skipped_rows > 0:
            self.logger.warning(f"Skipped {skipped_rows} invalid rows in CSV")
        
        # Filter submissions to only those with grades and populate score/feedback
        matched_submissions = []
        unmatched_students = []
        
        for submission in submissions:
            # Extract student name from submission file name
            student_name = self._extract_student_name(submission.file_name)
            
            if student_name in grade_mapping:
                submission.score = grade_mapping[student_name]['grade']
                submission.feedback = grade_mapping[student_name]['feedback']
                matched_submissions.append(submission)
            else:
                unmatched_students.append(student_name)
        
        # Log detailed matching information
        if unmatched_students:
            self.logger.warning(f"No grades found for {len(unmatched_students)} students:")
            for student in unmatched_students[:10]:  # Show first 10
                self.logger.warning(f"  - {student}")
            if len(unmatched_students) > 10:
                self.logger.warning(f"  ... and {len(unmatched_students) - 10} more")
        
        self.logger.info(f"Successfully matched {len(matched_submissions)} submissions with grades")
        
        if len(matched_submissions) == 0:
            available_names = list(grade_mapping.keys())[:10]
            self.logger.error("No submissions matched with grades!")
            self.logger.error(f"Available student names in CSV: {available_names}")
            self.logger.error(f"Sample submission file names: {[s.file_name for s in submissions[:5]]}")
            raise ValueError("No submissions could be matched with grades. Check student name extraction logic.")
        
        return matched_submissions
    
    def _extract_student_name(self, file_name: str) -> str:
        """Extract student name from submission file name"""
        # Remove common file extensions and path separators
        base_name = file_name.replace('.zip', '').replace('.rar', '')
        base_name = base_name.split('/')[-1].split('\\')[-1]
        
        # Extract first part (usually student name)
        if '_' in base_name:
            return base_name.split('_')[0]
        return base_name
    
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
                
                # Create individual FAISS index for this assignment
                assignment_manager = SimpleFAISSIndex(index_type=self.index_type)
                
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
    
    def build_assignment_indices_with_grades(self, submissions_by_assignment: Dict[str, List[Submission]], 
                                           embedder: JavaCodeEmbedder,
                                           grade_mapping_dir: str = "src/faiss/grade_mapping") -> Dict[str, Dict[str, Any]]:
        """
        Build assignment indices and populate submissions with grades from CSV files.
        
        Args:
            submissions_by_assignment: Dictionary mapping assignment_id to list of submissions
            embedder: Java code embedder instance
            grade_mapping_dir: Directory containing grade mapping CSV files
            
        Returns:
            Dictionary with build statistics for each assignment
        """
        self.logger.info(f"Building FAISS indices with grade mapping from {grade_mapping_dir}")
        
        grade_mapping_path = Path(grade_mapping_dir)
        enriched_submissions = {}
        
        for assignment_id, submissions in submissions_by_assignment.items():
            # Look for grade mapping CSV file
            csv_file = grade_mapping_path / f"{assignment_id}.csv"
            
            if csv_file.exists():
                self.logger.info(f"Found grade mapping for {assignment_id}: {csv_file}")
                try:
                    # Load grades and populate submissions
                    enriched_submissions[assignment_id] = self.load_grades_for_assignment(
                        assignment_id, submissions, str(csv_file)
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load grades for {assignment_id}: {e}")
                    # Use original submissions without grades
                    enriched_submissions[assignment_id] = submissions
            else:
                self.logger.warning(f"No grade mapping found for {assignment_id} at {csv_file}")
                # Use original submissions without grades
                enriched_submissions[assignment_id] = submissions
        
        # Build indices with enriched submissions
        return self.build_assignment_indices(enriched_submissions, embedder)
    
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
                        assignment_manager = SimpleFAISSIndex(index_type=self.index_type)
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