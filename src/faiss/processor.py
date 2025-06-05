"""
Submission Processing

This module handles extraction of submissions from ZIP files and processing
of code files along with metadata (scores, feedback).

Author: Auto-generated
"""

import os
import zipfile
import tempfile
import shutil
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils.logger import get_logger

# Import multi-language analyzer for file extension support
try:
    from .analyzers import MultiLanguageCodeAnalyzer
    MULTI_LANGUAGE_AVAILABLE = True
except ImportError:
    MULTI_LANGUAGE_AVAILABLE = False


@dataclass
class Submission:
    """Data class representing a code submission with metadata"""
    assignment_id: str
    file_name: str
    code_files: Dict[str, str]  # filename -> content
    score: float
    feedback: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class SubmissionProcessor:
    """Process zip file and extract submissions with metadata"""
    
    def __init__(self, zip_path: str, assignment_id: Optional[str] = None):
        """
        Initialize submission processor.
        
        Args:
            zip_path: Path to the ZIP file containing submissions
            assignment_id: Optional assignment ID (will be inferred if not provided)
        """
        self.zip_path = Path(zip_path)
        self.assignment_id = assignment_id or self._infer_assignment_id()
        self.submissions = []
        self.logger = get_logger(__name__)
        
        # Initialize multi-language analyzer if available
        if MULTI_LANGUAGE_AVAILABLE:
            self.analyzer = MultiLanguageCodeAnalyzer()
            self.supported_extensions = set(self.analyzer.get_supported_extensions())
        else:
            self.analyzer = None
            self.supported_extensions = {'.java'}  # Fallback to Java only
        
        # Validate ZIP file exists
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {self.zip_path}")
    
    def _infer_assignment_id(self) -> str:
        """Infer assignment ID from ZIP file name"""
        filename = self.zip_path.stem
        
        # Try to extract common assignment patterns
        patterns = [
            r'assignment[_-](\d+)',
            r'hw[_-](\d+)', 
            r'project[_-](\d+)',
            r'milestone[_-](\d+)',
            r'([a-zA-Z]+)[-_]?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # Fallback to filename
        return filename
    
    def extract_submissions(self) -> List[Submission]:
        """
        Extract all submissions from ZIP file.
        
        Returns:
            List of Submission objects
        """
        self.logger.info(f"Extracting submissions from {self.zip_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process extracted content
            self._process_extracted_content(temp_dir)
        
        self.logger.info(f"Processed {len(self.submissions)} submissions")
        return self.submissions
    
    def _process_extracted_content(self, extract_path: str):
        """Process the extracted content and identify submissions"""
        extract_dir = Path(extract_path)
        
        # Look for submission folders/files
        items = list(extract_dir.iterdir())
        
        if len(items) == 1 and items[0].is_dir():
            # Single top-level directory - look inside
            self._process_directory_structure(items[0])
        else:
            # Multiple items at root level
            for item in items:
                if item.is_dir():
                    # Skip macOS metadata directories
                    if item.name.startswith('__MACOSX'):
                        continue
                    # Always use directory structure processing for directories
                    self._process_directory_structure(item)
                elif item.suffix in self.supported_extensions:
                    # Single code file submission
                    self._process_single_file_submission(item, extract_dir)
                elif item.suffix == '.zip':
                    # Handle ZIP files at root level
                    self._process_nested_zip(item)
    
    def _process_directory_structure(self, base_dir: Path):
        """Process directory structure to find submissions"""
        # Look for student submission folders
        for item in base_dir.iterdir():
            if item.is_dir():
                # Check if this looks like a student submission
                if self._is_submission_folder(item):
                    self._process_submission_folder(item)
                else:
                    # Recurse into subdirectories
                    self._process_directory_structure(item)
            elif item.suffix == '.zip':
                # Handle nested ZIP files
                self._process_nested_zip(item)
            elif item.suffix in self.supported_extensions:
                # Code file at this level
                self._process_single_file_submission(item, base_dir)
    
    def _process_nested_zip(self, zip_path: Path):
        """Process a nested ZIP file (student submission)"""
        self.logger.debug(f"Processing nested ZIP: {zip_path.name}")
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_extract_dir = Path(temp_dir) / f"nested_{zip_path.stem}"
            temp_extract_dir.mkdir(exist_ok=True)
            
            try:
                # Extract the nested ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # Process the extracted contents as a submission folder
                self._process_submission_folder(temp_extract_dir)
                
            except Exception as e:
                self.logger.warning(f"Failed to process nested ZIP {zip_path.name}: {e}")
    
    def _is_submission_folder(self, folder: Path) -> bool:
        """Check if a folder contains a submission"""
        # Look for supported code files (Python, Java, etc.)
        for ext in self.supported_extensions:
            code_files = list(folder.glob(f'**/*{ext}'))
            if code_files:
                return True
        
        # Look for common submission indicators
        submission_indicators = [
            'main.py', 'solution.py', 'assignment.py',
            'Main.java', 'Solution.java', 'App.java',
            'src', 'code', 'submission'
        ]
        
        for item in folder.iterdir():
            if item.name.lower() in [s.lower() for s in submission_indicators]:
                return True
        
        return False
    
    def _process_submission_folder(self, folder_path: Path):
        """Process individual submission folder"""
        self.logger.debug(f"Processing submission folder: {folder_path.name}")
        
        code_files = {}
        score = 0.0
        feedback = ""
        metadata = {}
        
        # Recursively find all files
        for file_path in folder_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            relative_path = file_path.relative_to(folder_path)
            filename = str(relative_path)
            
            # Process different file types
            if file_path.suffix in self.supported_extensions:
                # Code file (Python, Java, etc.)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_files[filename] = f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")
                    
            elif self._is_feedback_file(file_path):
                # Feedback/grading file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        feedback = content
                        
                        # Try to extract score
                        extracted_score = self._extract_score_from_text(content)
                        if extracted_score is not None:
                            score = extracted_score
                            
                except Exception as e:
                    self.logger.warning(f"Failed to read feedback file {file_path}: {e}")
                    
            # elif file_path.suffix == '.json':
            #     # JSON metadata file
            #     try:
            #         with open(file_path, 'r', encoding='utf-8') as f:
            #             json_data = json.load(f)
            #             metadata.update(json_data)
                        
            #             # Look for score in JSON
            #             if 'score' in json_data:
            #                 score = float(json_data['score'])
            #             elif 'grade' in json_data:
            #                 score = float(json_data['grade'])
                            
            #     except Exception as e:
            #         self.logger.warning(f"Failed to read JSON file {file_path}: {e}")
        
        # Create submission if we found code files
        if code_files:
            submission = Submission(
                assignment_id=self.assignment_id,
                file_name=folder_path.name,
                code_files=code_files,
                score=score,
                feedback=feedback,
                metadata=metadata
            )
            self.submissions.append(submission)
        else:
            # Create more descriptive warning message
            supported_exts = ', '.join(self.supported_extensions)
            self.logger.warning(f"No supported code files ({supported_exts}) found in {folder_path.name}")
    
    def _process_single_file_submission(self, file_path: Path, base_dir: Path):
        """Process a single code file as a submission"""
        # Check if it's a supported file type
        if file_path.suffix not in self.supported_extensions:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            code_files = {file_path.name: content}
            
            submission = Submission(
                assignment_id=self.assignment_id,
                file_name=file_path.stem,
                code_files=code_files,
                score=0.0,  # No score available for single files
                feedback="",
                metadata={'type': 'single_file'}
            )
            self.submissions.append(submission)
            
        except Exception as e:
            self.logger.warning(f"Failed to process single file {file_path}: {e}")
    
    def _is_feedback_file(self, file_path: Path) -> bool:
        """Check if a file contains feedback/grading information"""
        feedback_indicators = [
            'feedback', 'grade', 'score', 'evaluation', 
            'comments', 'review', 'assessment'
        ]
        
        filename_lower = file_path.name.lower()
        return any(indicator in filename_lower for indicator in feedback_indicators)
    
    def _extract_score_from_text(self, text: str) -> Optional[float]:
        """Extract numeric score from text using various patterns"""
        # Common score patterns
        score_patterns = [
            r'score[:\s]*(\d+(?:\.\d+)?)',
            r'grade[:\s]*(\d+(?:\.\d+)?)',
            r'points?[:\s]*(\d+(?:\.\d+)?)',
            r'mark[:\s]*(\d+(?:\.\d+)?)',
            r'total[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[/\\]\s*\d+',  # X/Y format
            r'(\d+(?:\.\d+)?)\s*%',  # Percentage
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def add_submission_with_scores(self, submission_data: Dict[str, Any]) -> Submission:
        """
        Add a submission with predefined scores and feedback.
        
        Args:
            submission_data: Dictionary containing submission information
            
        Returns:
            Created Submission object
        """
        required_fields = ['file_name', 'code_files']
        for field in required_fields:
            if field not in submission_data:
                raise ValueError(f"Missing required field: {field}")
        
        submission = Submission(
            assignment_id=submission_data.get('assignment_id', self.assignment_id),
            file_name=submission_data['file_name'],
            code_files=submission_data['code_files'],
            score=submission_data.get('score', 0.0),
            feedback=submission_data.get('feedback', ''),
            metadata=submission_data.get('metadata', {})
        )
        
        self.submissions.append(submission)
        return submission
    
    def get_submission_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed submissions"""
        if not self.submissions:
            return {
                'total_submissions': 0,
                'avg_score': 0.0,
                'score_range': (0.0, 0.0),
                'avg_files_per_submission': 0.0,
                'total_code_files': 0
            }
        
        scores = [s.score for s in self.submissions if s.score > 0]
        file_counts = [len(s.code_files) for s in self.submissions]
        
        return {
            'total_submissions': len(self.submissions),
            'avg_score': np.mean(scores) if scores else 0.0,
            'score_range': (min(scores), max(scores)) if scores else (0.0, 0.0),
            'submissions_with_scores': len(scores),
            'avg_files_per_submission': np.mean(file_counts),
            'total_code_files': sum(file_counts),
            'assignment_id': self.assignment_id
        }
    
    def save_submissions_metadata(self, filepath: str):
        """Save submission metadata to JSON file"""
        metadata = {
            'assignment_id': self.assignment_id,
            'zip_path': str(self.zip_path),
            'statistics': self.get_submission_statistics(),
            'submissions': [
                {
                    'file_name': sub.file_name,
                    'score': sub.score,
                    'feedback_length': len(sub.feedback),
                    'num_code_files': len(sub.code_files),
                    'code_file_names': list(sub.code_files.keys()),
                    'metadata': sub.metadata
                }
                for sub in self.submissions
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Submission metadata saved to {filepath}")
    
    def filter_submissions(self, min_score: Optional[float] = None, 
                          has_feedback: bool = False,
                          min_files: int = 1) -> List[Submission]:
        """
        Filter submissions based on criteria.
        
        Args:
            min_score: Minimum score requirement
            has_feedback: Whether submission must have feedback
            min_files: Minimum number of code files
            
        Returns:
            Filtered list of submissions
        """
        filtered = []
        
        for submission in self.submissions:
            # Check criteria
            if min_score is not None and submission.score < min_score:
                continue
                
            if has_feedback and not submission.feedback.strip():
                continue
                
            if len(submission.code_files) < min_files:
                continue
            
            filtered.append(submission)
        
        self.logger.info(f"Filtered {len(self.submissions)} -> {len(filtered)} submissions")
        return filtered 