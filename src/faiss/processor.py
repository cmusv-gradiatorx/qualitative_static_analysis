"""
Submission Processing

This module handles extraction of submissions from individual ZIP files organized in task folders
and processing of code files along with metadata (scores, feedback).

Author: Auto-generated
"""

import os
import zipfile
import tempfile
import shutil
import re
import json
import subprocess
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


class TaskFolderProcessor:
    """Process individual student ZIP files from task folders"""
    
    def __init__(self, task_folder_path: str, assignment_id: Optional[str] = None):
        """
        Initialize task folder processor.
        
        Args:
            task_folder_path: Path to the task folder containing individual ZIP files
            assignment_id: Optional assignment ID (will be inferred if not provided)
        """
        self.task_folder_path = Path(task_folder_path)
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
        
        # Validate task folder exists
        if not self.task_folder_path.exists():
            raise FileNotFoundError(f"Task folder not found: {self.task_folder_path}")
    
    def _infer_assignment_id(self) -> str:
        """Infer assignment ID from task folder name"""
        folder_name = self.task_folder_path.name
        
        # Try to extract task patterns
        patterns = [
            r'task(\d+)_(.+)',  # task1_SOLID
            r'task(\d+)',       # task1
            r'assignment(\d+)',  # assignment1
            r'hw(\d+)',         # hw1
            r'project(\d+)',    # project1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                return folder_name
        
        # Fallback to folder name
        return folder_name
    
    def extract_all_submissions(self) -> List[Submission]:
        """
        Extract all submissions from individual ZIP files in the task folder.
        
        Returns:
            List of Submission objects
        """
        self.logger.info(f"Processing task folder: {self.task_folder_path}")
        
        # Find all ZIP files in the task folder
        zip_files = list(self.task_folder_path.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} ZIP files in {self.task_folder_path.name}")
        
        if not zip_files:
            self.logger.warning(f"No ZIP files found in {self.task_folder_path}")
            return []
        
        # Process each ZIP file as an individual submission
        for zip_file in zip_files:
            try:
                self.logger.debug(f"Processing ZIP file: {zip_file.name}")
                submission = self._process_individual_zip(zip_file)
                if submission:
                    self.submissions.append(submission)
            except Exception as e:
                self.logger.error(f"Error processing {zip_file.name}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(self.submissions)} submissions from {self.assignment_id}")
        return self.submissions
    
    def _process_individual_zip(self, zip_file: Path) -> Optional[Submission]:
        """
        Process an individual student ZIP file.
        
        Args:
            zip_file: Path to the ZIP file
            
        Returns:
            Submission object or None if processing failed
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract ZIP file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Extract student ID from filename
                student_id = self._extract_student_id(zip_file.name)
                
                # Process extracted content
                code_files = {}
                score = 0.0
                feedback = ""
                metadata = {'zip_file': zip_file.name, 'student_id': student_id}
                
                # Recursively find all files
                for root, dirs, files in os.walk(temp_dir):
                    # Skip macOS metadata directories
                    if '__MACOSX' in root:
                        continue
                        
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(temp_dir)
                        filename = str(relative_path)
                        
                        # Process different file types
                        if file_path.suffix in self.supported_extensions:
                            # Code file (Java, Python, etc.)
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
                
                # Create submission if we found code files
                if code_files:
                    submission = Submission(
                        assignment_id=self.assignment_id,
                        file_name=student_id,
                        code_files=code_files,
                        score=score,
                        feedback=feedback,
                        metadata=metadata
                    )
                    return submission
                else:
                    # Create more descriptive warning message
                    supported_exts = ', '.join(self.supported_extensions)
                    self.logger.warning(f"No supported code files ({supported_exts}) found in {zip_file.name}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Failed to process ZIP file {zip_file.name}: {str(e)}")
                return None
    
    def _extract_student_id(self, zip_filename: str) -> str:
        """Extract student ID from ZIP filename"""
        # Remove .zip extension
        base_name = zip_filename.replace('.zip', '').replace('.rar', '')
        
        # Common patterns for student submissions
        patterns = [
            r'^([a-zA-Z]+[a-zA-Z0-9]*)',  # Start with letters, then letters/numbers
            r'([a-zA-Z0-9]+)_\d+_\d+',   # student_id_number_number format
            r'^([^_]+)',                  # Everything before first underscore
        ]
        
        for pattern in patterns:
            match = re.search(pattern, base_name)
            if match:
                return match.group(1)
        
        # Fallback to base filename
        return base_name
    
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


class MultiFolderProcessor:
    """Process multiple task folders to extract submissions"""
    
    def __init__(self, data_root_path: str):
        """
        Initialize multi-folder processor.
        
        Args:
            data_root_path: Path to the root data directory containing task folders
        """
        self.data_root_path = Path(data_root_path)
        self.logger = get_logger(__name__)
        
        if not self.data_root_path.exists():
            raise FileNotFoundError(f"Data root path not found: {self.data_root_path}")
    
    def find_task_folders(self) -> List[Path]:
        """Find all task folders in the data root directory"""
        task_folders = []
        
        for item in self.data_root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it looks like a task folder
                if any(pattern in item.name.lower() for pattern in ['task', 'assignment', 'hw', 'project']):
                    task_folders.append(item)
        
        self.logger.info(f"Found {len(task_folders)} task folders: {[f.name for f in task_folders]}")
        return task_folders
    
    def process_all_tasks(self) -> Dict[str, List[Submission]]:
        """
        Process all task folders and return submissions grouped by assignment.
        
        Returns:
            Dictionary mapping assignment_id to list of submissions
        """
        task_folders = self.find_task_folders()
        all_submissions = {}
        
        for task_folder in task_folders:
            try:
                processor = TaskFolderProcessor(str(task_folder))
                submissions = processor.extract_all_submissions()
                
                assignment_id = processor.assignment_id
                all_submissions[assignment_id] = submissions
                
                self.logger.info(f"Processed {len(submissions)} submissions for {assignment_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to process task folder {task_folder.name}: {str(e)}")
                continue
        
        return all_submissions
    
    def get_overall_statistics(self, submissions_by_assignment: Dict[str, List[Submission]]) -> Dict[str, Any]:
        """Get overall statistics across all assignments"""
        total_submissions = sum(len(subs) for subs in submissions_by_assignment.values())
        total_assignments = len(submissions_by_assignment)
        
        assignment_stats = {}
        for assignment_id, submissions in submissions_by_assignment.items():
            if submissions:
                # Calculate statistics directly without creating a dummy processor
                scores = [s.score for s in submissions if s.score > 0]
                file_counts = [len(s.code_files) for s in submissions]
                
                assignment_stats[assignment_id] = {
                    'total_submissions': len(submissions),
                    'avg_score': np.mean(scores) if scores else 0.0,
                    'score_range': (min(scores), max(scores)) if scores else (0.0, 0.0),
                    'submissions_with_scores': len(scores),
                    'avg_files_per_submission': np.mean(file_counts),
                    'total_code_files': sum(file_counts),
                    'assignment_id': assignment_id
                }
        
        return {
            'total_assignments': total_assignments,
            'total_submissions': total_submissions,
            'assignment_stats': assignment_stats,
            'assignments': list(submissions_by_assignment.keys())
        }


# Backward compatibility class
class SubmissionProcessor(TaskFolderProcessor):
    """Backward compatibility wrapper for TaskFolderProcessor"""
    
    def __init__(self, path: str, assignment_id: Optional[str] = None):
        """
        Initialize submission processor with backward compatibility.
        
        Args:
            path: Path to either a ZIP file (old format) or task folder (new format)
            assignment_id: Optional assignment ID
        """
        path_obj = Path(path)
        
        if path_obj.is_file() and path_obj.suffix == '.zip':
            # Old format: single ZIP file
            self.logger = get_logger(__name__)
            self.logger.warning("Single ZIP file detected - using legacy mode")
            self._init_legacy_mode(path, assignment_id)
        else:
            # New format: task folder
            super().__init__(path, assignment_id)
    
    def _init_legacy_mode(self, zip_path: str, assignment_id: Optional[str] = None):
        """Initialize in legacy mode for single ZIP files"""
        # This would implement the old single ZIP processing logic
        # For now, raise an error to encourage using the new format
        raise NotImplementedError(
            "Legacy single ZIP file processing is deprecated. "
            "Please organize your data in task folders with individual ZIP files."
        )
    
    def extract_submissions(self) -> List[Submission]:
        """Extract submissions - delegates to new method"""
        return self.extract_all_submissions()


class RepomixProcessor:
    """
    Processor that uses repomix to process codebases into a single text file.
    Handles ZIP file extraction and repomix processing with various configuration options.
    """
    
    def __init__(self, 
                 max_tokens: int = 128000,
                 use_compression: bool = True,
                 remove_comments: bool = False,
                 ignore_patterns: Optional[List[str]] = None,
                 keep_patterns: Optional[List[str]] = None,
                 max_file_size: Optional[int] = None):
        """
        Initialize the RepomixProcessor.
        
        Args:
            max_tokens: Maximum token limit for repomix processing
            use_compression: Whether to use compression in repomix
            remove_comments: Whether to remove comments in repomix
            ignore_patterns: List of file/directory patterns to ignore
            keep_patterns: List of file patterns to keep
            max_file_size: Maximum file size in bytes for filtering
        """
        self.logger = get_logger(__name__)
        self.max_tokens = max_tokens
        self.use_compression = use_compression
        self.remove_comments = remove_comments
        self.ignore_patterns = ignore_patterns or []
        self.keep_patterns = keep_patterns or []
        self.max_file_size = max_file_size
        
        # Check if repomix is available
        self._check_repomix_availability()
    
    def _check_repomix_availability(self):
        """Check if repomix is available via npx."""
        try:
            # Use npx to check repomix availability
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
    
    def process_codebase(self, zip_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Process a codebase ZIP file using repomix.
        
        Args:
            zip_path: Path to the ZIP file containing the codebase
            temp_dir: Temporary directory for extraction
            
        Returns:
            Dictionary containing processed content and metadata
        """
        try:
            # Extract ZIP file
            extract_dir = temp_dir / "extracted"
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
            output_file = temp_dir / "repomix_output.txt"
            repomix_result = self._run_repomix(project_dir, output_file)
            
            # Read the processed content
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                raise RuntimeError("Repomix did not generate output file")
            
            # Extract project name from ZIP filename
            project_name = zip_path.stem.split('_')[0] if '_' in zip_path.stem else zip_path.stem
            
            return {
                'content': content,
                'token_count': repomix_result.get('token_count', 0),
                'compressed': repomix_result.get('compressed', False),
                'within_limit': repomix_result.get('within_limit', True),
                'project_name': project_name,
                'filtered_files_count': repomix_result.get('filtered_files_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process codebase {zip_path}: {str(e)}")
            raise
    
    def _run_repomix(self, project_dir: Path, output_file: Path) -> Dict[str, Any]:
        """
        Run repomix on the project directory.
        
        Args:
            project_dir: Directory containing the project files
            output_file: Output file for repomix
            
        Returns:
            Dictionary containing repomix execution metadata
        """
        try:
            # Build repomix command
            cmd = ['npx', 'repomix', str(project_dir), '--output', str(output_file)]
            
            # Add compression options
            if self.use_compression:
                cmd.append('--compress')
                cmd.append('--remove-empty-lines')  # Additional compression
            
            if self.remove_comments:
                cmd.append('--remove-comments')
            
            # Add token counting with encoding
            cmd.extend(['--token-count-encoding', 'cl100k_base'])
            
            # Add ignore patterns
            if self.ignore_patterns:
                ignore_patterns_str = ','.join(self.ignore_patterns)
                cmd.extend(['--ignore', ignore_patterns_str])
            
            # Add keep patterns (include patterns)
            if self.keep_patterns:
                include_patterns_str = ','.join(self.keep_patterns)
                cmd.extend(['--include', include_patterns_str])
            
            # Run repomix
            self.logger.debug(f"Running repomix command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Repomix failed: {result.stderr}")
                raise RuntimeError(f"Repomix execution failed: {result.stderr}")
            
            # Parse repomix output for metadata
            metadata = self._parse_repomix_output(result.stdout)
            
            return metadata
            
        except subprocess.TimeoutExpired:
            self.logger.error("Repomix execution timed out")
            raise RuntimeError("Repomix execution timed out")
        except Exception as e:
            self.logger.error(f"Error running repomix: {str(e)}")
            raise
    
    def _parse_repomix_output(self, stdout: str) -> Dict[str, Any]:
        """
        Parse repomix stdout to extract metadata.
        
        Args:
            stdout: Standard output from repomix command
            
        Returns:
            Dictionary containing parsed metadata
        """
        metadata = {
            'token_count': 0,
            'compressed': False,
            'within_limit': True,
            'filtered_files_count': 0
        }
        
        try:
            # Look for token count in output
            token_match = re.search(r'Token count:\s*(\d+)', stdout)
            if token_match:
                metadata['token_count'] = int(token_match.group(1))
            
            # Look for compression indicator
            if 'compressed' in stdout.lower():
                metadata['compressed'] = True
            
            # Look for limit exceeded indicator
            if 'limit exceeded' in stdout.lower() or 'truncated' in stdout.lower():
                metadata['within_limit'] = False
            
            # Look for file count
            file_match = re.search(r'(\d+)\s+files?\s+processed', stdout)
            if file_match:
                metadata['filtered_files_count'] = int(file_match.group(1))
            
        except Exception as e:
            self.logger.warning(f"Failed to parse repomix output: {e}")
        
        return metadata 