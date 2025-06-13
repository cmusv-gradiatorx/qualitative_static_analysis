"""
Submission Processor

Clean processor for handling student submissions organized in task folders.
Supports the data structure: task_folder/studentname_....zip

Author: Auto-generated
"""

import zipfile
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ...utils.logger import get_logger


@dataclass
class Submission:
    """Data class representing a student submission"""
    student_name: str
    zip_path: Path
    java_files: Dict[str, str]  # filename -> content
    all_files: Dict[str, str]   # filename -> content (all file types)
    metadata: Dict[str, Any]
    
    def get_file_count(self) -> int:
        """Get total number of files in submission"""
        return len(self.all_files)
    
    def get_java_file_count(self) -> int:
        """Get number of Java files in submission"""
        return len(self.java_files)
    
    def has_java_files(self) -> bool:
        """Check if submission contains Java files"""
        return len(self.java_files) > 0


class SubmissionProcessor:
    """
    Processes student submissions from task folders.
    
    Handles the data structure: task_folder/studentname_....zip
    Extracts and organizes submission content for clustering.
    """
    
    def __init__(self, task_folder: str):
        """
        Initialize processor for a specific task folder.
        
        Args:
            task_folder: Path to task folder containing ZIP files
        """
        self.task_folder = Path(task_folder)
        self.logger = get_logger(__name__)
        
        if not self.task_folder.exists():
            raise FileNotFoundError(f"Task folder not found: {self.task_folder}")
        
        self.logger.info(f"Initialized submission processor for: {self.task_folder}")
    
    def extract_submissions(self, 
                          java_only: bool = False,
                          max_file_size: int = 5_000_000) -> List[Submission]:
        """
        Extract all submissions from the task folder.
        
        Args:
            java_only: If True, only process submissions with Java files
            max_file_size: Maximum file size in bytes to process
            
        Returns:
            List of Submission objects
        """
        # Find all ZIP files in task folder
        zip_files = list(self.task_folder.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} ZIP files in {self.task_folder.name}")
        
        if not zip_files:
            self.logger.warning(f"No ZIP files found in {self.task_folder}")
            return []
        
        submissions = []
        
        for zip_file in zip_files:
            try:
                submission = self._process_zip_file(zip_file, max_file_size)
                
                # Filter by Java files if requested
                if java_only and not submission.has_java_files():
                    self.logger.debug(f"Skipping {submission.student_name} - no Java files")
                    continue
                
                submissions.append(submission)
                self.logger.debug(f"Processed submission: {submission.student_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_file.name}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(submissions)} submissions")
        return submissions
    
    def _process_zip_file(self, zip_path: Path, max_file_size: int) -> Submission:
        """
        Process a single ZIP file submission.
        
        Args:
            zip_path: Path to ZIP file
            max_file_size: Maximum file size to process
            
        Returns:
            Submission object
        """
        student_name = self._extract_student_name(zip_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract ZIP file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
            except zipfile.BadZipFile:
                raise ValueError(f"Invalid ZIP file: {zip_path}")
            
            # Process extracted files
            java_files = {}
            all_files = {}
            
            # Recursively find all files
            for file_path in temp_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Skip macOS metadata and hidden files
                if self._should_skip_file(file_path):
                    continue
                
                # Check file size
                if file_path.stat().st_size > max_file_size:
                    self.logger.warning(f"Skipping large file: {file_path.name} ({file_path.stat().st_size} bytes)")
                    continue
                
                # Get relative path from temp directory
                relative_path = file_path.relative_to(temp_path)
                filename = str(relative_path)
                
                try:
                    # Read file content
                    content = self._read_file_content(file_path)
                    all_files[filename] = content
                    
                    # Check if it's a Java file
                    if file_path.suffix.lower() == '.java':
                        java_files[filename] = content
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read file {filename}: {e}")
                    continue
            
            # Create metadata
            metadata = {
                'zip_name': zip_path.name,
                'zip_path': str(zip_path),
                'zip_size_bytes': zip_path.stat().st_size,
                'total_files': len(all_files),
                'java_files_count': len(java_files),
                'task_folder': self.task_folder.name
            }
            
            return Submission(
                student_name=student_name,
                zip_path=zip_path,
                java_files=java_files,
                all_files=all_files,
                metadata=metadata
            )
    
    def _extract_student_name(self, zip_path: Path) -> str:
        """
        Extract student name from ZIP filename.
        
        Expected format: studentname_otherthings....zip
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            Student name
        """
        # Remove extension
        base_name = zip_path.stem
        
        # Extract name before first underscore
        if '_' in base_name:
            student_name = base_name.split('_')[0]
        else:
            student_name = base_name
        
        # Clean up the name (remove special characters)
        student_name = re.sub(r'[^a-zA-Z0-9]', '', student_name)
        
        return student_name or 'unknown'
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '__MACOSX',
            '.DS_Store',
            'Thumbs.db',
            '.git',
            '.svn',
            '__pycache__',
            '*.pyc',
            '*.class',
            '*.o',
            '*.obj'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str or file_path.name.startswith('.') 
                  for pattern in skip_patterns)
    
    def _read_file_content(self, file_path: Path) -> str:
        """
        Read file content with encoding detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all encodings fail, read as binary and decode with errors='ignore'
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")
    
    def get_task_statistics(self, submissions: List[Submission]) -> Dict[str, Any]:
        """
        Get statistics about processed submissions.
        
        Args:
            submissions: List of submissions
            
        Returns:
            Dictionary with statistics
        """
        if not submissions:
            return {
                'task_folder': self.task_folder.name,
                'total_submissions': 0,
                'avg_files_per_submission': 0,
                'avg_java_files_per_submission': 0,
                'students_with_java': 0,
                'total_files': 0,
                'total_java_files': 0
            }
        
        total_files = sum(sub.get_file_count() for sub in submissions)
        total_java_files = sum(sub.get_java_file_count() for sub in submissions)
        students_with_java = sum(1 for sub in submissions if sub.has_java_files())
        
        return {
            'task_folder': self.task_folder.name,
            'total_submissions': len(submissions),
            'avg_files_per_submission': total_files / len(submissions),
            'avg_java_files_per_submission': total_java_files / len(submissions),
            'students_with_java': students_with_java,
            'students_with_java_percentage': (students_with_java / len(submissions)) * 100,
            'total_files': total_files,
            'total_java_files': total_java_files,
            'student_names': [sub.student_name for sub in submissions]
        }
    
    def filter_submissions(self, submissions: List[Submission], 
                         min_java_files: int = 1,
                         min_total_files: int = 1,
                         student_names: Optional[List[str]] = None) -> List[Submission]:
        """
        Filter submissions based on criteria.
        
        Args:
            submissions: List of submissions to filter
            min_java_files: Minimum number of Java files required
            min_total_files: Minimum total number of files required
            student_names: Optional list of student names to include
            
        Returns:
            Filtered list of submissions
        """
        filtered = []
        
        for submission in submissions:
            # Check Java file requirement
            if submission.get_java_file_count() < min_java_files:
                continue
            
            # Check total file requirement
            if submission.get_file_count() < min_total_files:
                continue
            
            # Check student name filter
            if student_names and submission.student_name not in student_names:
                continue
            
            filtered.append(submission)
        
        self.logger.info(f"Filtered {len(submissions)} -> {len(filtered)} submissions")
        return filtered 