"""
Report Processor

Handles processing of report-based submissions (PDFs/images) instead of code submissions.
Used when submission_is="report" in the project configuration.

Author: Auto-generated
"""

import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.logger import get_logger


class ReportProcessor:
    """
    Processes report-based submissions containing PDFs and images.
    
    This class handles extraction and validation of report files from ZIP submissions
    when the assignment is configured for report evaluation instead of code evaluation.
    """
    
    def __init__(self):
        """Initialize the report processor."""
        self.logger = get_logger(__name__)
        
        # Supported report file extensions
        self.supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    def process_report_submission(self, zip_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Process a report-based submission from a ZIP file.
        
        Args:
            zip_path: Path to the ZIP file containing report files
            temp_dir: Temporary directory for extraction
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            Exception: If processing fails
        """
        self.logger.info(f"Processing report submission: {zip_path.name}")
        
        try:
            # Extract ZIP file
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find report files
            report_files = self._find_report_files(extract_dir)
            
            if not report_files:
                raise Exception("No supported report files found in submission. Supported formats: PDF, JPG, JPEG, PNG, GIF, WEBP")
            
            self.logger.info(f"Found {len(report_files)} report file(s)")
            for file_path in report_files:
                self.logger.info(f"  - {file_path.name} ({file_path.suffix})")
            
            return {
                'success': True,
                'report_files': report_files,
                'file_count': len(report_files),
                'project_name': zip_path.stem,
                'submission_type': 'report'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process report submission: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'report_files': [],
                'file_count': 0,
                'project_name': zip_path.stem,
                'submission_type': 'report'
            }
    
    def _find_report_files(self, directory: Path) -> List[Path]:
        """
        Recursively find report files in the directory.
        
        Args:
            directory: Directory to search in
            
        Returns:
            List of paths to report files
        """
        report_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                report_files.append(file_path)
        
        return sorted(report_files)
    
    def validate_report_files(self, report_files: List[Path]) -> Dict[str, Any]:
        """
        Validate report files for size and accessibility.
        
        Args:
            report_files: List of report file paths
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size': 0,
            'warnings': []
        }
        
        max_file_size = 50 * 1024 * 1024  # 50MB per file
        max_total_size = 200 * 1024 * 1024  # 200MB total
        
        for file_path in report_files:
            try:
                file_size = file_path.stat().st_size
                
                if file_size > max_file_size:
                    validation_results['invalid_files'].append({
                        'path': file_path,
                        'reason': f'File too large: {file_size / (1024*1024):.1f}MB (max: 50MB)'
                    })
                    continue
                
                validation_results['valid_files'].append(file_path)
                validation_results['total_size'] += file_size
                
            except Exception as e:
                validation_results['invalid_files'].append({
                    'path': file_path,
                    'reason': f'Cannot access file: {str(e)}'
                })
        
        if validation_results['total_size'] > max_total_size:
            validation_results['warnings'].append(
                f"Total file size is large: {validation_results['total_size'] / (1024*1024):.1f}MB. "
                f"Processing may be slower."
            )
        
        return validation_results 