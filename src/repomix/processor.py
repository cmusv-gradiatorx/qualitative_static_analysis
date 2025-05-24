"""
Repomix Processor

Handles processing of codebases using repomix with token counting and compression.
Implements logic to handle token limits and apply compression when needed.

Author: Auto-generated
"""

import subprocess
import os
import shutil
import platform
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import zipfile

from ..utils.logger import get_logger


class RepomixProcessor:
    """
    Processes codebases using repomix with intelligent token management.
    
    This class handles the extraction of ZIP files, running repomix,
    and managing token limits with compression when necessary.
    """
    
    def __init__(self, max_tokens: int = 128000, use_compression: bool = True, 
                 remove_comments: bool = False):
        """
        Initialize the repomix processor.
        
        Args:
            max_tokens: Maximum token limit for the LLM
            use_compression: Whether to use compression if token limit exceeded
            remove_comments: Whether to remove comments from code
        """
        self.max_tokens = max_tokens
        self.use_compression = use_compression
        self.remove_comments = remove_comments
        self.logger = get_logger(__name__)
        self.is_windows = platform.system().lower() == 'windows'
        
        # Check if repomix is available
        self._check_repomix_availability()
    
    def _check_repomix_availability(self) -> None:
        """Check if repomix is available via npx."""
        try:
            # On Windows, we need shell=True to properly execute npx
            result = subprocess.run(
                ['npx', 'repomix', '--version'],
                capture_output=True,
                text=True,
                timeout=30,
                shell=self.is_windows
            )
            if result.returncode == 0:
                self.logger.info(f"Repomix available: {result.stdout.strip()}")
            else:
                self.logger.warning("Repomix not found, will attempt to install on first use")
        except Exception as e:
            self.logger.warning(f"Could not check repomix availability: {str(e)}")
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> Path:
        """
        Extract ZIP file to temporary directory.
        
        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract to
            
        Returns:
            Path to the extracted directory
            
        Raises:
            Exception: If extraction fails
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the main project directory (usually the first subdirectory)
            extracted_items = list(extract_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                return extracted_items[0]
            else:
                return extract_dir
                
        except Exception as e:
            raise Exception(f"Failed to extract ZIP file {zip_path}: {str(e)}")
    
    def _run_repomix(self, project_dir: Path, output_file: Path, 
                     style: str = "plain", compressed: bool = False) -> Dict[str, Any]:
        """
        Run repomix on the project directory.
        
        Args:
            project_dir: Path to the project directory
            output_file: Path for the output file
            style: Output style (plain, xml, markdown)
            compressed: Whether to use compression
            
        Returns:
            Dictionary with processing results including token count
            
        Raises:
            Exception: If repomix execution fails
        """
        try:
            # Build repomix command
            cmd = ['npx', 'repomix', str(project_dir)]
            cmd.extend(['--output', str(output_file)])
            cmd.extend(['--style', style])
            
            # Add compression options
            if compressed:
                cmd.append('--compress')
                cmd.append('--remove-empty-lines')
            
            if self.remove_comments:
                cmd.append('--remove-comments')
            
            # Add token counting with encoding
            cmd.extend(['--token-count-encoding', 'cl100k_base'])
            
            self.logger.info(f"Running repomix: {' '.join(cmd)}")
            
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            if self.is_windows:
                env['PYTHONIOENCODING'] = 'utf-8'
            
            # Execute repomix with proper encoding handling
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=project_dir.parent,
                shell=self.is_windows,
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace problematic characters instead of failing
            )
            
            if result.returncode != 0:
                error_msg = f"Repomix failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                if result.stdout:
                    error_msg += f"\nOutput: {result.stdout}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Read the output file
            if not output_file.exists():
                raise Exception("Repomix output file was not created")
            
            content = output_file.read_text(encoding='utf-8', errors='replace')
            
            # Extract token count from repomix output (if available)
            stdout_text = result.stdout if result.stdout else ""
            token_count = self._extract_token_count(stdout_text, content)
            
            return {
                'content': content,
                'token_count': token_count,
                'compressed': compressed,
                'output_file': output_file
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Repomix execution timed out")
        except Exception as e:
            raise Exception(f"Failed to run repomix: {str(e)}")
    
    def _extract_token_count(self, stdout: str, content: str) -> int:
        """
        Extract token count from repomix output or estimate it.
        
        Args:
            stdout: Standard output from repomix
            content: The generated content
            
        Returns:
            Token count
        """
        # Try to extract from repomix stdout
        lines = stdout.split('\n')
        for line in lines:
            if 'token' in line.lower() and any(char.isdigit() for char in line):
                # Extract numbers from the line
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    return max(numbers)  # Take the largest number
        
        # Fallback: estimate based on content length
        # Rough estimation: ~4 characters per token
        estimated_tokens = len(content) // 4
        self.logger.info(f"Estimated token count: {estimated_tokens}")
        return estimated_tokens
    
    def process_codebase(self, zip_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Process a codebase ZIP file using repomix.
        
        This method handles the complete workflow:
        1. Extract ZIP file
        2. Run repomix
        3. Check token count
        4. Apply compression if needed
        5. Return processed content
        
        Args:
            zip_path: Path to the ZIP file containing the codebase
            temp_dir: Temporary directory for processing
            
        Returns:
            Dictionary containing processed content and metadata
            
        Raises:
            Exception: If processing fails
        """
        self.logger.info(f"Processing codebase: {zip_path}")
        
        try:
            # Create temporary directories
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract ZIP file
            project_dir = self._extract_zip(zip_path, extract_dir)
            self.logger.info(f"Extracted to: {project_dir}")
            
            # First attempt: normal processing
            output_file = temp_dir / "repomix_output.txt"
            result = self._run_repomix(project_dir, output_file, style="plain", compressed=False)
            
            self.logger.info(f"Initial token count: {result['token_count']}")
            
            # Check if we need compression
            if result['token_count'] > self.max_tokens and self.use_compression:
                self.logger.info("Token limit exceeded, applying compression...")
                
                # Try with compression
                compressed_output = temp_dir / "repomix_compressed.txt"
                compressed_result = self._run_repomix(
                    project_dir, compressed_output, style="plain", compressed=True
                )
                
                self.logger.info(f"Compressed token count: {compressed_result['token_count']}")
                
                if compressed_result['token_count'] <= self.max_tokens:
                    result = compressed_result
                    self.logger.info("Compression successful, using compressed version")
                else:
                    self.logger.warning(
                        f"Even compressed version ({compressed_result['token_count']} tokens) "
                        f"exceeds limit ({self.max_tokens} tokens)"
                    )
                    # Still use compressed version as it's smaller
                    result = compressed_result
            
            return {
                'content': result['content'],
                'token_count': result['token_count'],
                'compressed': result['compressed'],
                'within_limit': result['token_count'] <= self.max_tokens,
                'project_name': project_dir.name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process codebase {zip_path}: {str(e)}")
            raise
        finally:
            # Cleanup extract directory
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True) 