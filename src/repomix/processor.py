"""
Repomix Processor

Handles processing of codebases using repomix with token counting and compression.
Implements logic to handle token limits and apply compression when needed.
Supports project-specific ignore patterns, keep patterns, and file size filtering.

Author: Auto-generated
"""

import subprocess
import os
import shutil
import platform
import json
import fnmatch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import tempfile
import zipfile

from ..utils.logger import get_logger


class RepomixProcessor:
    """
    Processes codebases using repomix with intelligent token management.
    
    This class handles the extraction of ZIP files, running repomix,
    and managing token limits with compression when necessary.
    Supports project-specific filtering via ignore patterns, keep patterns, and file size limits.
    """
    
    def __init__(self, max_tokens: int = 128000, use_compression: bool = True, 
                 remove_comments: bool = False, ignore_patterns: Optional[List[str]] = None,
                 keep_patterns: Optional[List[str]] = None, max_file_size: Optional[int] = None):
        """
        Initialize the repomix processor.
        
        Args:
            max_tokens: Maximum token limit for the LLM model (e.g., 128K for GPT-4) - used for compression decisions
            use_compression: Whether to use compression if token limit exceeded
            remove_comments: Whether to remove comments from code
            ignore_patterns: List of file/directory patterns to ignore
            keep_patterns: List of file patterns to keep (only these will be included)
            max_file_size: Maximum file size in bytes for filtering (files larger than this will be ignored before processing)
        """
        self.max_tokens = max_tokens
        self.use_compression = use_compression
        self.remove_comments = remove_comments
        self.ignore_patterns = ignore_patterns or []
        self.keep_patterns = keep_patterns or []
        self.max_file_size = max_file_size
        self.logger = get_logger(__name__)
        self.is_windows = platform.system().lower() == 'windows'
        
        # Check if repomix is available
        self._check_repomix_availability()
        
        # Log the configuration
        self.logger.info(f"RepomixProcessor initialized with:")
        self.logger.info(f"  - Max tokens (LLM limit): {self.max_tokens}")
        self.logger.info(f"  - Use compression: {self.use_compression}")
        self.logger.info(f"  - Remove comments: {self.remove_comments}")
        self.logger.info(f"  - Max file size (filtering): {self.max_file_size} bytes" if self.max_file_size else "  - Max file size (filtering): unlimited")
        self.logger.info(f"  - Ignore patterns: {self.ignore_patterns}")
        self.logger.info(f"  - Keep patterns: {self.keep_patterns}")
    
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
    
    def _should_ignore_file(self, file_path: Path, project_root: Path) -> bool:
        """
        Check if a file should be ignored based on configured patterns and file size.
        
        Args:
            file_path: Path to the file to check
            project_root: Root directory of the project
            
        Returns:
            True if the file should be ignored, False otherwise
        """
        try:
            # Get relative path for pattern matching
            relative_path = file_path.relative_to(project_root)
            relative_path_str = str(relative_path)
            
            # Check file size if max_file_size is set
            if self.max_file_size and file_path.is_file():
                try:
                    if file_path.stat().st_size > self.max_file_size:
                        self.logger.debug(f"Ignoring large file: {relative_path_str} ({file_path.stat().st_size} bytes > {self.max_file_size} bytes)")
                        return True
                except OSError:
                    # If we can't stat the file, err on the side of caution and include it
                    pass
            
            # Check ignore patterns
            for pattern in self.ignore_patterns:
                if fnmatch.fnmatch(relative_path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                    self.logger.debug(f"Ignoring file due to ignore pattern '{pattern}': {relative_path_str}")
                    return True
                # Also check if any parent directory matches the pattern
                for parent in relative_path.parents:
                    if fnmatch.fnmatch(str(parent), pattern) or fnmatch.fnmatch(parent.name, pattern):
                        self.logger.debug(f"Ignoring file due to parent directory matching ignore pattern '{pattern}': {relative_path_str}")
                        return True
            
            # If keep patterns are specified, file must match at least one keep pattern
            if self.keep_patterns:
                for pattern in self.keep_patterns:
                    if fnmatch.fnmatch(relative_path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                        self.logger.debug(f"Keeping file due to keep pattern '{pattern}': {relative_path_str}")
                        return False
                # No keep pattern matched, so ignore the file
                self.logger.debug(f"Ignoring file as it doesn't match any keep pattern: {relative_path_str}")
                return True
            
            # File should be included
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking file {file_path}: {str(e)}. Including file.")
            return False

    def _create_repomix_config(self, project_dir: Path) -> Path:
        """
        Create a repomix configuration file for the project with custom ignore patterns.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Path to the created configuration file
        """
        config_file = project_dir / "repomix.config.json"
        
        # Create base configuration
        config = {
            "output": {
                "style": "plain",
                "removeComments": self.remove_comments,
                "removeEmptyLines": False,
                "showLineNumbers": False
            },
            "ignore": {
                "useGitignore": True,
                "useDefaultPatterns": True,
                "customPatterns": []
            }
        }
        
        # Add custom ignore patterns from project configuration
        if self.ignore_patterns:
            config["ignore"]["customPatterns"].extend(self.ignore_patterns)
        
        # If keep patterns are specified, we need to handle this differently
        # Since repomix doesn't have a direct "keep only" feature, we'll create include patterns
        if self.keep_patterns:
            config["include"] = self.keep_patterns
        
        # Write configuration file
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            self.logger.debug(f"Created repomix config file: {config_file}")
            return config_file
        except Exception as e:
            self.logger.warning(f"Failed to create repomix config file: {str(e)}. Using default configuration.")
            return None

    def _filter_project_files(self, project_dir: Path) -> int:
        """
        Filter project files based on configured patterns and file size limits.
        This is a fallback method for cases where repomix configuration isn't sufficient.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        if not (self.ignore_patterns or self.keep_patterns or self.max_file_size):
            return removed_count
        
        try:
            # Walk through all files in the project
            for file_path in project_dir.rglob("*"):
                if file_path.is_file() and self._should_ignore_file(file_path, project_dir):
                    try:
                        file_path.unlink()
                        removed_count += 1
                        self.logger.debug(f"Removed file: {file_path.relative_to(project_dir)}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove file {file_path}: {str(e)}")
            
            # Remove empty directories
            for dir_path in sorted(project_dir.rglob("*"), key=lambda p: len(str(p)), reverse=True):
                if dir_path.is_dir() and dir_path != project_dir:
                    try:
                        if not any(dir_path.iterdir()):  # Directory is empty
                            dir_path.rmdir()
                            self.logger.debug(f"Removed empty directory: {dir_path.relative_to(project_dir)}")
                    except Exception as e:
                        # Directory might not be empty or might have permission issues
                        pass
                        
        except Exception as e:
            self.logger.warning(f"Error during file filtering: {str(e)}")
        
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} files based on project configuration")
            
        return removed_count

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
            # Apply custom file filtering if needed
            filtered_count = self._filter_project_files(project_dir)
            
            # Create repomix configuration file
            config_file = self._create_repomix_config(project_dir)
            
            # Build repomix command
            cmd = ['npx', 'repomix', str(project_dir)]
            cmd.extend(['--output', str(output_file)])
            cmd.extend(['--style', style])
            
            # Use configuration file if created
            if config_file and config_file.exists():
                cmd.extend(['--config', str(config_file)])
            
            # Add compression options
            if compressed:
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
            
            # Clean up configuration file
            if config_file and config_file.exists():
                try:
                    config_file.unlink()
                except Exception as e:
                    self.logger.debug(f"Failed to remove config file {config_file}: {str(e)}")
            
            return {
                'content': content,
                'token_count': token_count,
                'compressed': compressed,
                'output_file': output_file,
                'filtered_files_count': filtered_count
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
        2. Apply project-specific filtering (ignore patterns, keep patterns, file size limits)
        3. Run repomix
        4. Check token count
        5. Apply compression if needed
        6. Return processed content
        
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
            if result.get('filtered_files_count', 0) > 0:
                self.logger.info(f"Filtered {result['filtered_files_count']} files based on project configuration")
            
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
                'project_name': project_dir.name,
                'filtered_files_count': result.get('filtered_files_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process codebase {zip_path}: {str(e)}")
            raise
        finally:
            # Cleanup extract directory
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True) 