"""
AutoGrader Core

Main autograder class that orchestrates the entire grading process.
This is the central component that coordinates all other modules.

Author: Auto-generated
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import Settings
from ..llm.factory import LLMFactory
from ..prompts.prompt_manager import PromptManager
from ..repomix.processor import RepomixProcessor
from ..utils.logger import get_logger


class AutoGrader:
    """
    Main AutoGrader class that coordinates the grading process.
    
    This class implements the Facade pattern, providing a simple interface
    to the complex grading subsystem.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the AutoGrader with configuration settings.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.llm_provider = self._initialize_llm_provider()
        self.prompt_manager = PromptManager(Path("prompts"))
        self.repomix_processor = RepomixProcessor(
            max_tokens=settings.max_tokens,
            use_compression=settings.use_compression,
            remove_comments=settings.remove_comments
        )
        
        self.logger.info(f"AutoGrader initialized with {self.llm_provider}")
    
    def _initialize_llm_provider(self):
        """Initialize the LLM provider based on settings."""
        try:
            llm_config = self.settings.get_llm_config()
            provider = LLMFactory.create_provider(llm_config)
            self.logger.info(f"LLM provider initialized: {provider}")
            return provider
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {str(e)}")
            raise
    
    def _find_zip_files(self) -> List[Path]:
        """
        Find all ZIP files in the input directory.
        
        Returns:
            List of paths to ZIP files
        """
        zip_files = list(self.settings.input_folder.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} ZIP file(s) to process")
        return zip_files
    
    def _process_single_assignment(self, zip_path: Path) -> Dict[str, Any]:
        """
        Process a single assignment ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            Exception: If processing fails
        """
        self.logger.info(f"Processing assignment: {zip_path.name}")
        
        # Create temporary directory for this assignment
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                # Step 1: Process codebase with repomix
                self.logger.info("Step 1: Processing codebase with repomix")
                repomix_result = self.repomix_processor.process_codebase(zip_path, temp_dir)
                
                # Log processing results
                self.logger.info(f"Repomix processing complete:")
                self.logger.info(f"  - Token count: {repomix_result['token_count']}")
                self.logger.info(f"  - Compressed: {repomix_result['compressed']}")
                self.logger.info(f"  - Within limit: {repomix_result['within_limit']}")
                
                # Step 2: Create prompt
                self.logger.info("Step 2: Creating evaluation prompt")
                prompt = self.prompt_manager.create_final_prompt(repomix_result['content'])
                
                # Log prompt stats
                prompt_tokens = self.llm_provider.count_tokens(prompt)
                self.logger.info(f"Final prompt token count: {prompt_tokens}")
                
                # Check if prompt exceeds model limits
                max_model_tokens = self.llm_provider.get_max_tokens()
                if prompt_tokens > max_model_tokens:
                    self.logger.warning(
                        f"Prompt ({prompt_tokens} tokens) exceeds model limit "
                        f"({max_model_tokens} tokens). Results may be truncated."
                    )
                
                # Step 3: Generate evaluation
                self.logger.info("Step 3: Generating LLM evaluation")
                evaluation = self.llm_provider.generate_response(prompt)
                
                # Step 4: Save results
                self.logger.info("Step 4: Saving evaluation results")
                output_file = self._save_evaluation(zip_path, evaluation, repomix_result)
                
                return {
                    'filename': zip_path.name,
                    'output_file': output_file.name,
                    'success': True,
                    'token_count': repomix_result['token_count'],
                    'compressed': repomix_result['compressed'],
                    'project_name': repomix_result['project_name'],
                    'evaluation_length': len(evaluation)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path.name}: {str(e)}")
                raise
    
    def _save_evaluation(self, zip_path: Path, evaluation: str, 
                        repomix_result: Dict[str, Any]) -> Path:
        """
        Save the evaluation results to the output directory.
        
        Args:
            zip_path: Original ZIP file path
            evaluation: LLM evaluation text
            repomix_result: Repomix processing results
            
        Returns:
            Path to the saved evaluation file
        """
        # Create output filename based on input filename
        base_name = zip_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_evaluation_{timestamp}.txt"
        output_path = self.settings.output_folder / output_filename
        
        # Create evaluation report
        report = self._create_evaluation_report(
            zip_path, evaluation, repomix_result
        )
        
        # Save to file
        output_path.write_text(report, encoding='utf-8')
        self.logger.info(f"Evaluation saved to: {output_path}")
        
        return output_path
    
    def _create_evaluation_report(self, zip_path: Path, evaluation: str,
                                 repomix_result: Dict[str, Any]) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            zip_path: Original ZIP file path
            evaluation: LLM evaluation text
            repomix_result: Repomix processing results
            
        Returns:
            Formatted evaluation report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# AutoGrader Evaluation Report

**Assignment:** {zip_path.name}
**Project Name:** {repomix_result['project_name']}
**Evaluation Date:** {timestamp}
**LLM Provider:** {self.llm_provider}

## Processing Statistics

- **Original Token Count:** {repomix_result['token_count']:,}
- **Compression Applied:** {'Yes' if repomix_result['compressed'] else 'No'}
- **Within Token Limit:** {'Yes' if repomix_result['within_limit'] else 'No'}
- **Token Limit:** {self.settings.max_tokens:,}

## Evaluation Results

{evaluation}

---
*Generated by AutoGrader v1.0*
"""
        return report
    
    def process_assignments(self) -> List[Dict[str, Any]]:
        """
        Process all assignments in the input directory.
        
        Returns:
            List of processing results for each assignment
        """
        self.logger.info("Starting assignment processing")
        
        # Find ZIP files to process
        zip_files = self._find_zip_files()
        
        if not zip_files:
            self.logger.info("No ZIP files found to process")
            return []
        
        results = []
        
        for zip_path in zip_files:
            try:
                result = self._process_single_assignment(zip_path)
                results.append(result)
                
                # Clean up processed ZIP file
                self._cleanup_processed_file(zip_path)
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path.name}: {str(e)}")
                # Add failed result
                results.append({
                    'filename': zip_path.name,
                    'success': False,
                    'error': str(e)
                })
        
        self.logger.info(f"Processing complete. {len(results)} assignments processed")
        return results
    
    def _cleanup_processed_file(self, zip_path: Path) -> None:
        """
        Clean up the processed ZIP file.
        
        Args:
            zip_path: Path to the ZIP file to clean up
        """
        try:
            zip_path.unlink()
            self.logger.info(f"Cleaned up processed file: {zip_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up {zip_path.name}: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the autograder.
        
        Returns:
            Dictionary containing status information
        """
        zip_files = self._find_zip_files()
        
        return {
            'llm_provider': str(self.llm_provider),
            'max_tokens': self.settings.max_tokens,
            'pending_assignments': len(zip_files),
            'input_folder': str(self.settings.input_folder),
            'output_folder': str(self.settings.output_folder)
        } 