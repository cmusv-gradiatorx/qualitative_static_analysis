"""
AutoGrader Core

Main autograder class that orchestrates the entire grading process.
This is the central component that coordinates all other modules.
Supports parallel LLM processing for detailed rubric evaluation with structured output.

Author: Auto-generated
"""

import os
import json
import tempfile
import shutil
import zipfile
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import Settings
from ..llm.factory import LLMFactory
from ..prompts.prompt_manager import PromptManager
from ..repomix.processor import RepomixProcessor
from ..semgrep.analyzer import SemgrepAnalyzer
from ..utils.logger import get_logger
from .output_manager import OutputManager
from .report_processor import ReportProcessor


class AutoGrader:
    """
    Main AutoGrader class that coordinates the grading process.
    
    This class implements the Facade pattern, providing a simple interface
    to the complex grading subsystem with parallel LLM processing support
    and structured output generation.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the AutoGrader with configuration settings.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Get project-specific configuration
        self.project_config = settings.get_project_config()
        
        # Create extra_logs directory
        self.extra_logs_dir = Path("extra_logs")
        self.extra_logs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.llm_provider = self._initialize_llm_provider()
        self.prompt_manager = PromptManager(settings.get_prompts_dir())
        self.output_manager = OutputManager(settings.output_folder, self.llm_provider, self.prompt_manager)
        
        # Initialize report processor for PDF/image submissions
        self.report_processor = ReportProcessor()
        
        self.repomix_processor = RepomixProcessor(
            max_tokens=settings.max_tokens,
            use_compression=settings.use_compression,
            remove_comments=settings.remove_comments,
            ignore_patterns=self.project_config.get('ignore_patterns', []),
            keep_patterns=self.project_config.get('keep_patterns', []),
            max_file_size=self.project_config.get('max_file_size')
        )
        
        # Initialize semgrep analyzer if enabled
        self.semgrep_analyzer = None
        enable_semgrep = self.project_config.get('enable_semgrep_analysis', False)
        
        if enable_semgrep:
            try:
                semgrep_rules_file = self.project_config.get('semgrep_rules_file', 'config/semgrep_rules.yaml')
                semgrep_timeout = self.project_config.get('semgrep_timeout', 300)
                
                # Convert string path to Path object
                rules_file_path = Path(semgrep_rules_file)
                
                self.semgrep_analyzer = SemgrepAnalyzer(
                    rules_file=rules_file_path,
                    timeout=semgrep_timeout
                )
                self.logger.info("Semgrep analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Semgrep analyzer: {str(e)}")
                self.logger.warning("Continuing without static analysis")
        
        self.logger.info(f"AutoGrader initialized with {self.llm_provider}")
        self.logger.info(f"Project: {settings.project_assignment}")
        self.logger.info(f"Max parallel LLM runs: {self.project_config.get('max_parallel_llm', 2)}")
        self.logger.info(f"Extra logs directory: {self.extra_logs_dir}")
        
        # Log project-specific filtering configuration
        ignore_patterns = self.project_config.get('ignore_patterns', [])
        keep_patterns = self.project_config.get('keep_patterns', [])
        max_file_size = self.project_config.get('max_file_size')
        
        if ignore_patterns or keep_patterns or max_file_size:
            self.logger.info("Project-specific file filtering enabled:")
            if ignore_patterns:
                self.logger.info(f"  - Ignore patterns: {ignore_patterns}")
            if keep_patterns:
                self.logger.info(f"  - Keep patterns: {keep_patterns}")
            if max_file_size:
                self.logger.info(f"  - Max file size: {max_file_size} bytes")
        else:
            self.logger.info("No project-specific file filtering configured")
    
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
        Process a single assignment ZIP file with parallel rubric evaluation.
        Supports both code and report submissions with multimodal LLM support.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            Exception: If processing fails
        """
        self.logger.info(f"Processing assignment: {zip_path.name}")
        
        # Get submission type from project configuration
        submission_is = self.project_config.get('submission_is', 'code')
        prompt_has_img_pdf = self.project_config.get('prompt_has_img_pdf', False)
        
        self.logger.info(f"Submission type: {submission_is}")
        self.logger.info(f"Prompt has attachments: {prompt_has_img_pdf}")
        
        # Create temporary directory for this assignment
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                # Step 1: Process submission based on type
                if submission_is == 'report':
                    self.logger.info("Step 1: Processing report submission")
                    processing_result = self.report_processor.process_report_submission(zip_path, temp_dir)
                    content_for_evaluation = f"Report files: {', '.join([f.name for f in processing_result.get('report_files', [])])}"
                else:
                    self.logger.info("Step 1: Processing codebase with repomix")
                    processing_result = self.repomix_processor.process_codebase(zip_path, temp_dir)
                    content_for_evaluation = processing_result['content']
                    
                    # Log processing results for code submissions
                    self.logger.info(f"Repomix processing complete:")
                    self.logger.info(f"  - Token count: {processing_result['token_count']}")
                    self.logger.info(f"  - Compressed: {processing_result['compressed']}")
                    self.logger.info(f"  - Within limit: {processing_result['within_limit']}")
                
                # Step 2: Load prompt components
                self.logger.info("Step 2: Loading prompt components")
                assignment_details = self.prompt_manager.load_assignment_details()
                instructions = self.prompt_manager.load_instruction_content()
                general_rubric = self.prompt_manager.load_general_rubric()
                specific_rubrics = self.prompt_manager.load_specific_rubric()
                
                # Step 3: Check for multimodal support if needed
                attachment_files = []
                
                # Scan for attachments in code submissions
                code_attachments = []
                if submission_is == 'code':
                    code_attachments = self._extract_attachments_from_code_zip(zip_path, temp_dir)
                    if code_attachments:
                        self.logger.info(f"Found {len(code_attachments)} attachment file(s) in code submission")
                        for attachment in code_attachments:
                            self.logger.info(f"  - {attachment.name}")
                
                # Check if any multimodal processing is needed
                needs_multimodal = (prompt_has_img_pdf or 
                                   submission_is == 'report' or 
                                   len(code_attachments) > 0)
                
                if needs_multimodal:
                    if not self.llm_provider.supports_multimodal():
                        raise Exception(f"LLM provider {self.llm_provider} does not support multimodal content (images/PDFs). Please use Gemini or OpenAI for image/PDF processing.")
                    
                    if prompt_has_img_pdf:
                        # Get attachment files from prompts directory
                        prompt_attachments = self.prompt_manager.get_attachment_files()
                        attachment_files.extend(prompt_attachments)
                        self.logger.info(f"Found {len(prompt_attachments)} prompt attachment files")
                    
                    if submission_is == 'report':
                        # Add report files to attachments
                        report_files = processing_result.get('report_files', [])
                        attachment_files.extend(report_files)
                        self.logger.info(f"Added {len(report_files)} report files")
                    
                    if submission_is == 'code' and code_attachments:
                        # Add code submission attachments
                        attachment_files.extend(code_attachments)
                        self.logger.info(f"Added {len(code_attachments)} code submission attachment files")
                
                # Step 4: Divide rubrics for parallel processing
                max_parallel = self.project_config.get('max_parallel_llm', 2)
                rubric_groups = self.prompt_manager.divide_rubrics_for_parallel_processing(
                    specific_rubrics, max_parallel
                )
                
                self.logger.info(f"Step 4: Divided {len(specific_rubrics)} rubrics into {len(rubric_groups)} groups for parallel processing")
                
                # Step 5: Process rubric groups in parallel
                self.logger.info("Step 5: Processing rubric evaluations in parallel")
                rubric_evaluations = self._process_rubrics_parallel_multimodal(
                    assignment_details, instructions, content_for_evaluation, 
                    general_rubric, rubric_groups, zip_path.name, attachment_files, submission_is
                )
                
                # Step 6: Run semgrep analysis if enabled (only for code submissions)
                semgrep_result = None
                semgrep_raw_output = ""
                semgrep_structured = None
                
                if self.semgrep_analyzer and submission_is == 'code':
                    self.logger.info("Step 6: Running Semgrep static analysis")
                    semgrep_result = self.semgrep_analyzer.analyze_codebase(zip_path, temp_dir)
                    
                    # Save raw semgrep output
                    semgrep_raw_output = self._format_semgrep_raw_output(semgrep_result)
                    
                    if semgrep_result['success']:
                        self.logger.info("Step 6a: Generating static analysis evaluation")
                        semgrep_structured = self._process_semgrep_analysis(
                            semgrep_result, zip_path.name
                        )
                    else:
                        self.logger.warning(f"Semgrep analysis failed: {semgrep_result.get('error', 'Unknown error')}")
                elif submission_is == 'report':
                    self.logger.info("Skipping semgrep analysis for report submission")
                
                # Step 7: Save results using new output manager
                self.logger.info("Step 7: Saving evaluation results in three-folder structure")
                output_files = self.output_manager.save_evaluation_results(
                    zip_path, rubric_evaluations, semgrep_structured, 
                    processing_result, semgrep_result, semgrep_raw_output,
                    assignment_details, instructions, specific_rubrics
                )
                
                return {
                    'filename': zip_path.name,
                    'output_files': output_files,
                    'success': True,
                    'submission_type': submission_is,
                    'project_name': processing_result.get('project_name', zip_path.stem),
                    'rubric_groups_processed': len(rubric_groups),
                    'total_rubrics': len(specific_rubrics),
                    'semgrep_enabled': self.semgrep_analyzer is not None and submission_is == 'code',
                    'semgrep_findings': semgrep_result['findings_count'] if semgrep_result else 0,
                    'attachment_files_count': len(attachment_files),
                    'multimodal_used': len(attachment_files) > 0,
                    **processing_result  # Include all processing result data
                }
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path.name}: {str(e)}")
                raise
    
    
    
    def _process_rubrics_parallel_multimodal(self, assignment_details: str, instructions: str,
                                           content: str, general_rubric: str,
                                           rubric_groups: List[List[Dict[str, Any]]], 
                                           assignment_name: str, attachment_files: List[Path],
                                           submission_type: str = "code") -> List[Dict[str, Any]]:
        """
        Process rubric groups in parallel with multimodal support (images/PDFs).
        
        Args:
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            content: Processed content (codebase or report description)
            general_rubric: General rubric instructions
            rubric_groups: List of rubric groups to process
            assignment_name: Name of the assignment for logging
            attachment_files: List of attachment file paths
            
        Returns:
            List of structured evaluation results for each group
        """
        structured_evaluations = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rubric_groups)) as executor:
            # Submit all tasks
            future_to_group = {}
            for i, rubric_group in enumerate(rubric_groups):
                future = executor.submit(
                    self._evaluate_rubric_group_multimodal,
                    assignment_details, instructions, content,
                    general_rubric, rubric_group, i + 1, assignment_name, attachment_files, submission_type
                )
                future_to_group[future] = (rubric_group, i + 1)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_group):
                rubric_group, group_num = future_to_group[future]
                try:
                    structured_result = future.result()
                    structured_evaluations.append(structured_result)
                except Exception as e:
                    self.logger.error(f"Failed to evaluate rubric group {group_num}: {str(e)}")
                    # Create error evaluation
                    criteria_names = [criterion['criterion_name'] for criterion in rubric_group]
                    error_evaluation = {
                        'group_number': group_num,
                        'criteria_names': criteria_names,
                        'success': False,
                        'error': str(e),
                        'evaluations': []
                    }
                    structured_evaluations.append(error_evaluation)
        
        return structured_evaluations
    
    def _evaluate_rubric_group_multimodal(self, assignment_details: str, instructions: str,
                                        content: str, general_rubric: str,
                                        rubric_group: List[Dict[str, Any]], group_num: int,
                                        assignment_name: str, attachment_files: List[Path],
                                        submission_type: str = "code") -> Dict[str, Any]:
        """
        Evaluate a single group of rubric criteria with multimodal support.
        
        Args:
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            content: Content to evaluate (codebase or report description)
            general_rubric: General rubric instructions
            rubric_group: Group of rubric criteria to evaluate
            group_num: Group number for logging
            assignment_name: Assignment name for file naming
            attachment_files: List of attachment file paths
            
        Returns:
            Structured evaluation result for this group
        """
        criteria_names = [criterion['criterion_name'] for criterion in rubric_group]
        self.logger.info(f"Processing rubric group {group_num}: {', '.join(criteria_names)}")
        
        # Create prompt for this group
        prompt = self.prompt_manager.create_rubric_evaluation_prompt(
            assignment_details, instructions, content,
            general_rubric, rubric_group, submission_type
        )
        
        # Save prompt to extra_logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_filename = f"{assignment_name}_group_{group_num}_prompt_{timestamp}.txt"
        prompt_path = self.extra_logs_dir / prompt_filename
        prompt_path.write_text(prompt, encoding='utf-8')
        self.logger.info(f"Saved prompt for group {group_num} to: {prompt_path}")
        
        # Log prompt stats and attachment info
        prompt_tokens = self.llm_provider.count_tokens(prompt)
        self.logger.info(f"Group {group_num} prompt token count: {prompt_tokens}")
        self.logger.info(f"Group {group_num} attachment files: {len(attachment_files)}")
        
        # Check if prompt exceeds model limits
        max_model_tokens = self.llm_provider.get_max_tokens()
        if prompt_tokens > max_model_tokens:
            self.logger.warning(
                f"Group {group_num} prompt ({prompt_tokens} tokens) exceeds model limit "
                f"({max_model_tokens} tokens). Results may be truncated."
            )
        
        # Generate evaluation with or without attachments
        if attachment_files:
            # Use multimodal generation
            attachment_paths = [str(f) for f in attachment_files]
            raw_response = self.llm_provider.generate_response_with_attachments(
                prompt, attachment_paths
            )
            self.logger.info(f"Completed multimodal evaluation for group {group_num} with {len(attachment_files)} attachments")
        else:
            # Use regular text generation
            raw_response = self.llm_provider.generate_response(prompt)
            self.logger.info(f"Completed text evaluation for group {group_num}")
        
        # Save raw response to extra_logs
        response_filename = f"{assignment_name}_group_{group_num}_response_{timestamp}.txt"
        response_path = self.extra_logs_dir / response_filename
        response_path.write_text(raw_response, encoding='utf-8')
        self.logger.info(f"Saved raw response for group {group_num} to: {response_path}")
        
        # Parse structured response
        structured_data = self._parse_llm_response(raw_response, group_num, criteria_names)
        
        return {
            'group_number': group_num,
            'criteria_names': criteria_names,
            'success': structured_data['success'],
            'evaluations': structured_data['evaluations'],
            'raw_response': raw_response,
            'prompt_tokens': prompt_tokens,
            'attachment_files_count': len(attachment_files),
            'multimodal_used': len(attachment_files) > 0,
            'error': structured_data.get('error')
        }
    
    
    
    def _parse_llm_response(self, raw_response: str, group_num: int, 
                           expected_criteria: List[str]) -> Dict[str, Any]:
        """
        Parse LLM response to extract structured evaluation data.
        
        Args:
            raw_response: Raw response from LLM
            group_num: Group number for error reporting
            expected_criteria: List of expected criterion names
            
        Returns:
            Dictionary with parsed evaluations or error information
        """
        try:
            # Try to find JSON in the response
            response_text = raw_response.strip()
            
            # Handle cases where LLM adds extra text before/after JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_text = response_text[start_idx:end_idx]
            parsed_data = json.loads(json_text)
            
            # Validate structure
            if 'evaluations' not in parsed_data:
                raise ValueError("Missing 'evaluations' key in JSON response")
            
            evaluations = parsed_data['evaluations']
            if not isinstance(evaluations, list):
                raise ValueError("'evaluations' must be a list")
            
            # Validate each evaluation
            validated_evaluations = []
            for eval_item in evaluations:
                if not isinstance(eval_item, dict):
                    continue
                    
                required_fields = ['criterion_name', 'max_points', 'score_obtained', 
                                 'feedback_positive', 'feedback_negative', 'score_justification']
                
                if all(field in eval_item for field in required_fields):
                    # Ensure numeric fields are numbers
                    eval_item['max_points'] = float(eval_item['max_points'])
                    eval_item['score_obtained'] = float(eval_item['score_obtained'])
                    
                    # Handle optional issues field
                    if 'issues' not in eval_item:
                        eval_item['issues'] = []
                    elif not isinstance(eval_item['issues'], list):
                        eval_item['issues'] = []
                    
                    validated_evaluations.append(eval_item)
            
            self.logger.info(f"Successfully parsed {len(validated_evaluations)} evaluations for group {group_num}")
            
            return {
                'success': True,
                'evaluations': validated_evaluations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response for group {group_num}: {str(e)}")
            
            # Create fallback evaluations for expected criteria
            fallback_evaluations = []
            for criterion_name in expected_criteria:
                fallback_evaluations.append({
                    'criterion_name': criterion_name,
                    'max_points': 1.0,  # Default, will be corrected later
                    'score_obtained': 0.0,
                    'feedback_positive': 'Unable to parse LLM response',
                    'feedback_negative': f'Error parsing evaluation: {str(e)}',
                    'score_justification': 'Score unavailable due to parsing error',
                    'issues': ['LLM response parsing error']
                })
            
            return {
                'success': False,
                'evaluations': fallback_evaluations,
                'error': str(e)
            }
    
    def _format_semgrep_raw_output(self, semgrep_result: Dict[str, Any]) -> str:
        """
        Format raw semgrep output for saving to file.
        
        Args:
            semgrep_result: Semgrep analysis results
            
        Returns:
            Formatted raw output string
        """
        if not semgrep_result.get('success'):
            return f"Semgrep analysis failed: {semgrep_result.get('error', 'Unknown error')}"
        
        output = f"# Semgrep Analysis Raw Output\n\n"
        output += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"**Total Findings:** {semgrep_result.get('findings_count', 0)}\n"
        output += f"**Analysis Status:** {'Success' if semgrep_result['success'] else 'Failed'}\n\n"
        
        if semgrep_result.get('findings_count', 0) == 0:
            output += "No issues found by static analysis.\n"
            return output
        
        findings = semgrep_result.get('findings', [])
        if findings:
            output += "## Detailed Findings\n\n"
            for i, finding in enumerate(findings, 1):
                # Access SemgrepFinding dataclass attributes
                rule_id = finding.rule_id if hasattr(finding, 'rule_id') else 'Unknown rule'
                message = finding.message if hasattr(finding, 'message') else 'No message'
                file_path = finding.file_path if hasattr(finding, 'file_path') else 'Unknown file'
                line = finding.line if hasattr(finding, 'line') else 'Unknown line'
                severity = finding.severity if hasattr(finding, 'severity') else 'Unknown'
                
                output += f"### Finding #{i}\n"
                output += f"- **Rule ID:** {rule_id}\n"
                output += f"- **File:** {file_path}\n"
                output += f"- **Line:** {line}\n"
                output += f"- **Severity:** {severity}\n"
                output += f"- **Message:** {message}\n\n"
        
        return output

    

    
    def process_assignments(self) -> List[Dict[str, Any]]:
        """
        Process all ZIP files in the input directory.
        
        Returns:
            List of processing results for each assignment
        """
        self.logger.info("Starting assignment processing")
        
        zip_files = self._find_zip_files()
        if not zip_files:
            self.logger.warning("No ZIP files found in input directory")
            return []
        
        results = []
        for zip_path in zip_files:
            try:
                result = self._process_single_assignment(zip_path)
                results.append(result)
                
                # Clean up processed file if successful
                if result['success']:
                    self._cleanup_processed_file(zip_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path.name}: {str(e)}")
                results.append({
                    'filename': zip_path.name,
                    'success': False,
                    'error': str(e)
                })
        
        self.logger.info(f"Completed processing {len(results)} assignments")
        return results
    
    def _extract_attachments_from_code_zip(self, zip_path: Path, temp_dir: Path) -> List[Path]:
        """
        Extract images and PDF files from a code submission ZIP file.
        
        This method scans through the ZIP file to find any images or PDFs that
        students might have included in their code submission, as per Part 4 requirement.
        
        Args:
            zip_path: Path to the ZIP file
            temp_dir: Temporary directory for extraction
            
        Returns:
            List of paths to attachment files found in the ZIP
        """
        attachment_files = []
        
        # Supported attachment file extensions
        supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp'}
        
        try:
            # Create extraction directory for attachments
            attachments_dir = temp_dir / "code_attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract and scan ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Skip directories
                    if file_info.is_dir():
                        continue
                    
                    file_path = Path(file_info.filename)
                    file_extension = file_path.suffix.lower()
                    
                    # Check if it's a supported attachment type
                    if file_extension in supported_extensions:
                        # Extract the file
                        try:
                            # Create safe filename (avoid path traversal)
                            safe_filename = file_path.name
                            if not safe_filename:  # Handle edge cases
                                safe_filename = f"attachment_{len(attachment_files)}{file_extension}"
                            
                            extracted_path = attachments_dir / safe_filename
                            
                            # Handle duplicate filenames
                            counter = 1
                            original_path = extracted_path
                            while extracted_path.exists():
                                stem = original_path.stem
                                suffix = original_path.suffix
                                extracted_path = attachments_dir / f"{stem}_{counter}{suffix}"
                                counter += 1
                            
                            # Extract the file
                            with zip_ref.open(file_info) as source, open(extracted_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            
                            attachment_files.append(extracted_path)
                            self.logger.debug(f"Extracted attachment: {file_info.filename} -> {extracted_path.name}")
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to extract attachment {file_info.filename}: {str(e)}")
                            continue
            
            if attachment_files:
                self.logger.info(f"Successfully extracted {len(attachment_files)} attachment file(s) from code submission")
            else:
                self.logger.debug("No attachment files found in code submission")
                
        except Exception as e:
            self.logger.warning(f"Failed to scan ZIP file for attachments: {str(e)}")
            return []
        
        return attachment_files
    
    def _cleanup_processed_file(self, zip_path: Path) -> None:
        """
        Clean up the processed ZIP file by moving it to a processed folder.
        
        Args:
            zip_path: Path to the processed ZIP file
        """
        try:
            processed_folder = self.settings.input_folder / "processed"
            processed_folder.mkdir(exist_ok=True)
            
            new_path = processed_folder / zip_path.name
            shutil.move(str(zip_path), str(new_path))
            self.logger.info(f"Moved processed file to: {new_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to move processed file {zip_path.name}: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the autograder.
        
        Returns:
            Dictionary containing status information
        """
        zip_files = self._find_zip_files()
        content_status = self.prompt_manager.get_content_files_status()
        
        return {
            'llm_provider': str(self.llm_provider),
            'project_assignment': self.settings.project_assignment,
            'pending_assignments': len(zip_files),
            'assignment_files': [f.name for f in zip_files],
            'content_files_status': content_status,
            'semgrep_enabled': self.semgrep_analyzer is not None,
            'project_config': self.project_config,
            'prompts_directory': str(self.settings.get_prompts_dir()),
            'extra_logs_directory': str(self.extra_logs_dir)
        }
    
    def _process_semgrep_analysis(self, semgrep_result: Dict[str, Any], 
                                assignment_name: str) -> Dict[str, Any]:
        """
        Process semgrep analysis with structured output and detailed logging.
        
        Args:
            semgrep_result: Semgrep analysis results
            assignment_name: Assignment name for file naming
            
        Returns:
            Structured semgrep evaluation result
        """
        try:
            # Create static analysis prompt
            static_prompt = self.prompt_manager.create_static_analysis_prompt(semgrep_result)
            
            # Save prompt to extra_logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_filename = f"{assignment_name}_semgrep_prompt_{timestamp}.txt"
            prompt_path = self.extra_logs_dir / prompt_filename
            prompt_path.write_text(static_prompt, encoding='utf-8')
            self.logger.info(f"Saved semgrep prompt to: {prompt_path}")
            
            # Generate evaluation
            raw_response = self.llm_provider.generate_response(static_prompt)
            self.logger.info("Completed semgrep analysis evaluation")
            
            # Save raw response to extra_logs
            response_filename = f"{assignment_name}_semgrep_response_{timestamp}.txt"
            response_path = self.extra_logs_dir / response_filename
            response_path.write_text(raw_response, encoding='utf-8')
            self.logger.info(f"Saved semgrep raw response to: {response_path}")
            
            # Parse structured response
            structured_data = self._parse_llm_response(raw_response, "semgrep", ["Static Code Analysis"])
            
            return {
                'success': structured_data['success'],
                'evaluations': structured_data['evaluations'],
                'raw_response': raw_response,
                'error': structured_data.get('error')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process semgrep analysis: {str(e)}")
            return {
                'success': False,
                'evaluations': [{
                    'criterion_name': 'Static Code Analysis',
                    'max_points': 10.0,
                    'score_obtained': 0.0,
                    'feedback_positive': 'Unable to process static analysis',
                    'feedback_negative': f'Error processing semgrep analysis: {str(e)}',
                    'score_justification': 'Score unavailable due to processing error'
                }],
                'error': str(e)
            } 