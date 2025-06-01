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
        self.repomix_processor = RepomixProcessor(
            max_tokens=self.project_config.get('max_file_size', settings.max_tokens),
            use_compression=settings.use_compression,
            remove_comments=settings.remove_comments
        )
        
        # Initialize semgrep analyzer if enabled
        self.semgrep_analyzer = None
        enable_semgrep = self.project_config.get('enable_semgrep_analysis', False)
        
        if enable_semgrep:
            try:
                semgrep_rules_file = self.project_config.get('semgrep_rules_file', 'config/semgrep_rules.yaml')
                semgrep_timeout = self.project_config.get('semgrep_timeout', 300)
                
                self.semgrep_analyzer = SemgrepAnalyzer(
                    rules_file=semgrep_rules_file,
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
                
                # Step 2: Load prompt components
                self.logger.info("Step 2: Loading prompt components")
                assignment_details = self.prompt_manager.load_assignment_details()
                instructions = self.prompt_manager.load_instruction_content()
                general_rubric = self.prompt_manager.load_general_rubric()
                specific_rubrics = self.prompt_manager.load_specific_rubric()
                
                # Step 3: Divide rubrics for parallel processing
                max_parallel = self.project_config.get('max_parallel_llm', 2)
                rubric_groups = self.prompt_manager.divide_rubrics_for_parallel_processing(
                    specific_rubrics, max_parallel
                )
                
                self.logger.info(f"Step 3: Divided {len(specific_rubrics)} rubrics into {len(rubric_groups)} groups for parallel processing")
                
                # Step 4: Process rubric groups in parallel
                self.logger.info("Step 4: Processing rubric evaluations in parallel")
                rubric_evaluations = self._process_rubrics_parallel(
                    assignment_details, instructions, repomix_result['content'], 
                    general_rubric, rubric_groups, zip_path.name
                )
                
                # Step 5: Run semgrep analysis if enabled
                semgrep_analysis = ""
                semgrep_result = None
                semgrep_raw_output = ""
                semgrep_structured = None
                
                if self.semgrep_analyzer:
                    self.logger.info("Step 5: Running Semgrep static analysis")
                    semgrep_result = self.semgrep_analyzer.analyze_codebase(zip_path, temp_dir)
                    
                    # Save raw semgrep output
                    semgrep_raw_output = self._format_semgrep_raw_output(semgrep_result)
                    
                    if semgrep_result['success']:
                        self.logger.info("Step 5a: Generating static analysis evaluation")
                        semgrep_structured = self._process_semgrep_analysis(
                            semgrep_result, zip_path.name
                        )
                    else:
                        self.logger.warning(f"Semgrep analysis failed: {semgrep_result.get('error', 'Unknown error')}")
                
                # Step 6: Save results
                self.logger.info("Step 6: Saving evaluation results")
                output_files = self._save_evaluation_results(
                    zip_path, rubric_evaluations, semgrep_structured, 
                    repomix_result, semgrep_result, semgrep_raw_output
                )
                
                return {
                    'filename': zip_path.name,
                    'output_files': output_files,
                    'success': True,
                    'token_count': repomix_result['token_count'],
                    'compressed': repomix_result['compressed'],
                    'project_name': repomix_result['project_name'],
                    'rubric_groups_processed': len(rubric_groups),
                    'total_rubrics': len(specific_rubrics),
                    'semgrep_enabled': self.semgrep_analyzer is not None,
                    'semgrep_findings': semgrep_result['findings_count'] if semgrep_result else 0
                }
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path.name}: {str(e)}")
                raise
    
    def _process_rubrics_parallel(self, assignment_details: str, instructions: str,
                                 codebase_content: str, general_rubric: str,
                                 rubric_groups: List[List[Dict[str, Any]]], 
                                 assignment_name: str) -> List[Dict[str, Any]]:
        """
        Process rubric groups in parallel using ThreadPoolExecutor with structured output.
        
        Args:
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            codebase_content: Processed codebase content
            general_rubric: General rubric instructions
            rubric_groups: List of rubric groups to process
            assignment_name: Name of the assignment for logging
            
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
                    self._evaluate_rubric_group_structured,
                    assignment_details, instructions, codebase_content,
                    general_rubric, rubric_group, i + 1, assignment_name
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
    
    def _evaluate_rubric_group_structured(self, assignment_details: str, instructions: str,
                                        codebase_content: str, general_rubric: str,
                                        rubric_group: List[Dict[str, Any]], group_num: int,
                                        assignment_name: str) -> Dict[str, Any]:
        """
        Evaluate a single group of rubric criteria with structured output and detailed logging.
        
        Args:
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            codebase_content: Processed codebase content
            general_rubric: General rubric instructions
            rubric_group: Group of rubric criteria to evaluate
            group_num: Group number for logging
            assignment_name: Assignment name for file naming
            
        Returns:
            Structured evaluation result for this group
        """
        criteria_names = [criterion['criterion_name'] for criterion in rubric_group]
        self.logger.info(f"Processing rubric group {group_num}: {', '.join(criteria_names)}")
        
        # Create prompt for this group
        prompt = self.prompt_manager.create_rubric_evaluation_prompt(
            assignment_details, instructions, codebase_content,
            general_rubric, rubric_group
        )
        
        # Save prompt to extra_logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_filename = f"{assignment_name}_group_{group_num}_prompt_{timestamp}.txt"
        prompt_path = self.extra_logs_dir / prompt_filename
        prompt_path.write_text(prompt, encoding='utf-8')
        self.logger.info(f"Saved prompt for group {group_num} to: {prompt_path}")
        
        # Log prompt stats
        prompt_tokens = self.llm_provider.count_tokens(prompt)
        self.logger.info(f"Group {group_num} prompt token count: {prompt_tokens}")
        
        # Check if prompt exceeds model limits
        max_model_tokens = self.llm_provider.get_max_tokens()
        if prompt_tokens > max_model_tokens:
            self.logger.warning(
                f"Group {group_num} prompt ({prompt_tokens} tokens) exceeds model limit "
                f"({max_model_tokens} tokens). Results may be truncated."
            )
        
        # Generate evaluation
        raw_response = self.llm_provider.generate_response(prompt)
        self.logger.info(f"Completed evaluation for group {group_num}")
        
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
                    'score_justification': 'Score unavailable due to parsing error'
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
                rule_id = finding.get('rule_id', 'Unknown rule')
                message = finding.get('message', 'No message')
                file_path = finding.get('file', 'Unknown file')
                line = finding.get('line', 'Unknown line')
                severity = finding.get('severity', 'Unknown')
                
                output += f"### Finding #{i}\n"
                output += f"- **Rule ID:** {rule_id}\n"
                output += f"- **File:** {file_path}\n"
                output += f"- **Line:** {line}\n"
                output += f"- **Severity:** {severity}\n"
                output += f"- **Message:** {message}\n\n"
        
        return output
    
    def _save_evaluation_results(self, zip_path: Path, rubric_evaluations: List[Dict[str, Any]],
                               semgrep_structured: Optional[Dict[str, Any]], repomix_result: Dict[str, Any],
                               semgrep_result: Optional[Dict[str, Any]],
                               semgrep_raw_output: str) -> Dict[str, Path]:
        """
        Save all evaluation results to the output directory with structured format.
        
        Args:
            zip_path: Original ZIP file path
            rubric_evaluations: List of structured rubric evaluation results
            semgrep_structured: Structured semgrep analysis evaluation
            repomix_result: Repomix processing results
            semgrep_result: Semgrep analysis results
            semgrep_raw_output: Raw semgrep output
            
        Returns:
            Dictionary of output file paths
        """
        # Create output filenames based on input filename
        base_name = zip_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main evaluation report (Markdown format)
        main_output_filename = f"{base_name}_evaluation_{timestamp}.md"
        main_output_path = self.settings.output_folder / main_output_filename
        
        # Semgrep raw output file
        semgrep_output_filename = f"{base_name}_semgrep_raw_{timestamp}.txt"
        semgrep_output_path = self.settings.output_folder / semgrep_output_filename
        
        # Create comprehensive evaluation report
        main_report = self._create_comprehensive_evaluation_report(
            zip_path, rubric_evaluations, semgrep_structured, repomix_result, semgrep_result
        )
        
        # Save main report
        main_output_path.write_text(main_report, encoding='utf-8')
        self.logger.info(f"Main evaluation saved to: {main_output_path}")
        
        # Save semgrep raw output if analysis was performed
        output_files = {'main_evaluation': main_output_path}
        
        if semgrep_raw_output and self.semgrep_analyzer:
            semgrep_output_path.write_text(semgrep_raw_output, encoding='utf-8')
            self.logger.info(f"Semgrep raw output saved to: {semgrep_output_path}")
            output_files['semgrep_raw'] = semgrep_output_path
        
        return output_files
    
    def _create_comprehensive_evaluation_report(self, zip_path: Path, rubric_evaluations: List[Dict[str, Any]],
                                              semgrep_structured: Optional[Dict[str, Any]], repomix_result: Dict[str, Any],
                                              semgrep_result: Optional[Dict[str, Any]]) -> str:
        """
        Create a comprehensive structured evaluation report with calculated scores.
        
        Args:
            zip_path: Original ZIP file path
            rubric_evaluations: List of structured rubric evaluation results
            semgrep_structured: Structured semgrep analysis evaluation
            repomix_result: Repomix processing results
            semgrep_result: Semgrep analysis results
            
        Returns:
            Formatted comprehensive evaluation report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate processing statistics
        successful_groups = [eval for eval in rubric_evaluations if eval.get('success', True)]
        failed_groups = [eval for eval in rubric_evaluations if not eval.get('success', True)]
        
        processing_stats = f"""## Processing Statistics

- **Original Token Count:** {repomix_result['token_count']:,}
- **Compression Applied:** {'Yes' if repomix_result['compressed'] else 'No'}
- **Within Token Limit:** {'Yes' if repomix_result['within_limit'] else 'No'}
- **Token Limit:** {self.project_config.get('max_file_size', self.settings.max_tokens):,}
- **Parallel LLM Runs:** {len(rubric_evaluations)}
- **Successful Groups:** {len(successful_groups)}
- **Failed Groups:** {len(failed_groups)}
- **Max Parallel Configured:** {self.project_config.get('max_parallel_llm', 2)}"""
        
        # Add semgrep statistics if enabled
        if self.semgrep_analyzer and semgrep_result:
            processing_stats += f"""
- **Static Analysis:** {'Enabled' if self.semgrep_analyzer else 'Disabled'}
- **Static Analysis Findings:** {semgrep_result['findings_count']}
- **Static Analysis Status:** {'Success' if semgrep_result['success'] else 'Failed'}"""
        
        # Collect all evaluations for scoring
        all_evaluations = []
        total_score = 0.0
        total_max_score = 0.0
        
        # Process rubric evaluations
        rubric_section = "# Detailed Rubric Evaluations\n\n"
        
        for group_data in rubric_evaluations:
            group_num = group_data.get('group_number', 'Unknown')
            criteria_names = group_data.get('criteria_names', [])
            success = group_data.get('success', True)
            
            rubric_section += f"## Group {group_num}: {', '.join(criteria_names)}\n\n"
            
            if success and group_data.get('evaluations'):
                for evaluation in group_data['evaluations']:
                    criterion_name = evaluation.get('criterion_name', 'Unknown')
                    max_points = evaluation.get('max_points', 0.0)
                    score_obtained = evaluation.get('score_obtained', 0.0)
                    feedback_positive = evaluation.get('feedback_positive', 'No feedback available')
                    feedback_negative = evaluation.get('feedback_negative', 'No feedback available')
                    score_justification = evaluation.get('score_justification', 'No justification available')
                    
                    # Add to totals
                    total_max_score += max_points
                    total_score += score_obtained
                    all_evaluations.append(evaluation)
                    
                    rubric_section += f"""### {criterion_name} - Score: {score_obtained:.1f}/{max_points:.1f}

**What was done correctly:**
{feedback_positive}

**Major flaws identified:**
{feedback_negative}

**Score justification:**
{score_justification}

---

"""
            else:
                error_msg = group_data.get('error', 'Unknown error')
                rubric_section += f"**Error processing this group:** {error_msg}\n\n---\n\n"
        
        # Process semgrep evaluation if available
        semgrep_section = ""
        if semgrep_structured and semgrep_structured.get('success') and semgrep_structured.get('evaluations'):
            semgrep_section = "\n# Static Code Analysis Evaluation\n\n"
            
            for evaluation in semgrep_structured['evaluations']:
                criterion_name = evaluation.get('criterion_name', 'Static Code Analysis')
                max_points = evaluation.get('max_points', 10.0)
                score_obtained = evaluation.get('score_obtained', 0.0)
                feedback_positive = evaluation.get('feedback_positive', 'No feedback available')
                feedback_negative = evaluation.get('feedback_negative', 'No feedback available')
                score_justification = evaluation.get('score_justification', 'No justification available')
                
                # Add to totals
                total_max_score += max_points
                total_score += score_obtained
                all_evaluations.append(evaluation)
                
                semgrep_section += f"""## {criterion_name} - Score: {score_obtained:.1f}/{max_points:.1f}

**What was done correctly:**
{feedback_positive}

**Major flaws identified:**
{feedback_negative}

**Score justification:**
{score_justification}

"""
        elif self.semgrep_analyzer:
            semgrep_section = "\n# Static Code Analysis Evaluation\n\n**Static analysis was enabled but evaluation failed or was not performed.**\n"
        
        # Calculate final score percentage
        final_percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0.0
        
        # Create score summary
        score_summary = f"""# Final Score Summary

| Component | Score Obtained | Max Score | Percentage |
|-----------|---------------|-----------|------------|
| **Rubric Criteria** | **{sum(e['score_obtained'] for e in all_evaluations if e['criterion_name'] != 'Static Code Analysis'):.1f}** | **{sum(e['max_points'] for e in all_evaluations if e['criterion_name'] != 'Static Code Analysis'):.1f}** | **{(sum(e['score_obtained'] for e in all_evaluations if e['criterion_name'] != 'Static Code Analysis') / sum(e['max_points'] for e in all_evaluations if e['criterion_name'] != 'Static Code Analysis') * 100) if sum(e['max_points'] for e in all_evaluations if e['criterion_name'] != 'Static Code Analysis') > 0 else 0:.1f}%** |"""
        
        if any(e['criterion_name'] == 'Static Code Analysis' for e in all_evaluations):
            static_score = next((e['score_obtained'] for e in all_evaluations if e['criterion_name'] == 'Static Code Analysis'), 0)
            static_max = next((e['max_points'] for e in all_evaluations if e['criterion_name'] == 'Static Code Analysis'), 10)
            static_pct = (static_score / static_max * 100) if static_max > 0 else 0
            score_summary += f"""
| **Static Analysis** | **{static_score:.1f}** | **{static_max:.1f}** | **{static_pct:.1f}%** |"""
        
        score_summary += f"""
| **TOTAL** | **{total_score:.1f}** | **{total_max_score:.1f}** | **{final_percentage:.1f}%** |

## Grade: {final_percentage:.1f}% ({total_score:.1f}/{total_max_score:.1f} points)
"""
        
        # Build the complete report
        report = f"""# AutoGrader Comprehensive Evaluation Report

**Assignment:** {zip_path.name}
**Project Name:** {repomix_result['project_name']}
**Evaluation Date:** {timestamp}
**LLM Provider:** {self.llm_provider}
**Project Configuration:** {self.settings.project_assignment}

{processing_stats}

{score_summary}

{rubric_section}

{semgrep_section}

---
*Report generated by AutoGrader v2.0 - Enhanced with Parallel Processing and Structured Output*
*For detailed prompts and responses, see the extra_logs directory*
*For detailed static analysis findings, see the corresponding semgrep raw output file*
"""
        
        return report
    
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