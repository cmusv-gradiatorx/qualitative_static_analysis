"""
Output Manager

Manages the three-folder output structure for autograder results:
1. Complete Markdown - detailed technical evaluation
2. JSON - structured data format
3. Friendly - user-friendly summaries

Author: Auto-generated
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..llm.base import LLMProvider
from ..prompts.prompt_manager import PromptManager
from ..utils.logger import get_logger


class OutputManager:
    """
    Manages creation and organization of autograder output files.
    
    Creates three types of output:
    1. Complete Markdown: Comprehensive technical evaluation (existing format)
    2. JSON: Structured data format for programmatic access
    3. Friendly: User-friendly summary for students
    """
    
    def __init__(self, output_folder: Path, llm_provider: LLMProvider, prompt_manager: PromptManager):
        """
        Initialize the output manager.
        
        Args:
            output_folder: Base output directory
            llm_provider: LLM provider for generating friendly summaries
            prompt_manager: Prompt manager for final prompt generation
        """
        self.output_folder = Path(output_folder)
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = get_logger(__name__)
        
        # Create the three output subdirectories
        self.complete_markdown_dir = self.output_folder / "complete_markdown"
        self.json_dir = self.output_folder / "json"
        self.friendly_dir = self.output_folder / "friendly"
        
        for directory in [self.complete_markdown_dir, self.json_dir, self.friendly_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Output manager initialized with three-folder structure")
        self.logger.info(f"  - Complete Markdown: {self.complete_markdown_dir}")
        self.logger.info(f"  - JSON: {self.json_dir}")
        self.logger.info(f"  - Friendly: {self.friendly_dir}")
    
    def save_evaluation_results(self, zip_path: Path, rubric_evaluations: List[Dict[str, Any]],
                              semgrep_structured: Optional[Dict[str, Any]], 
                              processing_result: Dict[str, Any],
                              semgrep_result: Optional[Dict[str, Any]],
                              semgrep_raw_output: str,
                              assignment_details: str = "",
                              instructions: str = "",
                              rubrics: List[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Save all evaluation results in the three-folder structure.
        
        Args:
            zip_path: Original ZIP file path
            rubric_evaluations: List of structured rubric evaluation results
            semgrep_structured: Structured semgrep analysis evaluation
            processing_result: Code/report processing results
            semgrep_result: Raw semgrep analysis results
            semgrep_raw_output: Raw semgrep output text
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            rubrics: List of rubric criteria
            
        Returns:
            Dictionary of output file paths
        """
        base_name = zip_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create complete markdown report
        complete_markdown_content = self._create_comprehensive_evaluation_report(
            zip_path, rubric_evaluations, semgrep_structured, processing_result, semgrep_result
        )
        
        markdown_filename = f"{base_name}_evaluation_{timestamp}.md"
        markdown_path = self.complete_markdown_dir / markdown_filename
        markdown_path.write_text(complete_markdown_content, encoding='utf-8')
        
        # 2. Create JSON structured report
        json_data = self._create_json_report(
            zip_path, rubric_evaluations, semgrep_structured, processing_result, semgrep_result, timestamp
        )
        
        json_filename = f"{base_name}_evaluation_{timestamp}.json"
        json_path = self.json_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 3. Create friendly user report
        friendly_content = self._create_friendly_report(
            complete_markdown_content, assignment_details, instructions, rubrics or [], zip_path.stem
        )
        
        friendly_filename = f"{base_name}_friendly_{timestamp}.md"
        friendly_path = self.friendly_dir / friendly_filename
        friendly_path.write_text(friendly_content, encoding='utf-8')
        
        # 4. Save semgrep raw output if available
        output_files = {
            'complete_markdown': markdown_path,
            'json': json_path,
            'friendly': friendly_path
        }
        
        if semgrep_raw_output:
            semgrep_filename = f"{base_name}_semgrep_raw_{timestamp}.txt"
            semgrep_path = self.complete_markdown_dir / semgrep_filename
            semgrep_path.write_text(semgrep_raw_output, encoding='utf-8')
            output_files['semgrep_raw'] = semgrep_path
        
        self.logger.info(f"All outputs saved for {base_name}:")
        for output_type, path in output_files.items():
            self.logger.info(f"  - {output_type}: {path}")
        
        return output_files
    
    def _create_comprehensive_evaluation_report(self, zip_path: Path, rubric_evaluations: List[Dict[str, Any]],
                                              semgrep_structured: Optional[Dict[str, Any]], 
                                              processing_result: Dict[str, Any],
                                              semgrep_result: Optional[Dict[str, Any]]) -> str:
        """Create the complete markdown evaluation report (existing format)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate processing statistics
        successful_groups = [eval for eval in rubric_evaluations if eval.get('success', True)]
        failed_groups = [eval for eval in rubric_evaluations if not eval.get('success', True)]
        
        # Determine processing type
        submission_type = processing_result.get('submission_type', 'code')
        
        if submission_type == 'code':
            processing_stats = f"""## Processing Statistics

- **Original Token Count:** {processing_result.get('token_count', 0):,}
- **Compression Applied:** {'Yes' if processing_result.get('compressed', False) else 'No'}
- **Within Token Limit:** {'Yes' if processing_result.get('within_limit', True) else 'No'}
- **Files Filtered by Project Config:** {processing_result.get('filtered_files_count', 0)}"""
        else:  # report submission
            processing_stats = f"""## Processing Statistics

- **Submission Type:** Report (PDF/Images)
- **Report Files Count:** {processing_result.get('file_count', 0)}
- **Processing Status:** {'Success' if processing_result.get('success', False) else 'Failed'}"""
        
        processing_stats += f"""
- **Parallel LLM Runs:** {len(rubric_evaluations)}
- **Successful Groups:** {len(successful_groups)}
- **Failed Groups:** {len(failed_groups)}"""
        
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
                    issues = evaluation.get('issues', [])
                    
                    # Add to totals
                    total_max_score += max_points
                    total_score += score_obtained
                    all_evaluations.append(evaluation)
                    
                    issues_text = ""
                    if issues:
                        issues_text = f"""

**Issues Identified:**
{chr(10).join([f"- {issue}" for issue in issues])}"""
                    
                    rubric_section += f"""### {criterion_name} - Score: {score_obtained:.1f}/{max_points:.1f}

**What was done correctly:**
{feedback_positive}

**Major flaws identified:**
{feedback_negative}

**Score justification:**
{score_justification}{issues_text}

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
                issues = evaluation.get('issues', [])
                
                # Add to totals
                total_max_score += max_points
                total_score += score_obtained
                all_evaluations.append(evaluation)
                
                issues_text = ""
                if issues:
                    issues_text = f"""

**Issues Identified:**
{chr(10).join([f"- {issue}" for issue in issues])}"""
                
                semgrep_section += f"""## {criterion_name} - Score: {score_obtained:.1f}/{max_points:.1f}

**What was done correctly:**
{feedback_positive}

**Major flaws identified:**
{feedback_negative}

**Score justification:**
{score_justification}{issues_text}

"""
        
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
**Project Name:** {processing_result.get('project_name', zip_path.stem)}
**Evaluation Date:** {timestamp}

{score_summary}

{processing_stats}

{rubric_section}

{semgrep_section}

---
*Report generated by AutoGrader*
"""
        
        return report
    
    def _create_json_report(self, zip_path: Path, rubric_evaluations: List[Dict[str, Any]],
                          semgrep_structured: Optional[Dict[str, Any]], 
                          processing_result: Dict[str, Any],
                          semgrep_result: Optional[Dict[str, Any]], timestamp: str) -> Dict[str, Any]:
        """Create the JSON structured report."""
        # Collect all evaluations
        all_evaluations = []
        
        # Process rubric evaluations
        for group_data in rubric_evaluations:
            if group_data.get('success') and group_data.get('evaluations'):
                all_evaluations.extend(group_data['evaluations'])
        
        # Add semgrep evaluation if available
        if semgrep_structured and semgrep_structured.get('success') and semgrep_structured.get('evaluations'):
            all_evaluations.extend(semgrep_structured['evaluations'])
        
        # Calculate totals
        total_score = sum(e.get('score_obtained', 0) for e in all_evaluations)
        total_max_score = sum(e.get('max_points', 0) for e in all_evaluations)
        final_percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0.0
        
        # Collect all issues
        all_issues = []
        for evaluation in all_evaluations:
            all_issues.extend(evaluation.get('issues', []))
        
        json_data = {
            "metadata": {
                "assignment": zip_path.name,
                "project_name": processing_result.get('project_name', zip_path.stem),
                "evaluation_date": timestamp,
                "submission_type": processing_result.get('submission_type', 'code'),
                "autograder_version": "1.0"
            },
            "summary": {
                "total_score": total_score,
                "total_max_score": total_max_score,
                "percentage": round(final_percentage, 1),
                "grade": f"{final_percentage:.1f}% ({total_score:.1f}/{total_max_score:.1f} points)"
            },
            "processing_stats": {
                "parallel_llm_runs": len(rubric_evaluations),
                "successful_groups": len([e for e in rubric_evaluations if e.get('success', True)]),
                "failed_groups": len([e for e in rubric_evaluations if not e.get('success', True)])
            },
            "evaluations": all_evaluations,
            "issues": all_issues
        }
        
        # Add processing-specific stats
        if processing_result.get('submission_type') == 'code':
            json_data["processing_stats"].update({
                "token_count": processing_result.get('token_count', 0),
                "compressed": processing_result.get('compressed', False),
                "within_limit": processing_result.get('within_limit', True),
                "filtered_files_count": processing_result.get('filtered_files_count', 0)
            })
        elif processing_result.get('submission_type') == 'report':
            json_data["processing_stats"].update({
                "report_files_count": processing_result.get('file_count', 0),
                "processing_success": processing_result.get('success', False)
            })
        
        return json_data
    
    def _create_friendly_report(self, comprehensive_report: str, assignment_details: str, 
                              instructions: str, rubrics: List[Dict[str, Any]], assignment_name: str = "") -> str:
        """Create the user-friendly report using LLM."""
        try:
            # Generate friendly prompt
            friendly_prompt = self.prompt_manager.create_friendly_output_prompt(
                comprehensive_report, assignment_details, instructions, rubrics
            )
            
            # Save friendly prompt to extra_logs
            if assignment_name:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extra_logs_dir = Path("extra_logs")
                extra_logs_dir.mkdir(exist_ok=True)
                friendly_prompt_filename = f"{assignment_name}_friendly_prompt_{timestamp}.txt"
                friendly_prompt_path = extra_logs_dir / friendly_prompt_filename
                friendly_prompt_path.write_text(friendly_prompt, encoding='utf-8')
                self.logger.info(f"Saved friendly prompt to: {friendly_prompt_path}")
            
            # Generate friendly response using LLM
            friendly_content = self.llm_provider.generate_response(
                friendly_prompt,
                temperature=0.3,
                max_tokens=4096
            )
            
            return friendly_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate friendly report: {str(e)}")
            # Return a fallback friendly message
            return f"""# Assignment Evaluation Summary

I apologize, but I encountered an issue while generating your friendly evaluation summary. 

**Error:** {str(e)}

Please refer to the complete technical evaluation for detailed feedback on your assignment.

The technical evaluation contains all the detailed feedback, scores, and suggestions for improvement that were generated during the grading process.

If you continue to see this error, please contact your instructor for assistance.
""" 