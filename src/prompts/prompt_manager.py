"""
Prompt Manager

Manages loading and combining different prompt templates for the autograder.
Supports new structured rubric format with parallel LLM processing.

Author: Auto-generated
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple


class PromptManager:
    """
    Manages prompt templates and constructs final prompts.
    
    This class loads assignment details, instruction content, rubric content,
    and creates structured prompts for parallel LLM evaluation.
    """
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt content files
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Define required content files
        self.assignment_details_file = self.prompts_dir / "assignment_details.txt"
        self.instruction_content_file = self.prompts_dir / "instruction_content.txt"
        self.general_rubric_file = self.prompts_dir / "general_rubric.txt"
        self.specific_rubric_file = self.prompts_dir / "specific_rubric.json"
        self.static_instructions_file = self.prompts_dir / "static_instructions.txt"
        
        # Validate that required files exist
        self._validate_content_files()
    
    def _validate_content_files(self) -> None:
        """
        Validate that required content files exist.
        
        Raises:
            FileNotFoundError: If required content files are missing
        """
        missing_files = []
        
        required_files = [
            self.assignment_details_file,
            self.instruction_content_file,
            self.general_rubric_file,
            self.specific_rubric_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            error_msg = (
                f"Required prompt content files are missing:\n"
                f"  - {chr(10).join(missing_files)}\n\n"
                f"Please create these files in the '{self.prompts_dir}' directory:\n"
                f"  - 'assignment_details.txt': Define assignment specifications\n"
                f"  - 'instruction_content.txt': Specify evaluation instructions\n"
                f"  - 'general_rubric.txt': Common rubric instructions\n"
                f"  - 'specific_rubric.json': JSON array of specific rubric criteria\n"
                f"  - 'static_instructions.txt': Define static analysis evaluation instructions (optional)"
            )
            raise FileNotFoundError(error_msg)
    
    def load_assignment_details(self) -> str:
        """
        Load assignment details content from file.
        
        Returns:
            Assignment details content as string
            
        Raises:
            FileNotFoundError: If assignment details file doesn't exist
        """
        try:
            return self.assignment_details_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Assignment details file not found: {self.assignment_details_file}")
        except Exception as e:
            raise Exception(f"Error reading assignment details: {str(e)}")
    
    def load_instruction_content(self) -> str:
        """
        Load instruction content from file.
        
        Returns:
            Instruction content as string
            
        Raises:
            FileNotFoundError: If instruction content file doesn't exist
        """
        try:
            return self.instruction_content_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Instruction content file not found: {self.instruction_content_file}")
        except Exception as e:
            raise Exception(f"Error reading instruction content: {str(e)}")
    
    def load_general_rubric(self) -> str:
        """
        Load general rubric content from file.
        
        Returns:
            General rubric content as string
            
        Raises:
            FileNotFoundError: If general rubric file doesn't exist
        """
        try:
            return self.general_rubric_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"General rubric file not found: {self.general_rubric_file}")
        except Exception as e:
            raise Exception(f"Error reading general rubric: {str(e)}")
    
    def load_specific_rubric(self) -> List[Dict[str, Any]]:
        """
        Load specific rubric criteria from JSON file.
        
        Returns:
            List of rubric criteria dictionaries
            
        Raises:
            FileNotFoundError: If specific rubric file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        try:
            with open(self.specific_rubric_file, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
                
            if not isinstance(rubric_data, list):
                raise ValueError("Specific rubric must be a JSON array")
                
            # Validate rubric structure
            for i, criterion in enumerate(rubric_data):
                required_fields = ['criterion_name', 'max_points', 'specific_prompt']
                for field in required_fields:
                    if field not in criterion:
                        raise ValueError(f"Rubric criterion {i} missing required field: {field}")
                        
            return rubric_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Specific rubric file not found: {self.specific_rubric_file}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing specific rubric JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading specific rubric: {str(e)}")
    
    def load_static_instructions(self) -> str:
        """
        Load static analysis instruction content from file.
        
        Returns:
            Static analysis instruction content as string
            
        Raises:
            FileNotFoundError: If static instructions file doesn't exist
        """
        try:
            return self.static_instructions_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Static instructions file not found: {self.static_instructions_file}")
        except Exception as e:
            raise Exception(f"Error reading static instructions: {str(e)}")
    
    def divide_rubrics_for_parallel_processing(self, rubrics: List[Dict[str, Any]], 
                                             max_parallel: int) -> List[List[Dict[str, Any]]]:
        """
        Divide rubrics into groups for parallel LLM processing.
        
        Args:
            rubrics: List of rubric criteria
            max_parallel: Maximum number of parallel LLM runs
            
        Returns:
            List of rubric groups for parallel processing
        """
        if not rubrics:
            return []
            
        if len(rubrics) <= max_parallel:
            # Each rubric gets its own LLM run
            return [[rubric] for rubric in rubrics]
        else:
            # Divide rubrics into groups
            group_size = math.ceil(len(rubrics) / max_parallel)
            groups = []
            
            for i in range(0, len(rubrics), group_size):
                group = rubrics[i:i + group_size]
                groups.append(group)
                
            return groups
    
    def create_rubric_evaluation_prompt(self, assignment_details: str, instructions: str,
                                      codebase_content: str, general_rubric: str,
                                      rubric_group: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for evaluating a specific group of rubric criteria.
        
        Args:
            assignment_details: Assignment specification content
            instructions: General evaluation instructions
            codebase_content: Processed codebase content
            general_rubric: General rubric instructions
            rubric_group: Group of rubric criteria to evaluate
            
        Returns:
            Complete prompt for rubric evaluation
        """
        # Combine specific prompts for this group
        specific_prompts = []
        criteria_info = []
        
        for criterion in rubric_group:
            criterion_name = criterion['criterion_name']
            max_points = criterion['max_points']
            specific_prompt = criterion['specific_prompt']
            
            criteria_info.append(f"**{criterion_name}** (Max Points: {max_points})")
            specific_prompts.append(f"### {criterion_name} (Max Points: {max_points})\n\n{specific_prompt}")
        
        combined_specific_prompt = "\n\n".join(specific_prompts)
        criteria_list = "\n- ".join(criteria_info)
        
        return f"""You are an expert software engineering instructor grading a graduate-level assignment.

**ASSIGNMENT DETAILS:**
{assignment_details}

**EVALUATION INSTRUCTIONS:**
{instructions}

**CODEBASE TO EVALUATE:**
{codebase_content}

**GENERAL RUBRIC INSTRUCTIONS:**
{general_rubric}

**SPECIFIC CRITERIA TO EVALUATE:**
- {criteria_list}

**DETAILED RUBRIC CRITERIA:**
{combined_specific_prompt}

**CRITICAL: RESPOND ONLY IN VALID JSON FORMAT**

You must respond with a valid JSON object containing evaluations for each criterion. The JSON structure must be:

{{
  "evaluations": [
    {{
      "criterion_name": "Exact criterion name",
      "max_points": numerical_max_points,
      "score_obtained": numerical_score_obtained,
      "feedback_positive": "Detailed feedback on what was done correctly and well. Better to provide exact details",
      "feedback_negative": "Detailed feedback on major flaws and issues that caused point deductions. Better to provide exact details",
      "score_justification": "Clear explanation of why this specific score was assigned based on the rubric"
    }}
  ]
}}

**IMPORTANT REQUIREMENTS:**
1. Respond ONLY with valid JSON - no additional text before or after
2. Include all criteria from the rubric group
3. Ensure scores are in the range give in rubric and according to the rubric (0 to max_points)
4. Provide detailed, constructive feedback
5. Use exact criterion names as provided in the rubric

Evaluate ONLY the criteria specified above and provide your assessment in the exact JSON format requested.

IMPORTANT NOTE: As mentioned before, the analysis(feedback_positive, feedback_negative, score_justification) should be thorough and detailed. While giving the feedback make sure to provide all (ideally all) instances(like the exact code artifacts: class, variables, functions, tests, etc.) you can on which you are giving the feedback.
"""
    
    def create_static_analysis_prompt(self, semgrep_results: Dict[str, Any]) -> str:
        """
        Create a prompt for static analysis evaluation using semgrep results.
        
        Args:
            semgrep_results: Results from semgrep analysis
            
        Returns:
            Complete prompt for static analysis evaluation
            
        Raises:
            FileNotFoundError: If static instructions file doesn't exist
        """
        try:
            # Load static analysis instructions
            static_instructions = self.load_static_instructions()
            
            # Format semgrep results for the prompt
            findings_text = self._format_semgrep_findings(semgrep_results)
            
            # Create the complete static analysis prompt
            prompt = f"""You are an expert software engineering instructor performing static code analysis evaluation.

**STATIC ANALYSIS INSTRUCTIONS:**
{static_instructions}

**SEMGREP ANALYSIS RESULTS:**
{findings_text}

**CRITICAL: RESPOND ONLY IN VALID JSON FORMAT**

You must respond with a valid JSON object for the static analysis evaluation. The JSON structure must be:

{{
  "evaluations": [
    {{
      "criterion_name": "Static Code Analysis",
      "max_points": 10,
      "score_obtained": numerical_score_obtained,
      "feedback_positive": "Detailed feedback on what was done correctly and good practices identified",
      "feedback_negative": "Detailed feedback on major flaws found by static analysis that caused point deductions",
      "score_justification": "Clear explanation of why this specific score was assigned out of 10 based on the static analysis findings"
    }}
  ]
}}

**IMPORTANT REQUIREMENTS:**
1. Respond ONLY with valid JSON - no additional text before or after
2. Score must be between 0 and 10
3. Provide detailed, constructive feedback based on static analysis findings
4. Use "Static Code Analysis" as the criterion name

Analyze the static analysis findings and provide your assessment in the exact JSON format requested.
"""
            return prompt
            
        except Exception as e:
            raise Exception(f"Failed to create static analysis prompt: {str(e)}")
    
    def _format_semgrep_findings(self, semgrep_results: Dict[str, Any]) -> str:
        """
        Format semgrep findings for inclusion in the prompt.
        
        Args:
            semgrep_results: Results from semgrep analysis
            
        Returns:
            Formatted string describing the findings
        """
        if not semgrep_results.get('success'):
            return f"Static analysis failed: {semgrep_results.get('error', 'Unknown error')}"
        
        findings_count = semgrep_results.get('findings_count', 0)
        if findings_count == 0:
            return "No static analysis findings detected. The code appears to follow the defined rules."
        
        findings = semgrep_results.get('findings', [])
        if not findings:
            return f"Static analysis detected {findings_count} potential issues, but details are not available."
        
        # Format findings
        formatted_findings = []
        for finding in findings:
            rule_id = finding.get('rule_id', 'Unknown rule')
            message = finding.get('message', 'No message')
            file_path = finding.get('file', 'Unknown file')
            line = finding.get('line', 'Unknown line')
            
            formatted_findings.append(f"- **{rule_id}** in {file_path}:{line}: {message}")
        
        return f"""Total findings: {findings_count}

Detailed findings:
{chr(10).join(formatted_findings[:20])}  # Limit to first 20 findings

{f'... and {findings_count - 20} more findings' if findings_count > 20 else ''}"""
    
    def get_content_files_status(self) -> Dict[str, bool]:
        """
        Get the status of all required content files.
        
        Returns:
            Dictionary mapping file names to their existence status
        """
        return {
            'assignment_details.txt': self.assignment_details_file.exists(),
            'instruction_content.txt': self.instruction_content_file.exists(),
            'general_rubric.txt': self.general_rubric_file.exists(),
            'specific_rubric.json': self.specific_rubric_file.exists(),
            'static_instructions.txt': self.static_instructions_file.exists()
        } 