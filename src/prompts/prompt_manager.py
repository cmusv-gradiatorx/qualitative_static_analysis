"""
Prompt Manager

Manages loading and combining different prompt templates for the autograder.
Creates templates dynamically from instruction and rubric content files.

Author: Auto-generated
"""

from pathlib import Path
from typing import Dict, Any


class PromptManager:
    """
    Manages prompt templates and constructs final prompts.
    
    This class loads instruction and rubric content files and combines
    them with the codebase content to create the final LLM prompt.
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
        self.rubric_content_file = self.prompts_dir / "rubric_content.txt"
        self.instruction_content_file = self.prompts_dir / "instruction_content.txt"
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
        
        if not self.rubric_content_file.exists():
            missing_files.append(str(self.rubric_content_file))
        
        if not self.instruction_content_file.exists():
            missing_files.append(str(self.instruction_content_file))
        
        if missing_files:
            error_msg = (
                f"Required prompt content files are missing:\n"
                f"  - {chr(10).join(missing_files)}\n\n"
                f"Please create these files in the '{self.prompts_dir}' directory:\n"
                f"  - 'rubric_content.txt': Define your grading rubric\n"
                f"  - 'instruction_content.txt': Specify evaluation instructions\n"
                f"  - 'static_instructions.txt': Define static analysis evaluation instructions (optional)"
            )
            raise FileNotFoundError(error_msg)
    
    def load_rubric_content(self) -> str:
        """
        Load rubric content from file.
        
        Returns:
            Rubric content as string
            
        Raises:
            FileNotFoundError: If rubric content file doesn't exist
        """
        try:
            return self.rubric_content_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Rubric content file not found: {self.rubric_content_file}")
        except Exception as e:
            raise Exception(f"Error reading rubric content: {str(e)}")
    
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
    
    def _create_rubric_prompt(self, rubric_content: str) -> str:
        """
        Create the rubric prompt from content.
        
        Args:
            rubric_content: Raw rubric content
            
        Returns:
            Formatted rubric prompt
        """
        return f"""{rubric_content}

Please evaluate the codebase against these rubric criteria. For each criterion, provide:
1. Whether it is met, partially met, or not met
2. Specific evidence from the code
3. Suggestions for improvement if applicable
"""
    
    def _create_instruction_prompt(self, instruction_content: str) -> str:
        """
        Create the instruction prompt from content.
        
        Args:
            instruction_content: Raw instruction content
            
        Returns:
            Formatted instruction prompt
        """
        return f"""{instruction_content}

Please provide:
1. **Overall Assessment**: A comprehensive evaluation of the code quality
2. **Rubric Evaluation**: Detailed assessment against each rubric criterion
3. **Strengths**: What the code does well
4. **Areas for Improvement**: Specific issues and suggestions
5. **Detailed Feedback**: Actionable recommendations for improvement
5. **Score recommendation**: Suggested a score for the assignment on the basis of the rubric and the code. Make sure the score is a decimal between 0 and 10.

Format your response clearly with headers and bullet points for easy reading.
"""
    
    def _create_overall_prompt(self, rubric_prompt: str, instruction_prompt: str, codebase_content: str) -> str:
        """
        Create the complete prompt by combining all components.
        
        Args:
            rubric_prompt: Formatted rubric prompt
            instruction_prompt: Formatted instruction prompt
            codebase_content: Processed codebase content
            
        Returns:
            Complete prompt ready for LLM
        """
        return f"""You are an expert software engineering instructor grading a graduate-level assignment. You will evaluate the provided codebase against the given rubric and provide comprehensive feedback.

**RUBRIC:**
{rubric_prompt}

**INSTRUCTIONS:**
{instruction_prompt}

**CODEBASE TO EVALUATE:**
{codebase_content}

**INSTRUCTIONS:**
{instruction_prompt}

Please provide a thorough evaluation following the rubric and instructions above.
"""
    
    def create_final_prompt(self, codebase_content: str) -> str:
        """
        Create the final prompt by combining all templates with content.
        
        Args:
            codebase_content: The processed codebase content from repomix
            
        Returns:
            Complete prompt ready for LLM
            
        Raises:
            FileNotFoundError: If required content files are missing
            Exception: If there's an error processing the content
        """
        try:
            # Load content files
            rubric_content = self.load_rubric_content()
            instruction_content = self.load_instruction_content()
            
            # Create formatted prompts
            rubric_prompt = self._create_rubric_prompt(rubric_content)
            instruction_prompt = self._create_instruction_prompt(instruction_content)
            
            # Combine everything in the overall prompt
            final_prompt = self._create_overall_prompt(
                rubric_prompt, instruction_prompt, codebase_content
            )
            
            return final_prompt
            
        except Exception as e:
            raise Exception(f"Failed to create final prompt: {str(e)}")
    
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

Please analyze these static analysis findings and provide a comprehensive evaluation following the instructions above.
"""
            
            return prompt
            
        except Exception as e:
            raise Exception(f"Failed to create static analysis prompt: {str(e)}")
    
    def _format_semgrep_findings(self, semgrep_results: Dict[str, Any]) -> str:
        """
        Format semgrep findings for inclusion in the prompt.
        
        Args:
            semgrep_results: Semgrep analysis results
            
        Returns:
            Formatted findings text
        """
        if not semgrep_results['success']:
            return f"Static analysis failed: {semgrep_results.get('error', 'Unknown error')}"
        
        findings = semgrep_results['findings']
        summary = semgrep_results['summary']
        
        if not findings:
            return "No static analysis issues were found in the codebase."
        
        # Format summary
        summary_text = f"""## Analysis Summary
- **Total Issues Found:** {summary['total_findings']}
- **By Severity:** {', '.join(f"{sev}: {count}" for sev, count in summary['by_severity'].items())}
- **Files Affected:** {len(summary['by_file'])}

"""
        
        # Format detailed findings
        findings_text = "## Detailed Findings\n\n"
        for i, finding in enumerate(findings, 1):
            findings_text += f"""### Finding #{i}: {finding.rule_id}
- **File:** {finding.file_path} (Line {finding.line})
- **Severity:** {finding.severity}
- **Message:** {finding.message}
- **Code Snippet:**
```
{finding.code_snippet}
```

"""
        
        return summary_text + findings_text
    
    def get_content_files_status(self) -> Dict[str, bool]:
        """
        Get the status of required content files.
        
        Returns:
            Dictionary showing which files exist
        """
        return {
            'rubric_content.txt': self.rubric_content_file.exists(),
            'instruction_content.txt': self.instruction_content_file.exists(),
            'static_instructions.txt': self.static_instructions_file.exists()
        } 