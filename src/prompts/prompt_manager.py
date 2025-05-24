"""
Prompt Manager

Manages loading and combining different prompt templates for the autograder.
Uses Template Method pattern for prompt construction.

Author: Auto-generated
"""

from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    """
    Abstract base class for prompt templates.
    
    Implements Template Method pattern for prompt construction.
    """
    
    def __init__(self, template_path: Path):
        """
        Initialize the prompt template.
        
        Args:
            template_path: Path to the template file
        """
        self.template_path = template_path
        self._template_content = self._load_template()
    
    def _load_template(self) -> str:
        """Load template content from file."""
        try:
            return self.template_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        except Exception as e:
            raise Exception(f"Error loading template {self.template_path}: {str(e)}")
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        pass


class SimplePromptTemplate(PromptTemplate):
    """Simple prompt template that supports basic string formatting."""
    
    def format(self, **kwargs) -> str:
        """Format template using string format method."""
        try:
            return self._template_content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template: {str(e)}")


class PromptManager:
    """
    Manages prompt templates and constructs final prompts.
    
    This class is responsible for loading prompt templates and combining
    them with the codebase content to create the final LLM prompt.
    """
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize template instances
        self.rubric_template = SimplePromptTemplate(
            self.prompts_dir / "rubric_template.txt"
        )
        self.instruction_template = SimplePromptTemplate(
            self.prompts_dir / "instruction_template.txt"
        )
        self.overall_template = SimplePromptTemplate(
            self.prompts_dir / "overall_template.txt"
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self) -> None:
        """Create default prompt templates if they don't exist."""
        
        # Rubric template
        rubric_template_path = self.prompts_dir / "rubric_template.txt"
        if not rubric_template_path.exists():
            rubric_content = """{rubric_content}

Please evaluate the codebase against these rubric criteria. For each criterion, provide:
1. Whether it is met, partially met, or not met
2. Specific evidence from the code
3. Suggestions for improvement if applicable
"""
            rubric_template_path.write_text(rubric_content, encoding='utf-8')
        
        # Instruction template  
        instruction_template_path = self.prompts_dir / "instruction_template.txt"
        if not instruction_template_path.exists():
            instruction_content = """{instruction_content}

Please provide:
1. **Overall Assessment**: A comprehensive evaluation of the code quality
2. **Rubric Evaluation**: Detailed assessment against each rubric criterion
3. **Strengths**: What the code does well
4. **Areas for Improvement**: Specific issues and suggestions
5. **Grade Recommendation**: Suggested grade with justification
6. **Detailed Feedback**: Actionable recommendations for improvement

Format your response clearly with headers and bullet points for easy reading.
"""
            instruction_template_path.write_text(instruction_content, encoding='utf-8')
        
        # Overall template
        overall_template_path = self.prompts_dir / "overall_template.txt"
        if not overall_template_path.exists():
            overall_content = """You are an expert software engineering instructor grading a graduate-level assignment. You will evaluate the provided codebase against the given rubric and provide comprehensive feedback.

**RUBRIC:**
{rubric_prompt}

**INSTRUCTIONS:**
{instruction_prompt}

**CODEBASE TO EVALUATE:**
{codebase_content}

Please provide a thorough evaluation following the rubric and instructions above.
"""
            overall_template_path.write_text(overall_content, encoding='utf-8')
        
        # Create content files for instructor customization
        rubric_content_path = self.prompts_dir / "rubric_content.txt"
        if not rubric_content_path.exists():
            default_rubric = """# Graduate Software Engineering Assignment Rubric

## Code Quality (25 points)
- **Excellent (23-25)**: Clean, well-structured code with consistent style
- **Good (18-22)**: Generally well-written with minor style issues
- **Fair (13-17)**: Adequate code quality with some structural problems
- **Poor (0-12)**: Poorly structured or difficult to read code

## Design Patterns & Architecture (25 points)
- **Excellent (23-25)**: Appropriate use of design patterns, solid architecture
- **Good (18-22)**: Good architectural decisions with minor issues
- **Fair (13-17)**: Basic architecture with some design flaws
- **Poor (0-12)**: Poor architectural choices or missing design patterns

## Documentation & Comments (20 points)
- **Excellent (18-20)**: Comprehensive documentation and meaningful comments
- **Good (14-17)**: Good documentation with minor gaps
- **Fair (10-13)**: Basic documentation, some important areas missing
- **Poor (0-9)**: Minimal or poor documentation

## Functionality & Testing (20 points)
- **Excellent (18-20)**: Complete functionality with comprehensive tests
- **Good (14-17)**: Good functionality with adequate testing
- **Fair (10-13)**: Basic functionality with limited testing
- **Poor (0-9)**: Incomplete or non-functional code

## Best Practices (10 points)
- **Excellent (9-10)**: Follows all software engineering best practices
- **Good (7-8)**: Follows most best practices with minor violations
- **Fair (5-6)**: Some adherence to best practices
- **Poor (0-4)**: Poor adherence to best practices
"""
            rubric_content_path.write_text(default_rubric, encoding='utf-8')
        
        instruction_content_path = self.prompts_dir / "instruction_content.txt"
        if not instruction_content_path.exists():
            default_instructions = """Please evaluate this graduate-level software engineering assignment thoroughly. Focus on:

1. **Code Quality**: Assess readability, maintainability, and adherence to coding standards
2. **Design & Architecture**: Evaluate the overall design decisions and architectural patterns
3. **Documentation**: Review the quality and completeness of documentation and comments
4. **Functionality**: Assess whether the code meets requirements and includes appropriate testing
5. **Best Practices**: Check adherence to software engineering best practices

For each area, provide specific examples from the code and actionable feedback for improvement.
"""
            instruction_content_path.write_text(default_instructions, encoding='utf-8')
    
    def load_rubric_content(self) -> str:
        """Load rubric content from file."""
        rubric_file = self.prompts_dir / "rubric_content.txt"
        return rubric_file.read_text(encoding='utf-8')
    
    def load_instruction_content(self) -> str:
        """Load instruction content from file."""
        instruction_file = self.prompts_dir / "instruction_content.txt"
        return instruction_file.read_text(encoding='utf-8')
    
    def create_final_prompt(self, codebase_content: str) -> str:
        """
        Create the final prompt by combining all templates with content.
        
        Args:
            codebase_content: The processed codebase content from repomix
            
        Returns:
            Complete prompt ready for LLM
        """
        # Load content files
        rubric_content = self.load_rubric_content()
        instruction_content = self.load_instruction_content()
        
        # Format individual templates
        rubric_prompt = self.rubric_template.format(rubric_content=rubric_content)
        instruction_prompt = self.instruction_template.format(instruction_content=instruction_content)
        
        # Combine everything in the overall template
        final_prompt = self.overall_template.format(
            rubric_prompt=rubric_prompt,
            instruction_prompt=instruction_prompt,
            codebase_content=codebase_content
        )
        
        return final_prompt 