# AutoGrader Changelog

## Version 2.0.0 - Enhanced Parallel Processing and Structured Rubrics

### 🚀 Major New Features

#### Parallel LLM Processing
- **Configurable Parallelism**: Set `MAX_PARALLEL_LLM` to control how many LLM instances run simultaneously
- **Automatic Rubric Division**: Rubrics are automatically divided into groups for parallel processing
- **Load Balancing**: Groups are balanced to optimize processing time
- **Error Handling**: Failed groups are handled gracefully with error reporting

#### Structured Rubric System
- **JSON-Based Rubrics**: Replace text-based rubrics with structured JSON format
- **Detailed Scoring Criteria**: Each rubric criterion includes specific scoring guidelines
- **Flexible Point System**: Support for decimal point scoring with detailed justification
- **Structured Output Format**: Consistent evaluation format across all criteria

#### Project-Specific Configuration
- **Dynamic Configuration**: Different settings for different assignments/courses
- **Project-Specific Prompts**: Separate prompt directories for each project
- **Configurable File Patterns**: Project-specific ignore/keep patterns for file processing
- **Assignment-Specific Settings**: Override global settings per project

### 🔧 Enhanced Components

#### Settings Management (`src/config/settings.py`)
- Added `max_parallel_llm` configuration option
- Added `project_assignment` for dynamic project selection
- Added `get_project_config()` method for project-specific settings
- Added `get_prompts_dir()` method for project-specific prompt directories
- Enhanced validation for new configuration options

#### Prompt Manager (`src/prompts/prompt_manager.py`)
- **Complete Rewrite**: New architecture supporting structured rubrics
- Added support for `assignment_details.txt`, `general_rubric.txt`, and `specific_rubric.json`
- Added `divide_rubrics_for_parallel_processing()` method
- Added `create_rubric_evaluation_prompt()` for individual rubric groups
- Enhanced static analysis prompt creation with structured output format
- Added comprehensive validation for rubric JSON structure

#### AutoGrader Core (`src/core/autograder.py`)
- **Parallel Processing**: Added `_process_rubrics_parallel()` using ThreadPoolExecutor
- **Enhanced Evaluation**: Added `_evaluate_rubric_group()` for individual group processing
- **Structured Output**: Added `_create_comprehensive_evaluation_report()`
- **Raw Semgrep Output**: Added `_format_semgrep_raw_output()` for detailed analysis
- **Multiple Output Files**: Support for main evaluation and raw semgrep output files
- Project-specific configuration integration
- Enhanced logging and error handling

#### Main Application (`main.py`)
- **Enhanced UI**: Rich table display for processing results
- **Detailed Status**: Show configuration and processing statistics
- **Better Error Reporting**: Comprehensive error display and handling
- **Output File Management**: Display multiple output files per assignment

### 📁 New File Structure

#### Project Configuration
```
config/projects/[project_name].json
```
- Project-specific settings override global configuration
- Configurable file patterns, parallel processing, and analysis settings

#### Project-Specific Prompts
```
prompts/[project_name]/
├── assignment_details.txt      # Assignment specifications
├── instruction_content.txt     # Evaluation instructions  
├── general_rubric.txt         # Common rubric instructions
├── specific_rubric.json       # Structured rubric criteria
└── static_instructions.txt    # Static analysis instructions
```

### 🎯 Improved Output Format

#### Main Evaluation Report
- Comprehensive header with processing statistics
- Parallel processing information
- Structured rubric evaluations with consistent formatting
- Static analysis evaluation (if enabled)
- Enhanced footer with version information

#### Raw Semgrep Output
- Separate file for detailed static analysis findings
- Timestamped analysis results
- Structured finding reports with rule IDs, files, lines, and messages
- Technical review-friendly format

### ⚙️ Configuration Enhancements

#### New Environment Variables
```env
# Parallel Processing Configuration
MAX_PARALLEL_LLM=2

# Project Assignment Configuration  
PROJECT_ASSIGNMENT=functional_programming_milestone_3
```

#### Project Configuration Schema
```json
{
  "max_file_size": 128000,
  "ignore_patterns": ["*.pyc", "__pycache__", ".git"],
  "keep_patterns": ["*.py", "*.md", "*.txt"],
  "max_parallel_llm": 3,
  "semgrep_rules_file": "config/semgrep_rules.yaml",
  "semgrep_timeout": 300
}
```

#### Structured Rubric Schema
```json
[
  {
    "criterion_name": "Criterion Name",
    "max_points": 2.0,
    "specific_prompt": "Detailed evaluation instructions with scoring guidelines"
  }
]
```

### 🧪 Testing and Validation

#### New Test Script
- `test_new_features.py`: Comprehensive testing of new functionality
- Settings validation and project configuration loading
- Prompt manager functionality testing
- Rubric division and parallel processing validation

### 🔄 Migration Guide

#### From Version 1.x to 2.0

1. **Update Configuration**:
   ```bash
   # Add new settings to config.env
   echo "MAX_PARALLEL_LLM=2" >> config.env
   echo "PROJECT_ASSIGNMENT=your_project_name" >> config.env
   ```

2. **Create Project Structure**:
   ```bash
   mkdir -p config/projects
   mkdir -p prompts/your_project_name
   ```

3. **Migrate Prompts**:
   - Move existing `prompts/rubric_content.txt` content to `prompts/your_project_name/general_rubric.txt`
   - Move existing `prompts/instruction_content.txt` to `prompts/your_project_name/instruction_content.txt`
   - Create new `assignment_details.txt` and `specific_rubric.json` files

4. **Create Project Configuration**:
   ```json
   {
     "max_parallel_llm": 2,
     "max_file_size": 128000
   }
   ```

### 🐛 Bug Fixes and Improvements

- Enhanced error handling for missing prompt files
- Improved token counting and compression logic
- Better logging and status reporting
- More robust file cleanup and processing
- Enhanced validation for configuration files

### 📈 Performance Improvements

- **Parallel Processing**: Significant speed improvement for multi-criteria evaluations
- **Optimized Token Management**: Better handling of large codebases
- **Efficient File Processing**: Improved repomix integration
- **Reduced Memory Usage**: Better temporary file management

### 🔧 Developer Experience

- **Comprehensive Documentation**: Updated README with all new features
- **Test Coverage**: New test scripts for validation
- **Error Messages**: More descriptive error messages and troubleshooting guides
- **Logging**: Enhanced logging for debugging and monitoring

---

## Version 1.x - Legacy Features

### Core Features (Maintained)
- Multi-LLM support (Gemini, OpenAI, Ollama)
- Repomix integration for codebase processing
- Semgrep static analysis
- Token management and compression
- Configurable evaluation criteria

### Architecture (Enhanced)
- Strategy pattern for LLM providers
- Factory pattern for provider creation
- Facade pattern for simplified API
- Singleton pattern for configuration management 