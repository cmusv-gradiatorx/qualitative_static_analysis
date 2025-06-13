# AutoGrader - Graduate Assignment Evaluation System

An autograder for graduate-level software engineering assignments that uses Large Language Models (LLMs) for qualitative code analysis. The system processes entire codebases using repomix and provides comprehensive feedback based on customizable rubrics.

## 🆕Features

- **Parallel LLM Processing**: Process multiple rubric criteria simultaneously for faster evaluation
- **Structured Rubric System**: JSON-based rubric definitions with detailed scoring criteria
- **Project-Specific Configurations**: Dynamic configuration based on assignment type
- **Enhanced Output Format**: Structured evaluation reports with detailed feedback
- **Raw Semgrep Output**: Separate files for detailed static analysis findings

## Features

- **Multi-LLM Support**: Supports Gemini, OpenAI GPT, and Ollama (Llama) models with plug-and-play architecture
- **Token Management**: Automatic token counting and compression using repomix
- **Customizable Rubrics**: Flexible JSON-based rubric system for different assignment requirements
- **Static Code Analysis**: Configurable Semgrep-based static analysis for code quality assessment
- **Parallel Processing**: Configurable parallel LLM evaluation for detailed rubric assessment
- **Project Management**: Dynamic configuration system for different courses and assignments

## Architecture

The system follows software engineering best practices with modular design:

- **Strategy Pattern**: For pluggable LLM providers
- **Factory Pattern**: For LLM provider creation
- **Facade Pattern**: For simplified API interface
- **Singleton Pattern**: For configuration management

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cmusv-gradiatorx/qualitative_static_analysis.git
   cd autograder
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js and repomix**
   ```bash
   # Install Node.js (version 18+)
   npm install -g repomix
   ```

## Configuration

1. **Copy the environment configuration**
   ```bash
   cp config.env.local config.env
   ```

2. **Edit `config.env` with your credentials**
   ```env
   # LLM Configuration
   # Choose the LLM provider: gemini, openai, ollama
   LLM_PROVIDER=ollama

   # Gemini Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-pro

   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-4o
   OPENAI_ORG_ID=optional

   # Ollama Configuration (for local Llama models)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=deepseek-r1

   # Application Configuration
   INPUT_FOLDER=input
   OUTPUT_FOLDER=output
   TEMP_FOLDER=temp

   # Repomix Configuration
   MAX_TOKENS=128000
   USE_COMPRESSION=true
   REMOVE_COMMENTS=false 

   # Semgrep Static Analysis Configuration
   ENABLE_SEMGREP_ANALYSIS=false
   SEMGREP_RULES_FILE=config/semgrep_rules.yaml
   SEMGREP_TIMEOUT=300 

   # Parallel Processing Configuration
   MAX_PARALLEL_LLM=2

   # Project Assignment Configuration
   PROJECT_ASSIGNMENT=functional_programming_milestone_3
   ```

## Usage

### Basic Usage

1. **Place ZIP files** in the `input/` directory
   - Each ZIP should contain a complete project/assignment
   - Multiple ZIP files can be processed in batch

2. **Run the autograder**
   ```bash
   python main.py
   ```

3. **Check results** in the `output/` directory
   - Each assignment gets a detailed evaluation report
   - Reports include grades, feedback, and improvement suggestions
   - Separate files for raw static analysis findings

### Project-Specific Configuration

The system supports dynamic configuration for different assignments and courses:

1. **Create project configuration** in `config/projects/[project_name].json`:
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

2. **Create project prompts** in `prompts/[project_name]/`:
   - `assignment_details.txt`: Assignment specifications
   - `instruction_content.txt`: Evaluation instructions
   - `general_rubric.txt`: Common rubric instructions
   - `specific_rubric.json`: Detailed rubric criteria
   - `static_instructions.txt`: Static analysis instructions

3. **Set project in configuration**:
   ```env
   PROJECT_ASSIGNMENT=your_project_name
   ```

### Customizing Evaluation Criteria

#### Structured Rubric System

Create a `specific_rubric.json` file with detailed criteria:

```json
[
  {
    "criterion_name": "Code Quality",
    "max_points": 2.0,
    "specific_prompt": "Evaluate code quality including readability, maintainability, and adherence to best practices.\n\n**Assessment Guidelines:**\n- **Excellent (1.8-2.0 points)**: Clean, well-structured code with excellent practices\n- **Good (1.4-1.7 points)**: Good code quality with minor issues\n- **Satisfactory (1.0-1.3 points)**: Adequate code with some quality issues\n- **Poor (0.0-0.9 points)**: Poor code quality with significant issues"
  }
]
```

#### Parallel Processing Configuration

Configure how many LLM instances run simultaneously:

```env
MAX_PARALLEL_LLM=3  # Global default
```

Or in project configuration:
```json
{
  "max_parallel_llm": 4  // Project-specific override
}
```

## LLM Provider Configuration

### Gemini (Google)
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-pro
```

### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_ORG_ID=your_org_id  # optional
```

### Ollama (Local)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

## Static Code Analysis

The system includes optional static code analysis using Semgrep rules:

### Enabling Static Analysis
Set `ENABLE_SEMGREP_ANALYSIS=true` in your configuration to enable static analysis.

### Configuring Semgrep Rules
Edit `config/semgrep_rules.yaml` to define custom rules for your assignments

### Static Analysis Instructions
Customize `prompts/[project]/static_instructions.txt` to define how the LLM should evaluate static analysis findings.

## Project Structure

```
autograder/
├── main.py                     # Application entry point
├── config.env                  # Environment configuration
├── requirements.txt            # Python dependencies
├── src/
│   ├── config/                 # Configuration management
│   │   └── settings.py
│   ├── llm/                    # LLM provider implementations
│   │   ├── base.py            # Abstract base class
│   │   ├── factory.py         # Factory for provider creation
│   │   ├── gemini_provider.py # Gemini implementation
│   │   ├── openai_provider.py # OpenAI implementation
│   │   └── ollama_provider.py # Ollama implementation
│   ├── prompts/               # Prompt management
│   │   └── prompt_manager.py
│   ├── repomix/               # Codebase processing
│   │   └── processor.py
│   ├── semgrep/               # Static code analysis
│   │   └── analyzer.py
│   ├── core/                  # Main application logic
│   │   └── autograder.py
│   └── utils/                 # Utilities
│       └── logger.py
├── input/                     # Place ZIP files here
├── output/                    # Evaluation reports appear here
├── config/                    # Configuration files
│   ├── projects/              # Project-specific configurations
│   │   └── [project_name].json
│   └── semgrep_rules.yaml    # Semgrep rules for static analysis
├── prompts/                   # Project-specific prompt templates
│   └── [project_name]/        # Project-specific prompts
│       ├── assignment_details.txt
│       ├── instruction_content.txt
│       ├── general_rubric.txt
│       ├── specific_rubric.json
│       └── static_instructions.txt
├── temp/                      # Temporary processing files
└── logs/                      # Application logs
```

## Enhanced Output Format

The system now generates structured evaluation reports:

### Main Evaluation Report
- Comprehensive rubric-based evaluation
- Parallel processing statistics
- Structured feedback for each criterion
- Static analysis evaluation (if enabled)

### Raw Semgrep Output
- Detailed static analysis findings
- Separate file for technical review
- Timestamped analysis results

## Parallel Processing

The system divides rubric criteria into groups for parallel evaluation:

1. **Automatic Division**: Rubrics are automatically divided based on `max_parallel_llm` setting
2. **Load Balancing**: Groups are balanced to optimize processing time
3. **Error Handling**: Failed groups are handled gracefully with error reporting
4. **Structured Output**: Results are combined into a comprehensive report

## Token Management

The system intelligently handles token limits:

1. **Token Counting**: Uses repomix's built-in token counting
2. **Automatic Compression**: Applies compression if content exceeds limits
3. **Model-Aware**: Respects each LLM's specific token limits
4. **Fallback Estimation**: Provides estimates when exact counts unavailable

## Troubleshooting

### Common Issues

1. **Repomix not found**
   - Ensure Node.js is installed
   - Run `npm install -g repomix`

2. **API Key errors**
   - Verify your API keys in `config.env`
   - Check that the correct provider is selected

3. **Token limit exceeded**
   - Enable compression: `USE_COMPRESSION=true`
   - Consider using a model with higher token limits

4. **ZIP extraction fails**
   - Ensure ZIP files contain valid project structures
   - Check file permissions

5. **Missing prompt files**
   - Ensure all required files exist in the project prompts directory
   - Check the error message for specific missing files

### Logging

- Application logs are saved to `logs/autograder.log`
- Adjust log level in `config.env`: `LOG_LEVEL=DEBUG`

## Extension Points

The system is designed for easy extension:

1. **Add new LLM providers**: Inherit from `LLMProvider` and register with `LLMFactory`
2. **Custom prompt templates**: Create new project-specific prompt directories
3. **New processing steps**: Extend the `AutoGrader` class
4. **Custom analyzers**: Add new modules following the existing patterns
5. **Project configurations**: Add new JSON configuration files for different assignments

## Testing

Run the test script to verify new features:

```bash
python test_new_features.py
```

This will test:
- Settings and project configuration loading
- Prompt manager functionality
- Rubric division for parallel processing
- Content file validation
