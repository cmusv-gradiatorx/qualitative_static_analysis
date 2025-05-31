# AutoGrader - Graduate Assignment Evaluation System

An autograder for graduate-level software engineering assignments that uses Large Language Models (LLMs) for qualitative code analysis. The system processes entire codebases using repomix and provides comprehensive feedback based on customizable rubrics.

## Features

- **Multi-LLM Support**: Supports Gemini, OpenAI GPT, and Ollama (Llama) models with plug-and-play architecture
- **Token Management**: Automatic token counting and compression using repomix
- **Customizable Rubrics**: Flexible prompt system for different assignment requirements
- **Static Code Analysis**: Configurable Semgrep-based static analysis for code quality assessment

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
   #OLLAMA_MODEL=llama3.1
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

### Customizing Evaluation Criteria

The system creates default prompt templates that you can customize:

- **`prompts/rubric_content.txt`**: Define your grading rubric
- **`prompts/instruction_content.txt`**: Specify evaluation instructions
- **`prompts/static_instructions.txt`**: Define static analysis evaluation criteria
- **`config/semgrep_rules.yaml`**: Configure Semgrep rules for static analysis

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
Customize `prompts/static_instructions.txt` to define how the LLM should evaluate static analysis findings.

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
│   └── semgrep_rules.yaml    # Semgrep rules for static analysis
├── prompts/                   # Customizable prompt templates
├── temp/                      # Temporary processing files
└── logs/                      # Application logs
```

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

### Logging

- Application logs are saved to `logs/autograder.log`
- Adjust log level in `config.env`: `LOG_LEVEL=DEBUG`

## Extension Points

The system is designed for easy extension:

1. **Add new LLM providers**: Inherit from `LLMProvider` and register with `LLMFactory`
2. **Custom prompt templates**: Inherit from `PromptTemplate`
3. **New processing steps**: Extend the `AutoGrader` class
4. **Custom analyzers**: Add new modules following the existing patterns
