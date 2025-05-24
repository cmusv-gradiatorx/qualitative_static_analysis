# AutoGrader - Graduate Assignment Evaluation System

A sophisticated autograder for graduate-level software engineering assignments that uses Large Language Models (LLMs) for qualitative code analysis. The system processes entire codebases using repomix and provides comprehensive feedback based on customizable rubrics.

## Features

- **Multi-LLM Support**: Supports Gemini, OpenAI GPT, and Ollama (Llama) models with plug-and-play architecture
- **Intelligent Token Management**: Automatic token counting and compression using repomix
- **Customizable Rubrics**: Flexible prompt system for different assignment requirements
- **Comprehensive Analysis**: Evaluates code quality, architecture, documentation, and best practices
- **Professional Reporting**: Detailed evaluation reports with processing statistics

## Architecture

The system follows software engineering best practices with modular design:

- **Strategy Pattern**: For pluggable LLM providers
- **Factory Pattern**: For LLM provider creation
- **Template Method Pattern**: For prompt construction
- **Facade Pattern**: For simplified API interface
- **Singleton Pattern**: For configuration management

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
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
   cp config.env config.env.local
   ```

2. **Edit `config.env` with your credentials**
   ```env
   # Choose your LLM provider
   LLM_PROVIDER=gemini  # or openai, ollama
   
   # Add your API keys
   GEMINI_API_KEY=your_actual_api_key_here
   OPENAI_API_KEY=your_actual_api_key_here
   
   # Configure token limits and processing
   MAX_TOKENS=128000
   USE_COMPRESSION=true
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
- **`prompts/rubric_template.txt`**: Template for rubric formatting
- **`prompts/instruction_template.txt`**: Template for instruction formatting
- **`prompts/overall_template.txt`**: Overall prompt structure

### Example Rubric Customization

Edit `prompts/rubric_content.txt`:

```markdown
# Software Design Assignment Rubric

## Design Patterns (30 points)
- **Excellent (27-30)**: Appropriate use of design patterns
- **Good (21-26)**: Generally good pattern usage
- **Fair (15-20)**: Basic patterns with some issues
- **Poor (0-14)**: Inappropriate or missing patterns

## Code Quality (25 points)
- Clean, readable code with consistent style
- Proper error handling and validation
- Meaningful variable and function names

## Testing (25 points)
- Comprehensive unit tests
- Integration tests where appropriate
- Good test coverage

## Documentation (20 points)
- Clear README with setup instructions
- Well-commented code
- API documentation if applicable
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
│   ├── core/                  # Main application logic
│   │   └── autograder.py
│   └── utils/                 # Utilities
│       └── logger.py
├── input/                     # Place ZIP files here
├── output/                    # Evaluation reports appear here
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

## Example Output

```markdown
# AutoGrader Evaluation Report

**Assignment:** student_project.zip
**Project Name:** student_project
**Evaluation Date:** 2024-01-15 14:30:22
**LLM Provider:** GeminiProvider(model=gemini-1.5-pro)

## Processing Statistics

- **Original Token Count:** 45,230
- **Compression Applied:** No
- **Within Token Limit:** Yes
- **Token Limit:** 128,000

## Evaluation Results

### Overall Assessment
This is a well-structured project that demonstrates good understanding of software engineering principles...

### Rubric Evaluation
**Code Quality (23/25)**: The code is generally clean and well-organized...
**Design Patterns (18/25)**: Some design patterns are used appropriately...
...

### Strengths
- Clean, readable code structure
- Good separation of concerns
- Comprehensive error handling

### Areas for Improvement
- Consider implementing the Strategy pattern for payment processing
- Add more comprehensive unit tests
- Improve documentation for complex algorithms

### Grade Recommendation: 85/100
```

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

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive docstrings and comments
3. Include appropriate error handling
4. Write tests for new functionality
5. Update documentation for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details. 