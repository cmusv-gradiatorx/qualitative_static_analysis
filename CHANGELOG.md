# AutoGrader Changelog

## Version 2.1.0 - FAISS Historical Context System & Multi-Language Refactoring

### 🚀 Major New Features

#### FAISS-Based Historical Context System
- **Optional Historical Context**: Provides LLMs with relevant examples from past submissions
- **Multi-Layer Code Embeddings**: Combines semantic (CodeBERT), structural (AST), and graph features
- **FAISS Vector Database**: Efficient similarity search with multiple index types (flat, IVF, HNSW)
- **Automatic Submission Processing**: Extracts submissions from ZIP files with metadata parsing
- **Assignment-Specific Context**: Filters historical examples by assignment for relevance
- **Plagiarism Detection**: Analyze submission uniqueness and potential plagiarism
- **Apple Silicon Support**: Fully compatible with M1/M2/M3 MacBook (Air/Pro)

#### Multi-Language Code Analysis (Refactored)
- **Restructured Analyzers**: Moved to `src/faiss/analyzers/` directory for better organization
- **Base Interface**: `LanguageAnalyzer` abstract base class for consistent implementation
- **Pluggable Architecture**: Easy addition of new language analyzers
- **Current Support**: Python (AST-based) and Java (regex-based) analyzers
- **Language Detection**: Automatic detection from file extensions
- **Cross-Language Dependencies**: Track dependencies between different programming languages
- **Unified Metrics**: Normalized metrics across all supported languages

### 🔧 Enhanced Components

#### FAISS Integration (`src/faiss/`)
- **HybridCodeEmbedder**: Multi-layer embedding generation with 768+ dimensions
- **SubmissionProcessor**: ZIP file processing with automatic score/feedback extraction
- **FAISSManager**: Vector database management with save/load capability
- **HistoricalContextProvider**: Integration with autograder prompt system
- **Graceful Degradation**: System works without FAISS when not available

#### Refactored Analyzer Structure
- **src/faiss/analyzers/base_analyzer.py**: Abstract interface for all analyzers
- **src/faiss/analyzers/python_analyzer.py**: Comprehensive Python analysis using AST
- **src/faiss/analyzers/java_analyzer.py**: Java analysis using regex patterns
- **src/faiss/analyzers/multi_language_analyzer.py**: Coordinating multi-language analyzer
- **Backward Compatibility**: Maintained for existing integrations

#### Settings Management (`src/config/settings.py`)
- Added `enable_historical_context` configuration option
- Added `faiss_index_path` for FAISS index location
- Added `historical_context_examples` to control number of examples
- Added `similarity_threshold` for context relevance filtering
- Added `code_embedding_model` configuration
- Enhanced `get_faiss_config()` method for comprehensive FAISS settings

#### AutoGrader Core (`src/core/autograder.py`)
- **FAISS Integration**: Automatic FAISS initialization when enabled
- **Historical Context Generation**: Seamless integration with prompt creation
- **Enhanced Logging**: Detailed statistics about historical context usage
- **Error Handling**: Graceful fallback when FAISS unavailable
- **Statistics Reporting**: Index status, submissions count, and assignments

### 📁 New File Structure

#### FAISS Historical Context System
```
src/faiss/
├── __init__.py                              # Updated module exports
├── analyzers/                               # Language analyzers directory
│   ├── __init__.py                         # Analyzer module initialization
│   ├── base_analyzer.py                    # Abstract analyzer interface
│   ├── python_analyzer.py                 # Python AST-based analyzer
│   ├── java_analyzer.py                   # Java regex-based analyzer
│   └── multi_language_analyzer.py         # Multi-language coordinator
├── embedder.py                             # Hybrid embedding generation
├── faiss_manager.py                        # FAISS vector database management
├── historical_context.py                  # Context provider for LLM integration
├── processor.py                            # ZIP submission processing
├── build_index.py                          # Index building script
├── data/                                   # Historical submission data
│   └── GildedRoseKata_submissions.zip
├── MULTI_LANGUAGE_GUIDE.md                # Comprehensive guide
└── README.md                               # FAISS system documentation
```

### ⚙️ Configuration Enhancements

#### New Environment Variables
```env
# FAISS Historical Context Configuration
ENABLE_HISTORICAL_CONTEXT=false
FAISS_INDEX_PATH=src/faiss/index
HISTORICAL_CONTEXT_EXAMPLES=3
SIMILARITY_THRESHOLD=0.3
INCLUDE_ASSIGNMENT_STATS=true
CODE_EMBEDDING_MODEL=microsoft/codebert-base
FAISS_INDEX_TYPE=flat
```

#### Installation Requirements
```bash
# Core FAISS dependencies
pip install faiss-cpu  # For CPU-only (recommended for most users)
pip install faiss-gpu  # For GPU acceleration

# Optional: Better embeddings
pip install torch transformers sentence-transformers

# Analysis dependencies  
pip install numpy pandas scikit-learn networkx
```

### 🎯 Educational Benefits

#### Enhanced Assignment Evaluation
- **Consistent Grading**: Historical context helps maintain consistency across evaluations
- **Calibration**: LLMs receive examples of similar submissions with scores
- **Pattern Recognition**: Identify common approaches and their relative quality
- **Learning Analytics**: Track coding patterns and improvement over time

#### Plagiarism Detection
- **Similarity Analysis**: Identify potentially plagiarized submissions
- **Threshold Configuration**: Adjustable similarity thresholds per assignment
- **Historical Comparison**: Compare against all past submissions
- **Detailed Reports**: Similarity scores and potentially problematic cases

#### Multi-Language Projects
- **Full-Stack Support**: Handle JavaScript frontend + Python/Java backend
- **Polyglot Analysis**: Analyze projects using multiple programming languages
- **Architecture Assessment**: Evaluate cross-language design decisions
- **Language Distribution**: Track usage patterns across different languages

### 🔄 Migration Guide

#### Enabling Historical Context (Optional)
1. **Install FAISS**:
   ```bash
   pip install faiss-cpu  # For Apple Silicon M1/M2/M3 compatibility
   ```

2. **Update Configuration**:
   ```bash
   # Add to config.env
   echo "ENABLE_HISTORICAL_CONTEXT=true" >> config.env
   echo "FAISS_INDEX_PATH=src/faiss/index" >> config.env
   ```

3. **Build Historical Index** (when you have historical data):
   ```bash
   python src/faiss/build_index.py --zip-path path/to/historical_submissions.zip
   ```

#### Code Compatibility
- **Analyzer Imports**: Updated imports use new `src.faiss.analyzers` structure
- **Backward Compatibility**: Existing code continues to work unchanged
- **Optional Features**: All FAISS features are optional and fail gracefully

### 📈 Performance Improvements

#### FAISS Search Performance
- **Fast Similarity Search**: Sub-second search over thousands of submissions
- **Memory Efficiency**: Optimized embedding storage and retrieval
- **Scalable Indexing**: Support for flat, IVF, and HNSW index types
- **Apple Silicon Optimized**: Native performance on M1/M2/M3 Macs

#### Multi-Language Analysis
- **Efficient Parsing**: AST-based Python analysis with regex fallbacks
- **Cross-Language Insights**: Dependency tracking between languages
- **Normalized Metrics**: Consistent feature extraction across languages
- **Pluggable Design**: Easy addition of new language support

### 🐛 Bug Fixes and Improvements

- **Enhanced Error Handling**: Graceful degradation when FAISS unavailable
- **Improved Logging**: Detailed status reporting for FAISS operations
- **Better File Processing**: Robust ZIP extraction and content analysis
- **Memory Management**: Efficient handling of large embedding datasets
- **Configuration Validation**: Comprehensive validation of FAISS settings

### 🔧 Developer Experience

#### Comprehensive Documentation
- **FAISS Guide**: Step-by-step setup and usage instructions
- **Multi-Language Guide**: Comprehensive guide for language analyzer extension
- **API Documentation**: Detailed class and method documentation
- **Example Code**: Working examples for common use cases

#### Testing and Validation
- **Index Building Scripts**: Automated FAISS index creation
- **Validation Tools**: Test embedding quality and search performance
- **Debug Support**: Detailed logging and error reporting
- **M3 Mac Testing**: Verified compatibility with Apple Silicon

---

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