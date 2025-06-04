# FAISS-Based Historical Context System

This module provides historical context for LLM evaluations by finding similar past submissions using multi-layer code embeddings and FAISS similarity search.

## Overview

The historical context system enhances autograder evaluations by providing LLMs with relevant examples from past submissions. This helps maintain consistency in grading and provides better calibration for scores and feedback.

## Architecture

### Multi-Layer Code Embeddings

The system uses a hybrid approach combining three types of features:

1. **Semantic Embeddings** (768 dimensions)
   - Uses CodeBERT or similar transformer models
   - Captures semantic meaning and programming patterns
   - Handles code understanding at the language level

2. **Structural Features** (Variable dimensions)
   - Multi-language AST/regex-based metrics: classes, functions, complexity, nesting depth
   - Code quality indicators: imports, decorators, error handling
   - Statistical aggregations: mean, std, max values per metric
   - Language diversity analysis: cross-language dependencies and ratios

3. **Dependency Graph Features** (20 dimensions)
   - Multi-language file-level dependency analysis using NetworkX
   - Centrality measures: in-degree, out-degree, betweenness
   - Graph topology: density, components, clustering coefficient
   - Cross-language dependency tracking and language connectivity

### FAISS Vector Database

- **Index Types**: Supports flat (exact), IVF (approximate fast), and HNSW (very fast)
- **Similarity Metric**: Cosine similarity using normalized L2 vectors
- **Scalability**: Efficient search over thousands of submissions
- **Persistence**: Save/load capability with metadata preservation

## Components

### Core Classes

1. **`HybridCodeEmbedder`** - Multi-layer embedding generation with multi-language support
2. **`SubmissionProcessor`** - ZIP file processing and metadata extraction
3. **`FAISSManager`** - Vector database management and similarity search
4. **`HistoricalContextProvider`** - Integration with autograder prompts
5. **Multi-Language Analyzers** (in `src/faiss/analyzers/`):
   - **`LanguageAnalyzer`** - Abstract base interface for all analyzers
   - **`PythonAnalyzer`** - AST-based Python analysis
   - **`JavaAnalyzer`** - Regex-based Java analysis
   - **`MultiLanguageCodeAnalyzer`** - Coordinating analyzer for multiple languages
   - **`MultiLanguageDependencyGraphBuilder`** - Cross-language dependency analysis

### Supported Languages

- **Python** (.py, .pyw) - Full AST analysis with comprehensive metrics
- **Java** (.java) - Regex-based analysis with method/class detection
- **Extensible Architecture** - Easy addition of new language analyzers

### Key Features

- **Multi-Language Support**: Analyzes Python and Java codebases with cross-language dependencies
- **Automatic ZIP Processing**: Extracts code files and attempts to parse scores/feedback
- **Flexible Metadata**: Supports JSON metadata and various feedback file formats
- **Robust Error Handling**: Graceful degradation when models unavailable
- **Normalization**: Statistical scaling of structural and graph features
- **Assignment Filtering**: Search within specific assignments for relevant context
- **Language Detection**: Automatic detection and appropriate analyzer selection
- **Apple Silicon Support**: Optimized for M1/M2/M3 MacBook (Air/Pro)

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install numpy pandas scikit-learn networkx

# FAISS (choose one - CPU recommended for M1/M2/M3 Macs)
pip install faiss-cpu  # CPU version (Apple Silicon compatible)
pip install faiss-gpu  # GPU version (requires CUDA)

# Optional: For better embeddings
pip install torch transformers sentence-transformers
```

### Directory Structure

```
src/faiss/
├── __init__.py                      # Module initialization with multi-language exports
├── analyzers/                      # Language-specific analyzers
│   ├── __init__.py                 # Analyzer module initialization
│   ├── base_analyzer.py            # Abstract LanguageAnalyzer interface
│   ├── python_analyzer.py          # Python AST-based analyzer
│   ├── java_analyzer.py            # Java regex-based analyzer
│   └── multi_language_analyzer.py  # Multi-language coordinator
├── embedder.py                     # Hybrid embedding generation
├── faiss_manager.py                # FAISS vector database management
├── historical_context.py           # Context provider for LLM integration
├── processor.py                    # Submission processing from ZIP files
├── build_index.py                  # Script to build FAISS index
├── data/                           # Historical submission data
│   └── GildedRoseKata_submissions.zip
├── README.md                       # This documentation
└── MULTI_LANGUAGE_GUIDE.md        # Comprehensive multi-language guide
```

## Usage

### 1. Building the Historical Index

```python
from src.faiss import SubmissionProcessor, HybridCodeEmbedder, FAISSManager

# Initialize components
zip_path = "src/faiss/data/GildedRoseKata_submissions.zip"
processor = SubmissionProcessor(zip_path)
embedder = HybridCodeEmbedder()  # Automatically uses multi-language analyzers
faiss_manager = FAISSManager()

# Extract submissions
submissions = processor.extract_submissions()

# Build FAISS index
build_stats = faiss_manager.build_index(
    submissions=submissions,
    embedder=embedder,
    save_path="src/faiss/index"
)

print(f"Built index with {build_stats['valid_embeddings']} submissions")
```

### 2. Multi-Language Analysis

```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

# Analyze mixed-language codebase
analyzer = MultiLanguageCodeAnalyzer()

code_files = {
    'Main.java': 'public class Main { public static void main(String[] args) {...} }',
    'utils.py': 'def helper_function(): return "Python helper"',
    'config.py': 'import json\nclass Config: pass'
}

# Get comprehensive analysis
results = analyzer.analyze_codebase(code_files)

print(f"Languages detected: {results['language_diversity']}")
print(f"Primary language: {results['primary_language']}")
print(f"Python files: {results.get('python_file_ratio', 0):.1%}")
print(f"Java files: {results.get('java_file_ratio', 0):.1%}")
print(f"Total functions: {results['num_functions_total']}")
```

### 3. Integration with AutoGrader

The system integrates with the existing autograder through the `HistoricalContextProvider`:

```python
from src.faiss import FAISSManager, HybridCodeEmbedder, HistoricalContextProvider

# Load existing index
faiss_manager = FAISSManager()
faiss_manager.load_index("src/faiss/index")

embedder = HybridCodeEmbedder()
embedder.load_scalers("src/faiss/index/scalers.pkl")

# Create context provider
context_provider = HistoricalContextProvider(faiss_manager, embedder)

# Get historical context for a multi-language submission
code_files = {
    "main.py": "def process_data():\n    pass",
    "Database.java": "public class Database { }"
}
assignment_id = "FullStackProject"

context = context_provider.get_context_for_prompt_manager(
    code_files=code_files,
    assignment_id=assignment_id,
    max_examples=3
)

print(context)  # Formatted text for LLM prompt with multi-language context
```

### 4. Searching Similar Submissions

```python
# Search for similar submissions
similar = faiss_manager.search_by_code_similarity(
    code_files=code_files,
    embedder=embedder,
    assignment_id="FullStackProject",
    top_k=5
)

for result in similar:
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Score: {result['score']}")
    print(f"Languages: {result.get('languages', 'N/A')}")
    print(f"Feedback: {result['feedback'][:100]}...")
```

### 5. Analyzing Submission Uniqueness

```python
# Check for potential plagiarism across languages
uniqueness = context_provider.analyze_submission_uniqueness(
    code_files=code_files,
    assignment_id="FullStackProject",
    similarity_threshold=0.8
)

print(f"Uniqueness: {uniqueness['uniqueness_assessment']}")
if uniqueness['potentially_plagiarized']:
    print("⚠️ High similarity to existing submissions detected")
    print(f"Language distribution: {uniqueness.get('language_analysis', {})}")
```

## Configuration

### Environment Variables

Add these to your `config.env`:

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

### Project Configuration

Add to your project config JSON:

```json
{
  "enable_historical_context": true,
  "faiss_index_path": "src/faiss/index",
  "historical_context_examples": 3,
  "similarity_threshold": 0.3,
  "include_assignment_stats": true,
  "supported_languages": ["python", "java"],
  "multi_language_analysis": true
}
```

## Adding New Language Analyzers

### 1. Create Language Analyzer

```python
# src/faiss/analyzers/rust_analyzer.py
from .base_analyzer import LanguageAnalyzer

class RustAnalyzer(LanguageAnalyzer):
    def get_supported_extensions(self) -> List[str]:
        return ['.rs']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        # Implement Rust-specific analysis
        return {
            'num_functions': len(re.findall(r'\bfn\s+\w+', content)),
            'num_structs': len(re.findall(r'\bstruct\s+\w+', content)),
            # ... other metrics
        }
    
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        # Implement Rust dependency extraction
        return []
```

### 2. Register the Analyzer

```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer
from src.faiss.analyzers.rust_analyzer import RustAnalyzer

analyzer = MultiLanguageCodeAnalyzer()
analyzer.add_language_analyzer('rust', RustAnalyzer())
```

## Performance Considerations

### Embedding Generation
- **Multi-Language CodeBERT**: ~1-2 seconds per submission on CPU, ~0.1s on GPU
- **Python AST Analysis**: ~0.005 seconds per file
- **Java Regex Analysis**: ~0.01 seconds per file
- **Graph Analysis**: ~0.01-0.02 seconds per multi-language submission

### Search Performance
- **Flat Index**: O(n) search, exact results
- **IVF Index**: O(log n) search, 95%+ recall
- **HNSW Index**: O(log n) search, 99%+ recall, fastest

### Memory Usage
- **Multi-Language Embeddings**: ~820 bytes per submission (820-dim vectors)
- **Metadata**: ~1-2 KB per submission (depends on feedback length)
- **Index**: 2-3x embedding size depending on index type
- **Language Analysis**: Minimal overhead (~50 bytes per file)

## Apple Silicon Compatibility

### Optimized for M1/M2/M3 Macs

```bash
# Recommended installation for Apple Silicon
pip install faiss-cpu  # Native Apple Silicon support
pip install torch      # Apple Silicon optimized
```

### Performance on Apple Silicon
- **M3 MacBook Air**: Excellent performance for most use cases
- **Memory Efficiency**: Benefits from unified memory architecture
- **Local Development**: Perfect for building and testing FAISS indices

## Educational Use Cases

### 1. Full-Stack Web Development
- **Frontend**: JavaScript/TypeScript + HTML/CSS
- **Backend**: Python/Java + SQL
- **Analysis**: Track integration patterns and technology stack decisions

### 2. Systems Programming
- **Core**: Java + Python
- **Analysis**: Evaluate multi-language design patterns and code organization

### 3. Data Science Projects
- **Analysis**: Python + Java
- **Analysis**: Assess data pipeline design and cross-language integration

## Troubleshooting

### Common Issues

1. **"FAISS not available"**
   ```bash
   pip install faiss-cpu  # For Apple Silicon compatibility
   ```

2. **"No language analyzers available"**
   - Check if `src/faiss/analyzers/` directory exists
   - Verify Python and Java analyzers are properly imported

3. **"Unknown language detected"**
   - File extensions are case-sensitive
   - Check supported extensions: `.py`, `.pyw`, `.java`

4. **Memory issues with large multi-language datasets**
   - Use IVF or HNSW index types
   - Process submissions in batches
   - Consider language-specific filtering

### Debugging Multi-Language Analysis

```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

analyzer = MultiLanguageCodeAnalyzer()

# Check supported languages
print(f"Supported extensions: {analyzer.get_supported_extensions()}")

# Test language detection
print(f"main.py -> {analyzer.detect_language('main.py')}")
print(f"Main.java -> {analyzer.detect_language('Main.java')}")

# Debug analysis
code_files = {"test.py": "def hello(): pass"}
results = analyzer.analyze_codebase(code_files)
print(f"Analysis results: {results}")
```

## Future Enhancements

### Planned Multi-Language Features

1. **Additional Languages**
   - JavaScript/TypeScript analyzer with ES6+ support
   - C++ analyzer with template detection
   - C# analyzer with LINQ pattern recognition
   - Go analyzer with goroutine pattern analysis

2. **Advanced Cross-Language Analysis**
   - API boundary detection between languages
   - Cross-language design pattern recognition
   - Architecture quality assessment for polyglot projects

3. **Enhanced Context**
   - Language-specific historical examples
   - Cross-language plagiarism detection
   - Multi-language complexity assessment

## Citation

If you use this system in academic work, please cite:

```
AutoGrader FAISS Historical Context System with Multi-Language Support
Graduate Assignment Evaluation using Multi-layer Code Embeddings
Version 2.1.0, 2025
```

## License

This system is part of the AutoGrader project and follows the same licensing terms. 