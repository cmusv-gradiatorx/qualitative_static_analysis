# Multi-Language FAISS Historical Context System

## 🌐 Overview

The enhanced FAISS system now supports **multiple programming languages** in a single codebase analysis. This is perfect for software engineering courses where assignments might include Java, Python, JavaScript, C++, and other languages together.

## 🎯 Supported Languages

### **Built-in Support:**
- **Python** (.py, .pyw) - Full AST analysis
- **Java** (.java) - Regex-based analysis  

### **Easily Extensible:**
- **JavaScript/TypeScript** (.js, .jsx, .ts, .tsx) - Template available
- **C++** (.cpp, .cxx, .cc, .c, .h, .hpp, .hxx) - Template available
- **C#** (.cs) - Template available
- **Go** (.go) - Template available
- **And any language you implement!**

## 🏗️ Architecture

### **Pluggable Language Analyzers**
All language analyzers are located in `src/faiss/analyzers/` and implement the common interface:

```python
# src/faiss/analyzers/base_analyzer.py
class LanguageAnalyzer(ABC):
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return file extensions this analyzer supports"""
        
    @abstractmethod  
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract structural metrics from file"""
        
    @abstractmethod
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """Extract import/dependency relationships"""
```

### **Analyzer Directory Structure**
```
src/faiss/analyzers/
├── __init__.py                 # Exports all analyzers
├── base_analyzer.py            # Abstract LanguageAnalyzer interface
├── python_analyzer.py          # Python AST-based analyzer
├── java_analyzer.py            # Java regex-based analyzer
└── multi_language_analyzer.py  # Multi-language coordinator
```

### **Multi-Language Features**
- **Language Detection** - Automatic detection from file extensions
- **Unified Metrics** - Normalized metrics across all languages
- **Cross-Language Dependencies** - Track dependencies between different languages
- **Language Diversity Analysis** - Metrics about language usage in projects

## 🚀 Quick Start

### **1. Basic Multi-Language Analysis**
```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

# Create analyzer (automatically loads Python and Java analyzers)
analyzer = MultiLanguageCodeAnalyzer()

# Analyze mixed-language codebase
code_files = {
    'Main.java': 'public class Main { public static void main(String[] args) { System.out.println("Hello"); } }',
    'utils.py': 'def helper_function(): return "Python helper"',
    'processor.py': 'import json\ndef process(): pass'
}

# Get comprehensive analysis
results = analyzer.analyze_codebase(code_files)

print(f"Languages detected: {results['language_diversity']}")
print(f"Primary language: {results['primary_language']}")
print(f"Total functions: {results['num_functions_total']}")
print(f"Python files: {results.get('python_file_ratio', 0):.1%}")
print(f"Java files: {results.get('java_file_ratio', 0):.1%}")
```

### **2. Historical Context with Multi-Language**
```python
from src.faiss import HybridCodeEmbedder, FAISSManager, HistoricalContextProvider

# Initialize with multi-language support (automatically detects analyzers)
embedder = HybridCodeEmbedder()
faiss_manager = FAISSManager()
faiss_manager.load_index("src/faiss/index")

context_provider = HistoricalContextProvider(faiss_manager, embedder)

# Mixed-language submission
submission_files = {
    'DatabaseManager.java': '/* Java database code */',
    'data_processor.py': '# Python data processing',
    'config.py': '# Python configuration'
}

# Get historical context
historical_context = context_provider.get_context_for_prompt_manager(
    code_files=submission_files,
    assignment_id="FullStackProject",
    max_examples=3
)
```

### **3. Check Available Languages**
```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

analyzer = MultiLanguageCodeAnalyzer()
print(f"Supported extensions: {analyzer.get_supported_extensions()}")
print(f"Available analyzers: {list(analyzer.analyzers.keys())}")

# Test language detection
print(f"main.py -> {analyzer.detect_language('main.py')}")
print(f"Main.java -> {analyzer.detect_language('Main.java')}")
print(f"unknown.txt -> {analyzer.detect_language('unknown.txt')}")
```

## 🔧 Adding New Languages

### **Step 1: Create Language Analyzer**
Create a new file in `src/faiss/analyzers/`:

```python
# src/faiss/analyzers/rust_analyzer.py
import re
from typing import Dict, Any, List, Tuple
from .base_analyzer import LanguageAnalyzer

class RustAnalyzer(LanguageAnalyzer):
    """Rust code analyzer using regex patterns"""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.rs']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract structural metrics from Rust file"""
        return {
            'num_functions': len(re.findall(r'\bfn\s+\w+', content)),
            'num_structs': len(re.findall(r'\bstruct\s+\w+', content)),
            'num_traits': len(re.findall(r'\btrait\s+\w+', content)),
            'num_impls': len(re.findall(r'\bimpl\s+', content)),
            'num_mods': len(re.findall(r'\bmod\s+\w+', content)),
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
            'cyclomatic_complexity': self._calculate_rust_complexity(content),
            # Normalize to common metrics
            'num_classes': len(re.findall(r'\bstruct\s+\w+', content)),  # Structs as classes
            'num_imports': len(re.findall(r'\buse\s+', content)),
            'num_methods': len(re.findall(r'\bfn\s+\w+', content)),
        }
    
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """Extract use statements and dependencies"""
        dependencies = []
        use_statements = re.findall(r'\buse\s+([\w:]+)', content)
        
        for use_stmt in use_statements:
            module_name = use_stmt.split('::')[0]
            for target_file in all_files:
                if target_file.endswith('.rs') and module_name in target_file:
                    dependencies.append((filename, target_file))
        
        return dependencies
    
    def _calculate_rust_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity for Rust"""
        complexity = 1  # Base complexity
        
        # Control flow keywords that increase complexity
        complexity_keywords = [
            r'\bif\b', r'\belse\b', r'\bmatch\b', r'\bwhile\b', 
            r'\bfor\b', r'\bloop\b', r'\b\?\b'  # ? operator
        ]
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(keyword, content))
        
        return complexity
```

### **Step 2: Register the Analyzer**
Add the new analyzer to the multi-language analyzer:

```python
# Option 1: Modify __init__.py to include the new analyzer
# src/faiss/analyzers/__init__.py
from .rust_analyzer import RustAnalyzer

# Option 2: Add dynamically at runtime
from src.faiss.analyzers import MultiLanguageCodeAnalyzer
from src.faiss.analyzers.rust_analyzer import RustAnalyzer

analyzer = MultiLanguageCodeAnalyzer()
analyzer.add_language_analyzer('rust', RustAnalyzer())

print(f"Now supports: {analyzer.get_supported_extensions()}")
```

### **Step 3: Update Module Exports (Optional)**
```python
# src/faiss/analyzers/__init__.py
from .base_analyzer import LanguageAnalyzer
from .python_analyzer import PythonAnalyzer
from .java_analyzer import JavaAnalyzer
from .rust_analyzer import RustAnalyzer  # Add new analyzer
from .multi_language_analyzer import MultiLanguageCodeAnalyzer, MultiLanguageDependencyGraphBuilder

__all__ = [
    'LanguageAnalyzer',
    'PythonAnalyzer', 
    'JavaAnalyzer',
    'RustAnalyzer',  # Export new analyzer
    'MultiLanguageCodeAnalyzer',
    'MultiLanguageDependencyGraphBuilder'
]
```

## 📊 Multi-Language Metrics

### **Language Distribution Analysis**
```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

analyzer = MultiLanguageCodeAnalyzer()
results = analyzer.analyze_codebase(code_files)

print(f"Language diversity score: {results['language_diversity']}")
print(f"Primary language: {results['primary_language']}")

# Language-specific ratios
for lang in ['python', 'java']:
    ratio_key = f'{lang}_file_ratio'
    if ratio_key in results:
        print(f"{lang.capitalize()} files: {results[ratio_key]:.2%}")
```

### **Cross-Language Dependencies**
```python
from src.faiss.analyzers import MultiLanguageDependencyGraphBuilder

graph_builder = MultiLanguageDependencyGraphBuilder()
graph = graph_builder.build_from_files(code_files)

# Get graph features including cross-language metrics
features = graph_builder.get_graph_features()
cross_lang_ratio = features[-1]  # Last feature is cross-language dependency ratio

print(f"Cross-language dependency ratio: {cross_lang_ratio:.2%}")

# Analyze specific files
for filename in code_files.keys():
    file_features = graph_builder.get_file_level_features(filename)
    print(f"{filename}: connects to {file_features['language_connectivity']} languages")
```

### **Detailed Language Analysis**
```python
def analyze_project_languages(code_files):
    """Comprehensive language analysis"""
    analyzer = MultiLanguageCodeAnalyzer()
    results = analyzer.analyze_codebase(code_files)
    
    print("=== PROJECT LANGUAGE ANALYSIS ===")
    print(f"Total files: {len(code_files)}")
    print(f"Primary language: {results['primary_language']}")
    print(f"Language diversity: {results['language_diversity']}")
    print(f"Total functions: {results.get('num_functions_total', 0)}")
    print(f"Total classes: {results.get('num_classes_total', 0)}")
    
    # Per-language breakdown
    print("\n=== LANGUAGE BREAKDOWN ===")
    for lang in ['python', 'java']:
        if f'{lang}_file_ratio' in results:
            ratio = results[f'{lang}_file_ratio']
            print(f"{lang.capitalize()}: {ratio:.1%} of files")
    
    return results

# Usage
analysis = analyze_project_languages(code_files)
```

## 🎓 Educational Use Cases

### **1. Full-Stack Web Development**
- **Frontend**: JavaScript/TypeScript + HTML/CSS
- **Backend**: Python/Java + SQL
- **Analysis**: Track how students integrate different technology stacks

```python
# Example full-stack analysis
fullstack_files = {
    'app.py': '# Flask backend',
    'DatabaseManager.java': '// Spring backend service',
    'frontend.js': '// React frontend',
    'api.py': '# API endpoints'
}

results = analyzer.analyze_codebase(fullstack_files)
print(f"Full-stack complexity: {results['cyclomatic_complexity_total']}")
```

### **2. Systems Programming**
- **Core**: Java + Python
- **Analysis**: Evaluate system-level design patterns

```python
# Example systems programming analysis
systems_files = {
    'SystemManager.java': '// Core system management',
    'utils.py': '# Python utilities',
    'monitor.py': '# System monitoring'
}

results = analyzer.analyze_codebase(systems_files)
```

### **3. Data Science Projects**
- **Analysis**: Python + Java
- **Visualization**: JavaScript + Python
- **Analysis**: Assess data pipeline design

```python
# Example data science analysis
datascience_files = {
    'data_processor.py': '# Python data processing',
    'DataAnalyzer.java': '// Java data analysis engine',
    'visualizations.py': '# Python plotting'
}

results = analyzer.analyze_codebase(datascience_files)
```

## ⚙️ Configuration

### **Environment Variables**
```env
# config.env
ENABLE_HISTORICAL_CONTEXT=true
MULTI_LANGUAGE_ANALYSIS=true
FAISS_INDEX_PATH=src/faiss/index
CODE_EMBEDDING_MODEL=microsoft/codebert-base
LANGUAGE_DIVERSITY_WEIGHT=0.2
```

### **Assignment-Specific Language Configuration**
```json
// config/projects/fullstack_project.json
{
  "supported_languages": ["python", "java"],
  "primary_language": "python",
  "language_weights": {
    "python": 0.6,
    "java": 0.4
  },
  "cross_language_analysis": true,
  "require_multi_language": true
}
```

### **Dynamic Language Configuration**
```python
from src.faiss.analyzers import MultiLanguageCodeAnalyzer

class CustomMultiLanguageAnalyzer(MultiLanguageCodeAnalyzer):
    def __init__(self, language_weights=None):
        super().__init__()
        self.language_weights = language_weights or {
            'python': 1.0,
            'java': 1.0,
            'unknown': 0.5
        }
    
    def analyze_codebase(self, code_files):
        results = super().analyze_codebase(code_files)
        
        # Apply custom language weights
        weighted_complexity = 0
        total_weight = 0
        
        for filename, content in code_files.items():
            lang = self.detect_language(filename) or 'unknown'
            weight = self.language_weights.get(lang, 0.5)
            file_metrics = self.analyze_file(content, filename)
            
            weighted_complexity += file_metrics['cyclomatic_complexity'] * weight
            total_weight += weight
        
        results['weighted_complexity'] = weighted_complexity / max(total_weight, 1)
        return results
```

## 🔍 Analysis Features

### **1. Language Quality Assessment**
```python
def assess_language_quality(results):
    """Assess quality per language"""
    print("=== LANGUAGE QUALITY ASSESSMENT ===")
    
    for lang in ['python', 'java']:
        ratio_key = f'{lang}_file_ratio'
        if ratio_key in results:
            ratio = results[ratio_key]
            complexity = results.get('cyclomatic_complexity_mean', 0)
            functions = results.get('num_functions_total', 0)
            
            print(f"{lang.capitalize()}:")
            print(f"  File ratio: {ratio:.1%}")
            print(f"  Avg complexity: {complexity:.1f}")
            print(f"  Total functions: {functions}")
```

### **2. Architecture Analysis**
```python
def analyze_multi_language_architecture(code_files):
    """Analyze multi-language architecture patterns"""
    from src.faiss.analyzers import MultiLanguageDependencyGraphBuilder
    
    graph_builder = MultiLanguageDependencyGraphBuilder()
    graph = graph_builder.build_from_files(code_files)
    
    # Find hub files that connect multiple languages
    hubs = []
    for filename in code_files.keys():
        features = graph_builder.get_file_level_features(filename)
        if features['language_connectivity'] > 1:
            hubs.append((filename, features['language_connectivity']))
    
    print("=== ARCHITECTURE ANALYSIS ===")
    print("Multi-language hub files:")
    for filename, connectivity in sorted(hubs, key=lambda x: x[1], reverse=True):
        lang = MultiLanguageCodeAnalyzer().detect_language(filename)
        print(f"  {filename} ({lang}): connects {connectivity} languages")
    
    return hubs
```

### **3. Language-Specific Historical Context**
```python
def get_language_filtered_context(context_provider, code_files, assignment_id):
    """Get historical context filtered by primary language"""
    from src.faiss.analyzers import MultiLanguageCodeAnalyzer
    
    # Detect primary language of current submission
    analyzer = MultiLanguageCodeAnalyzer()
    results = analyzer.analyze_codebase(code_files)
    primary_lang = results['primary_language']
    
    print(f"Filtering historical context for primary language: {primary_lang}")
    
    # Get similar submissions
    similar_submissions = context_provider.get_similar_submissions_context(
        code_files, assignment_id, top_k=10
    )
    
    # Filter for same primary language
    language_filtered = []
    for submission in similar_submissions.get('similar_submissions', []):
        hist_results = analyzer.analyze_codebase(submission['submission'].code_files)
        if hist_results['primary_language'] == primary_lang:
            language_filtered.append(submission)
    
    return language_filtered[:3]  # Top 3 same-language examples
```

## 🚧 Advanced Features

### **1. Custom Language Weighting**
```python
class WeightedMultiLanguageAnalyzer(MultiLanguageCodeAnalyzer):
    def __init__(self, language_weights=None):
        super().__init__()
        self.language_weights = language_weights or {
            'python': 1.0,
            'java': 1.0,
            'unknown': 0.5
        }
    
    def analyze_codebase(self, code_files):
        results = super().analyze_codebase(code_files)
        
        # Apply language weights to complexity calculations
        weighted_metrics = self._calculate_weighted_metrics(code_files)
        results.update(weighted_metrics)
        
        return results
    
    def _calculate_weighted_metrics(self, code_files):
        """Calculate metrics with language-specific weights"""
        total_weighted_complexity = 0
        total_weight = 0
        
        for filename, content in code_files.items():
            lang = self.detect_language(filename) or 'unknown'
            weight = self.language_weights.get(lang, 0.5)
            file_metrics = self.analyze_file(content, filename)
            
            total_weighted_complexity += file_metrics['cyclomatic_complexity'] * weight
            total_weight += weight
        
        return {
            'weighted_complexity': total_weighted_complexity / max(total_weight, 1),
            'total_language_weight': total_weight
        }
```

### **2. Language Evolution Tracking**
```python
def track_language_evolution(historical_submissions):
    """Track how language usage evolves over time"""
    from src.faiss.analyzers import MultiLanguageCodeAnalyzer
    
    analyzer = MultiLanguageCodeAnalyzer()
    evolution_data = []
    
    for submission in sorted(historical_submissions, key=lambda x: x.timestamp):
        results = analyzer.analyze_codebase(submission.code_files)
        evolution_data.append({
            'timestamp': submission.timestamp,
            'primary_language': results['primary_language'],
            'language_diversity': results['language_diversity'],
            'python_ratio': results.get('python_file_ratio', 0),
            'java_ratio': results.get('java_file_ratio', 0)
        })
    
    return evolution_data
```

## 🔬 Research Opportunities

### **1. Language Learning Progression**
- Track how students progress from single-language to multi-language projects
- Analyze language integration patterns over time
- Study correlation between language diversity and project complexity

### **2. Code Quality Across Languages**
- Compare complexity patterns across different languages
- Analyze how language choice affects code quality metrics
- Study language-specific best practices adoption

### **3. Cross-Language Design Patterns**
- Identify common integration patterns between Python and Java
- Analyze dependency structures in multi-language projects
- Study architectural decisions in polyglot programming

## 🤝 Contributing

### **Adding New Languages**
1. Create analyzer class in `src/faiss/analyzers/[language]_analyzer.py`
2. Implement the `LanguageAnalyzer` interface
3. Add comprehensive test cases
4. Update `__init__.py` exports
5. Add documentation and examples
6. Submit PR with example usage

### **Improving Analysis**
1. Enhance dependency detection algorithms
2. Add language-specific complexity calculations
3. Improve cross-language pattern recognition
4. Add AST-based analysis for more languages

### **Directory Structure for New Languages**
```
src/faiss/analyzers/
├── __init__.py
├── base_analyzer.py
├── python_analyzer.py          # ✅ Implemented
├── java_analyzer.py            # ✅ Implemented
├── javascript_analyzer.py      # 🔄 Add next
├── cpp_analyzer.py             # 🔄 Add next
├── csharp_analyzer.py          # 🔄 Add next
└── multi_language_analyzer.py  # ✅ Coordinator
```

## 📚 Language Template Examples

### **JavaScript/TypeScript Analyzer Template**
```python
# src/faiss/analyzers/javascript_analyzer.py
import re
from typing import Dict, Any, List, Tuple
from .base_analyzer import LanguageAnalyzer

class JavaScriptAnalyzer(LanguageAnalyzer):
    def get_supported_extensions(self) -> List[str]:
        return ['.js', '.jsx', '.ts', '.tsx', '.mjs']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        return {
            'num_functions': len(re.findall(r'\bfunction\s+\w+|const\s+\w+\s*=\s*\(|\w+\s*=>\s*', content)),
            'num_classes': len(re.findall(r'\bclass\s+\w+', content)),
            'num_imports': len(re.findall(r'\bimport\s+.*from|require\s*\(', content)),
            'num_exports': len(re.findall(r'\bexport\s+', content)),
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
            'cyclomatic_complexity': self._calculate_js_complexity(content),
            'num_methods': len(re.findall(r'\w+\s*\([^)]*\)\s*{', content)),
        }
```

---

**Ready to handle any programming language!** 🚀

Your FAISS historical context system now provides intelligent analysis across multiple programming languages with a clean, extensible architecture. The `src/faiss/analyzers/` directory makes it easy to add new languages while maintaining consistency across the system. 