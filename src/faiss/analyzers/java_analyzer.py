"""
Java Code Analyzer

Provides analysis of Java code using regex-based pattern matching.
Extracts structural metrics and dependency information.

Author: Auto-generated
"""

import re
from typing import Dict, Any, List, Tuple
from .base_analyzer import LanguageAnalyzer


class JavaAnalyzer(LanguageAnalyzer):
    """Java-specific code analyzer using regex patterns"""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.java']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract structural metrics from Java file using regex patterns"""
        metrics = {
            'num_classes': len(re.findall(r'\bclass\s+\w+', content)),
            'num_interfaces': len(re.findall(r'\binterface\s+\w+', content)),
            'num_methods': len(re.findall(r'\b(?:public|private|protected|static|\s)+\w+\s+\w+\s*\([^)]*\)\s*\{', content)),
            'num_imports': len(re.findall(r'^\s*import\s+', content, re.MULTILINE)),
            'num_packages': len(re.findall(r'^\s*package\s+', content, re.MULTILINE)),
            'lines_of_code': len(content.split('\n')),
            'num_try_blocks': len(re.findall(r'\btry\s*\{', content)),
            'num_loops': len(re.findall(r'\b(for|while)\s*\(', content)),
            'num_conditionals': len(re.findall(r'\bif\s*\(', content)),
            'cyclomatic_complexity': self._calculate_java_complexity(content),
            # Normalize to Python-style metrics for aggregation
            'num_functions': len(re.findall(r'\b(?:public|private|protected|static|\s)+\w+\s+\w+\s*\([^)]*\)\s*\{', content)),
            'max_nesting_depth': self._estimate_nesting_depth(content),
            'num_variables': len(re.findall(r'\b(?:int|String|boolean|double|float|char|long)\s+\w+', content)),
            'num_decorators': len(re.findall(r'@\w+', content)),  # Annotations in Java
        }
        return metrics
    
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """Extract Java import dependencies"""
        dependencies = []
        import_matches = re.findall(r'^\s*import\s+([^;]+);', content, re.MULTILINE)
        
        for import_name in import_matches:
            import_name = import_name.strip()
            # Look for matching Java files
            for target_file in all_files:
                if not target_file.endswith('.java'):
                    continue
                    
                target_class = target_file.replace('.java', '')
                if (import_name.endswith(target_class) or 
                    target_class in import_name.split('.')):
                    dependencies.append((filename, target_file))
                    break
        
        return dependencies
    
    def _calculate_java_complexity(self, content: str) -> int:
        """Calculate approximate cyclomatic complexity for Java"""
        complexity = 1  # Base complexity
        
        # Decision points
        complexity += len(re.findall(r'\bif\s*\(', content))
        complexity += len(re.findall(r'\belse\b', content))
        complexity += len(re.findall(r'\bwhile\s*\(', content))
        complexity += len(re.findall(r'\bfor\s*\(', content))
        complexity += len(re.findall(r'\bcase\s+', content))
        complexity += len(re.findall(r'\bcatch\s*\(', content))
        complexity += len(re.findall(r'\b&&\b', content))
        complexity += len(re.findall(r'\b\|\|\b', content))
        
        return complexity
    
    def _estimate_nesting_depth(self, content: str) -> int:
        """Estimate maximum nesting depth by counting braces"""
        max_depth = 0
        current_depth = 0
        
        for char in content:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth 