"""
Python Code Analyzer

Provides comprehensive analysis of Python code using AST parsing.
Extracts structural metrics and dependency information.

Author: Auto-generated
"""

import ast
from typing import Dict, Any, List, Tuple, Optional
from .base_analyzer import LanguageAnalyzer


class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer using AST"""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.py', '.pyw']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract structural metrics from Python file using AST"""
        try:
            tree = ast.parse(content)
            metrics = {
                'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'num_imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'max_nesting_depth': self._get_max_nesting_depth(tree),
                'cyclomatic_complexity': self._calculate_complexity(tree),
                'lines_of_code': len(content.split('\n')),
                'num_variables': len([n for n in ast.walk(tree) if isinstance(n, ast.Name)]),
                'num_decorators': len([n for n in ast.walk(tree) if hasattr(n, 'decorator_list') and n.decorator_list]),
                'num_try_blocks': len([n for n in ast.walk(tree) if isinstance(n, ast.Try)]),
                'num_loops': len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]),
            }
            return metrics
        except SyntaxError:
            return self._get_zero_metrics()
        except Exception:
            return self._get_zero_metrics()
    
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """Extract Python import dependencies"""
        dependencies = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = self._find_matching_file(alias.name, all_files)
                        if target:
                            dependencies.append((filename, target))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        target = self._find_matching_file(node.module, all_files)
                        if target:
                            dependencies.append((filename, target))
        except:
            pass
        return dependencies
    
    def _get_zero_metrics(self) -> Dict[str, Any]:
        """Return zero metrics when parsing fails"""
        return {k: 0 for k in ['num_classes', 'num_functions', 'num_imports', 
                             'max_nesting_depth', 'cyclomatic_complexity', 
                             'lines_of_code', 'num_variables', 'num_decorators',
                             'num_try_blocks', 'num_loops']}
    
    def _get_max_nesting_depth(self, node, depth=0):
        """Calculate maximum nesting depth recursively"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, 
                                ast.FunctionDef, ast.ClassDef, ast.AsyncFor, ast.AsyncWith)):
                child_depth = self._get_max_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _calculate_complexity(self, tree):
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                               ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        return complexity
    
    def _find_matching_file(self, import_name: str, all_files: List[str]) -> Optional[str]:
        """Find file that matches import name"""
        for target_file in all_files:
            if not target_file.endswith('.py'):
                continue
            target_name = target_file.replace('.py', '')
            if (import_name == target_name or 
                import_name.endswith(target_name) or
                target_name in import_name):
                return target_file
        return None 