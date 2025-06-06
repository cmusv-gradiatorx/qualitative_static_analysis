"""
AST-Based Java Code Analyzer

Provides advanced Java code analysis using Abstract Syntax Tree parsing.
Extracts structural fingerprints and semantic features for better code similarity.

Author: Auto-generated
"""

import re
import hashlib
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict
from .base_analyzer import LanguageAnalyzer


class ASTJavaAnalyzer(LanguageAnalyzer):
    """Advanced Java analyzer using AST-like pattern extraction"""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.java']
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract comprehensive structural and semantic metrics"""
        
        # Basic metrics
        basic_metrics = self._extract_basic_metrics(content)
        
        # Structural patterns
        structural_patterns = self._extract_structural_patterns(content)
        
        # Method signatures and complexity
        method_analysis = self._analyze_methods(content)
        
        # Class hierarchy and relationships
        class_analysis = self._analyze_classes(content)
        
        # Code fingerprints for similarity
        fingerprints = self._generate_code_fingerprints(content)
        
        return {
            **basic_metrics,
            **structural_patterns,
            **method_analysis,
            **class_analysis,
            **fingerprints
        }
    
    def _extract_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Extract basic code metrics"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len([line for line in lines if line.strip().startswith('//')]),
            'import_count': len(re.findall(r'^\s*import\s+', content, re.MULTILINE)),
            'package_declarations': len(re.findall(r'^\s*package\s+', content, re.MULTILINE)),
        }
    
    def _extract_structural_patterns(self, content: str) -> Dict[str, Any]:
        """Extract structural programming patterns"""
        
        # Control flow patterns
        control_flow = {
            'if_statements': len(re.findall(r'\bif\s*\(', content)),
            'switch_statements': len(re.findall(r'\bswitch\s*\(', content)),
            'for_loops': len(re.findall(r'\bfor\s*\(', content)),
            'while_loops': len(re.findall(r'\bwhile\s*\(', content)),
            'try_catch_blocks': len(re.findall(r'\btry\s*\{', content)),
            'lambda_expressions': len(re.findall(r'->', content))
        }
        
        # OOP patterns
        oop_patterns = {
            'class_declarations': len(re.findall(r'\bclass\s+\w+', content)),
            'interface_declarations': len(re.findall(r'\binterface\s+\w+', content)),
            'abstract_classes': len(re.findall(r'\babstract\s+class', content)),
            'inheritance_patterns': len(re.findall(r'\bextends\s+\w+', content)),
            'implementation_patterns': len(re.findall(r'\bimplements\s+', content)),
            'annotation_usage': len(re.findall(r'@\w+', content))
        }
        
        # Design patterns (basic detection)
        design_patterns = self._detect_design_patterns(content)
        
        return {
            'control_flow': control_flow,
            'oop_patterns': oop_patterns,
            'design_patterns': design_patterns,
            'complexity_indicators': {
                'nested_blocks': self._count_nested_blocks(content),
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(content)
            }
        }
    
    def _analyze_methods(self, content: str) -> Dict[str, Any]:
        """Analyze method signatures and characteristics"""
        
        # Extract method signatures
        method_pattern = r'(public|private|protected|static|\s)*\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^{]+)?\s*\{'
        methods = re.findall(method_pattern, content)
        
        method_metrics = {
            'total_methods': len(methods),
            'public_methods': 0,
            'private_methods': 0,
            'static_methods': 0,
            'parameter_counts': [],
            'return_types': defaultdict(int),
            'method_names': []
        }
        
        for modifiers, return_type, method_name, params in methods:
            # Count access modifiers
            if 'public' in modifiers:
                method_metrics['public_methods'] += 1
            elif 'private' in modifiers:
                method_metrics['private_methods'] += 1
            
            if 'static' in modifiers:
                method_metrics['static_methods'] += 1
            
            # Parameter analysis
            param_count = len([p.strip() for p in params.split(',') if p.strip()])
            method_metrics['parameter_counts'].append(param_count)
            
            # Return type analysis
            method_metrics['return_types'][return_type] += 1
            
            # Store method name (for signature analysis)
            method_metrics['method_names'].append(method_name)
        
        # Calculate method statistics
        if method_metrics['parameter_counts']:
            method_metrics['avg_parameters'] = sum(method_metrics['parameter_counts']) / len(method_metrics['parameter_counts'])
            method_metrics['max_parameters'] = max(method_metrics['parameter_counts'])
        else:
            method_metrics['avg_parameters'] = 0
            method_metrics['max_parameters'] = 0
        
        return {'method_analysis': method_metrics}
    
    def _analyze_classes(self, content: str) -> Dict[str, Any]:
        """Analyze class structure and relationships"""
        
        # Extract class information
        class_pattern = r'(public|private|protected|abstract|final|\s)*\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{'
        classes = re.findall(class_pattern, content)
        
        class_metrics = {
            'total_classes': len(classes),
            'abstract_classes': 0,
            'final_classes': 0,
            'inheritance_depth': 0,
            'implemented_interfaces': [],
            'class_names': []
        }
        
        for modifiers, class_name, parent_class, interfaces in classes:
            if 'abstract' in modifiers:
                class_metrics['abstract_classes'] += 1
            if 'final' in modifiers:
                class_metrics['final_classes'] += 1
            
            class_metrics['class_names'].append(class_name)
            
            if parent_class:
                class_metrics['inheritance_depth'] = max(class_metrics['inheritance_depth'], 1)
            
            if interfaces:
                interface_list = [i.strip() for i in interfaces.split(',')]
                class_metrics['implemented_interfaces'].extend(interface_list)
        
        return {'class_analysis': class_metrics}
    
    def _detect_design_patterns(self, content: str) -> Dict[str, int]:
        """Detect common design patterns"""
        patterns = {
            'singleton_pattern': len(re.findall(r'private\s+static\s+.*\s+instance', content)),
            'factory_pattern': len(re.findall(r'create\w*\(', content)),
            'builder_pattern': len(re.findall(r'\.build\(\)', content)),
            'observer_pattern': len(re.findall(r'(addListener|addEventListener|notify)', content)),
            'strategy_pattern': len(re.findall(r'Strategy\b', content)),
            'decorator_pattern': len(re.findall(r'Decorator\b', content)),
            'adapter_pattern': len(re.findall(r'Adapter\b', content))
        }
        return patterns
    
    def _count_nested_blocks(self, content: str) -> int:
        """Count maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for char in content:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        # Decision points
        decision_keywords = [
            r'\bif\s*\(',
            r'\belse\b',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\bcase\s+',
            r'\bcatch\s*\(',
            r'\b&&\b',
            r'\|\|',
            r'\?\s*[^:]+:',  # Ternary operator
        ]
        
        for pattern in decision_keywords:
            complexity += len(re.findall(pattern, content))
        
        return complexity
    
    def _generate_code_fingerprints(self, content: str) -> Dict[str, Any]:
        """Generate unique fingerprints for code similarity detection"""
        
        # Structural fingerprint (method signatures, class names)
        structural_elements = []
        structural_elements.extend(re.findall(r'class\s+(\w+)', content))
        structural_elements.extend(re.findall(r'interface\s+(\w+)', content))
        structural_elements.extend(re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content))  # Method names
        
        structural_fingerprint = hashlib.md5(''.join(sorted(structural_elements)).encode()).hexdigest()[:16]
        
        # Control flow fingerprint
        control_flow_elements = []
        control_flow_patterns = [r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b', r'\btry\b', r'\bcatch\b']
        for pattern in control_flow_patterns:
            count = len(re.findall(pattern, content))
            if count > 0:
                control_flow_elements.append(f"{pattern}:{count}")
        
        control_flow_fingerprint = hashlib.md5(''.join(control_flow_elements).encode()).hexdigest()[:16]
        
        # Algorithm fingerprint (unique method combinations)
        method_calls = re.findall(r'\.(\w+)\(', content)
        algorithm_signature = hashlib.md5(''.join(sorted(set(method_calls))).encode()).hexdigest()[:16]
        
        return {
            'fingerprints': {
                'structural': structural_fingerprint,
                'control_flow': control_flow_fingerprint,
                'algorithm_signature': algorithm_signature,
                'combined_hash': hashlib.md5(f"{structural_fingerprint}{control_flow_fingerprint}{algorithm_signature}".encode()).hexdigest()[:16]
            },
            'unique_elements': {
                'class_names': list(set(re.findall(r'class\s+(\w+)', content))),
                'method_names': list(set(re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content))),
                'api_calls': list(set(method_calls))
            }
        }
    
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """Extract dependencies using both imports and internal references"""
        dependencies = []
        
        # Standard import dependencies
        import_matches = re.findall(r'^\s*import\s+([^;]+);', content, re.MULTILINE)
        
        for import_name in import_matches:
            import_name = import_name.strip()
            for target_file in all_files:
                if not target_file.endswith('.java'):
                    continue
                
                target_class = target_file.replace('.java', '')
                if (import_name.endswith(target_class) or 
                    target_class in import_name.split('.')):
                    dependencies.append((filename, target_file))
                    break
        
        # Internal class references (without imports)
        class_names = set(re.findall(r'class\s+(\w+)', content))
        for target_file in all_files:
            if target_file == filename or not target_file.endswith('.java'):
                continue
            
            target_class = target_file.replace('.java', '').split('/')[-1]
            if target_class in content and target_class not in [imp.split('.')[-1] for imp in import_matches]:
                dependencies.append((filename, target_file))
        
        return dependencies
    
    def calculate_code_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two code files using structural features"""
        
        # Analyze both files
        analysis1 = self.analyze_file(content1, "file1")
        analysis2 = self.analyze_file(content2, "file2")
        
        # Compare fingerprints
        fp1 = analysis1['fingerprints']
        fp2 = analysis2['fingerprints']
        
        # Structural similarity
        structural_sim = 1.0 if fp1['structural'] == fp2['structural'] else 0.0
        
        # Control flow similarity
        control_sim = 1.0 if fp1['control_flow'] == fp2['control_flow'] else 0.0
        
        # Algorithm similarity
        algo_sim = 1.0 if fp1['algorithm_signature'] == fp2['algorithm_signature'] else 0.0
        
        # Method name overlap
        methods1 = set(analysis1['unique_elements']['method_names'])
        methods2 = set(analysis2['unique_elements']['method_names'])
        method_sim = len(methods1 & methods2) / max(len(methods1 | methods2), 1)
        
        # Weighted similarity score
        similarity = (
            structural_sim * 0.3 +
            control_sim * 0.25 +
            algo_sim * 0.25 +
            method_sim * 0.2
        )
        
        return similarity 