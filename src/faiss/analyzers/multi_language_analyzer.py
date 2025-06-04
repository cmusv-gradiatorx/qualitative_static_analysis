"""
Multi-Language Code Analyzer and Dependency Graph Builder

This module coordinates multiple language-specific analyzers to provide
comprehensive code analysis across different programming languages.

Author: Auto-generated
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Set, Optional, Tuple
from pathlib import Path

from .base_analyzer import LanguageAnalyzer
from .python_analyzer import PythonAnalyzer
from .java_analyzer import JavaAnalyzer


class MultiLanguageCodeAnalyzer:
    """Multi-language code structure analyzer that coordinates language-specific analyzers"""
    
    def __init__(self):
        self.analyzers = {
            'python': PythonAnalyzer(),
            'java': JavaAnalyzer(),
        }
        
        # Build extension to analyzer mapping
        self.extension_map = {}
        for lang_name, analyzer in self.analyzers.items():
            for ext in analyzer.get_supported_extensions():
                self.extension_map[ext] = analyzer
    
    def add_language_analyzer(self, language_name: str, analyzer: LanguageAnalyzer):
        """Add a new language analyzer"""
        self.analyzers[language_name] = analyzer
        for ext in analyzer.get_supported_extensions():
            self.extension_map[ext] = analyzer
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        return list(self.extension_map.keys())
    
    def detect_language(self, filename: str) -> Optional[str]:
        """Detect programming language from filename"""
        ext = Path(filename).suffix.lower()
        if ext in self.extension_map:
            analyzer = self.extension_map[ext]
            for lang_name, lang_analyzer in self.analyzers.items():
                if analyzer == lang_analyzer:
                    return lang_name
        return None
    
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Analyze a single file using appropriate language analyzer"""
        ext = Path(filename).suffix.lower()
        
        if ext in self.extension_map:
            analyzer = self.extension_map[ext]
            metrics = analyzer.analyze_file(content, filename)
            metrics['language'] = self.detect_language(filename)
            metrics['file_extension'] = ext
            return metrics
        else:
            # Fallback for unsupported languages
            return self._analyze_generic_file(content, filename)
    
    def _analyze_generic_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Generic analysis for unsupported file types"""
        ext = Path(filename).suffix.lower()
        lines = content.split('\n')
        
        return {
            'language': 'unknown',
            'file_extension': ext,
            'lines_of_code': len(lines),
            'num_functions': len(re.findall(r'\bfunction\b|\bdef\b|\bmethod\b|\bpublic\s+\w+\s+\w+\s*\(', content, re.IGNORECASE)),
            'num_classes': len(re.findall(r'\bclass\b|\bstruct\b', content, re.IGNORECASE)),
            'num_imports': len(re.findall(r'\bimport\b|\binclude\b|\brequire\b|\busing\b', content, re.IGNORECASE)),
            'cyclomatic_complexity': 1 + len(re.findall(r'\bif\b|\bwhile\b|\bfor\b', content, re.IGNORECASE)),
            'max_nesting_depth': 0,  # Hard to calculate generically
            'num_variables': 0,
            'num_decorators': 0,
            'num_try_blocks': len(re.findall(r'\btry\b|\bcatch\b', content, re.IGNORECASE)),
            'num_loops': len(re.findall(r'\bfor\b|\bwhile\b', content, re.IGNORECASE)),
        }
    
    def analyze_codebase(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze entire multi-language codebase and return aggregated metrics"""
        all_metrics = []
        language_distribution = {}
        
        for filename, content in code_files.items():
            file_metrics = self.analyze_file(content, filename)
            all_metrics.append(file_metrics)
            
            # Track language distribution
            lang = file_metrics.get('language', 'unknown')
            language_distribution[lang] = language_distribution.get(lang, 0) + 1
        
        if not all_metrics:
            return self._get_zero_aggregated_metrics()
        
        # Aggregate metrics across all files
        aggregated = {}
        
        # Get all metric keys (union of all languages)
        all_keys = set()
        for metrics in all_metrics:
            all_keys.update(metrics.keys())
        
        # Remove non-numeric keys
        numeric_keys = {k for k in all_keys if k not in ['language', 'file_extension']}
        
        for key in numeric_keys:
            values = [metrics.get(key, 0) for metrics in all_metrics if key in metrics]
            if values:
                aggregated[f'{key}_total'] = sum(values)
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f'{key}_max'] = max(values)
        
        # Add derived metrics
        aggregated['num_files'] = len(all_metrics)
        aggregated['language_diversity'] = len(language_distribution)
        aggregated['primary_language'] = max(language_distribution.items(), key=lambda x: x[1])[0] if language_distribution else 'unknown'
        
        # Language-specific ratios
        total_files = len(all_metrics)
        for lang, count in language_distribution.items():
            aggregated[f'{lang}_file_ratio'] = count / total_files
        
        # Safe division for derived metrics
        total_functions = aggregated.get('num_functions_total', 0)
        total_classes = aggregated.get('num_classes_total', 0)
        
        aggregated['complexity_per_function'] = (
            aggregated.get('cyclomatic_complexity_total', 0) / max(total_functions, 1)
        )
        aggregated['functions_per_class'] = (
            total_functions / max(total_classes, 1)
        )
        
        return aggregated
    
    def _get_zero_aggregated_metrics(self) -> Dict[str, Any]:
        """Return zero metrics when no files are analyzed"""
        base_keys = ['num_classes', 'num_functions', 'num_imports', 
                    'max_nesting_depth', 'cyclomatic_complexity', 
                    'lines_of_code', 'num_variables', 'num_decorators',
                    'num_try_blocks', 'num_loops']
        
        result = {}
        for key in base_keys:
            result[f'{key}_total'] = 0.0
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
            result[f'{key}_max'] = 0.0
        
        result.update({
            'num_files': 0,
            'language_diversity': 0,
            'primary_language': 'unknown',
            'complexity_per_function': 0.0,
            'functions_per_class': 0.0
        })
        
        return result


class MultiLanguageDependencyGraphBuilder:
    """Build dependency graphs from multi-language codebases"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.analyzer = MultiLanguageCodeAnalyzer()
    
    def build_from_files(self, code_files: Dict[str, str]) -> nx.DiGraph:
        """Build dependency graph from multi-language code files"""
        self.graph.clear()
        
        # Add nodes for each supported file
        supported_files = {}
        for filename, content in code_files.items():
            ext = Path(filename).suffix.lower()
            if ext in self.analyzer.get_supported_extensions():
                supported_files[filename] = content
                lang = self.analyzer.detect_language(filename)
                self.graph.add_node(filename, file_type=lang, extension=ext)
        
        # Analyze dependencies for each supported language
        for filename, content in supported_files.items():
            ext = Path(filename).suffix.lower()
            if ext in self.analyzer.extension_map:
                analyzer = self.analyzer.extension_map[ext]
                try:
                    dependencies = analyzer.extract_dependencies(content, filename, list(supported_files.keys()))
                    for from_file, to_file in dependencies:
                        if from_file in supported_files and to_file in supported_files:
                            self.graph.add_edge(from_file, to_file, weight=1.0)
                except Exception:
                    continue
        
        return self.graph
    
    def get_graph_features(self) -> np.ndarray:
        """Extract graph-level features including language diversity metrics"""
        if len(self.graph.nodes) == 0:
            return np.zeros(20)  # Feature count for multi-language
        
        # Basic graph features
        features = [
            len(self.graph.nodes),  # Number of files
            len(self.graph.edges),  # Number of dependencies
            nx.density(self.graph),  # Graph density
            len(list(nx.weakly_connected_components(self.graph))),  # Weakly connected components
            len(list(nx.strongly_connected_components(self.graph))),  # Strongly connected components
        ]
        
        # Language diversity features
        languages = set()
        extensions = set()
        for node, data in self.graph.nodes(data=True):
            languages.add(data.get('file_type', 'unknown'))
            extensions.add(data.get('extension', ''))
        
        features.extend([
            len(languages),  # Number of different languages
            len(extensions),  # Number of different file types
        ])
        
        # Centrality measures
        try:
            in_centrality = nx.in_degree_centrality(self.graph)
            out_centrality = nx.out_degree_centrality(self.graph)
            
            features.extend([
                np.mean(list(in_centrality.values())),
                np.std(list(in_centrality.values())),
                max(in_centrality.values()) if in_centrality else 0,
                np.mean(list(out_centrality.values())),
                np.std(list(out_centrality.values())),
                max(out_centrality.values()) if out_centrality else 0,
            ])
        except:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Additional graph metrics
        try:
            clustering_coeff = nx.average_clustering(self.graph.to_undirected()) if len(self.graph.nodes) > 0 else 0
            features.append(clustering_coeff)
        except:
            features.append(0)
        
        try:
            cycle_count = len(list(nx.simple_cycles(self.graph))) if len(self.graph.nodes) < 20 else 0
            features.append(cycle_count)
        except:
            features.append(0)
        
        # Degree distribution features
        try:
            degrees = [d for n, d in self.graph.degree()]
            features.extend([
                np.mean(degrees) if degrees else 0,
                np.std(degrees) if len(degrees) > 1 else 0,
                max(degrees) if degrees else 0
            ])
        except:
            features.extend([0, 0, 0])
        
        # Cross-language dependency features
        cross_lang_edges = 0
        for u, v, data in self.graph.edges(data=True):
            u_lang = self.graph.nodes[u].get('file_type', 'unknown')
            v_lang = self.graph.nodes[v].get('file_type', 'unknown')
            if u_lang != v_lang:
                cross_lang_edges += 1
        
        cross_lang_ratio = cross_lang_edges / max(len(self.graph.edges), 1)
        features.append(cross_lang_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def get_file_level_features(self, filename: str) -> Dict[str, float]:
        """Get dependency features for a specific file"""
        if filename not in self.graph.nodes:
            return {
                'in_degree': 0.0,
                'out_degree': 0.0,
                'betweenness_centrality': 0.0,
                'closeness_centrality': 0.0,
                'language_connectivity': 0.0
            }
        
        try:
            in_degree = self.graph.in_degree(filename)
            out_degree = self.graph.out_degree(filename)
            
            # Calculate centrality measures
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(filename, 0.0)
            closeness_centrality = nx.closeness_centrality(self.graph).get(filename, 0.0)
            
            # Language connectivity: how many different languages this file connects to
            connected_languages = set()
            file_lang = self.graph.nodes[filename].get('file_type', 'unknown')
            
            for neighbor in self.graph.neighbors(filename):
                neighbor_lang = self.graph.nodes[neighbor].get('file_type', 'unknown')
                if neighbor_lang != file_lang:
                    connected_languages.add(neighbor_lang)
            
            language_connectivity = len(connected_languages)
            
            return {
                'in_degree': float(in_degree),
                'out_degree': float(out_degree),
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'language_connectivity': float(language_connectivity)
            }
        except:
            return {
                'in_degree': 0.0,
                'out_degree': 0.0,
                'betweenness_centrality': 0.0,
                'closeness_centrality': 0.0,
                'language_connectivity': 0.0
            }


# Import statement for re module (needed for generic analysis)
import re 