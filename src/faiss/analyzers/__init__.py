"""
Language Analyzers for Multi-Language Code Analysis

This module provides language-specific code analyzers for the FAISS historical context system.
Each analyzer implements the LanguageAnalyzer interface for consistent analysis across languages.

Author: Auto-generated
"""

from .base_analyzer import LanguageAnalyzer
from .python_analyzer import PythonAnalyzer
from .java_analyzer import JavaAnalyzer
from .multi_language_analyzer import MultiLanguageCodeAnalyzer, MultiLanguageDependencyGraphBuilder

__all__ = [
    'LanguageAnalyzer',
    'PythonAnalyzer', 
    'JavaAnalyzer',
    'MultiLanguageCodeAnalyzer',
    'MultiLanguageDependencyGraphBuilder'
]

__version__ = "2.0.0" 