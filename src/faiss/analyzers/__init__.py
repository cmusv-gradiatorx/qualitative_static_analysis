"""
Language Analyzers for Multi-Language Code Analysis

This module provides language-specific code analyzers for the FAISS historical context system.
Each analyzer implements the LanguageAnalyzer interface for consistent analysis across languages.

Author: Auto-generated
"""

from .base_analyzer import LanguageAnalyzer
from .java_analyzer import JavaAnalyzer
from .ast_java_analyzer import ASTJavaAnalyzer

__all__ = [
    'LanguageAnalyzer',
    'JavaAnalyzer',
    'ASTJavaAnalyzer'
]

__version__ = "2.0.0" 