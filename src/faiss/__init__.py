"""
FAISS-based Historical Context System

This module provides historical context for LLM evaluations by finding similar
past submissions using multi-layer code embeddings and FAISS similarity search.

Supports multiple programming languages: Python, Java, and easily extensible to more.

Author: Auto-generated
"""

from .embedder import HybridCodeEmbedder
from .processor import SubmissionProcessor
from .faiss_manager import FAISSManager
from .historical_context import HistoricalContextProvider

# Multi-language analyzers from the analyzers directory
try:
    from .analyzers import (
        LanguageAnalyzer,
        PythonAnalyzer,
        JavaAnalyzer,
        MultiLanguageCodeAnalyzer,
        MultiLanguageDependencyGraphBuilder
    )
    MULTI_LANGUAGE_AVAILABLE = True
except ImportError:
    MULTI_LANGUAGE_AVAILABLE = False

# Backward compatibility - import original analyzer if still exists
try:
    from .analyzer import CodeStructureAnalyzer, DependencyGraphBuilder
    LEGACY_ANALYZER_AVAILABLE = True
except ImportError:
    LEGACY_ANALYZER_AVAILABLE = False

__all__ = [
    'HybridCodeEmbedder',
    'SubmissionProcessor', 
    'FAISSManager',
    'HistoricalContextProvider',
]

# Add multi-language exports if available
if MULTI_LANGUAGE_AVAILABLE:
    __all__.extend([
        'LanguageAnalyzer',
        'PythonAnalyzer',
        'JavaAnalyzer',
        'MultiLanguageCodeAnalyzer',
        'MultiLanguageDependencyGraphBuilder'
    ])

# Add legacy exports for backward compatibility
if LEGACY_ANALYZER_AVAILABLE:
    __all__.extend([
        'CodeStructureAnalyzer',
        'DependencyGraphBuilder'
    ])

__version__ = "2.1.0"  # Updated to reflect refactored analyzer structure 