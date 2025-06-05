"""
FAISS Historical Context System

This module provides vector database capabilities for historical code context.
It includes enhanced similarity calculation and assignment-specific indexing.

Components:
- Embedder: Multi-layer code embeddings (semantic + structural + dependency)
- Processor: Extract and process submissions from ZIP files
- FAISS Manager: Traditional single-index FAISS management
- Assignment FAISS Manager: Per-assignment FAISS indices for better efficiency
- Similarity Enhancer: Enhanced similarity calculation with component weighting
- Historical Context: Interface for retrieving contextual examples

Author: Auto-generated
"""

from .embedder import HybridCodeEmbedder, ImprovedCodeEmbedder
from .processor import SubmissionProcessor, Submission
from .faiss_manager import FAISSManager
from .assignment_faiss_manager import AssignmentFAISSManager
from .similarity_enhancer import SimilarityEnhancer
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
    # Core embedding and processing
    'HybridCodeEmbedder',
    'ImprovedCodeEmbedder', 
    'SubmissionProcessor',
    'Submission',
    
    # FAISS management
    'FAISSManager',
    'AssignmentFAISSManager',
    
    # Enhanced similarity
    'SimilarityEnhancer',
    
    # Historical context interface
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

__version__ = "2.1.0"

# Configuration constants
DEFAULT_MODEL = "microsoft/codebert-base"
DEFAULT_INDEX_TYPE = "flat"
DEFAULT_SIMILARITY_WEIGHTS = [0.5, 0.3, 0.15, 0.05]  # [semantic, structural, pattern, graph]

# Quick start functions
def create_assignment_manager(base_path: str, enhanced_similarity: bool = True):
    """
    Quick start function to create an assignment FAISS manager.
    
    Args:
        base_path: Base directory for assignment indices
        enhanced_similarity: Enable enhanced similarity calculation
        
    Returns:
        AssignmentFAISSManager instance
    """
    return AssignmentFAISSManager(
        base_index_path=base_path,
        use_enhanced_similarity=enhanced_similarity
    )

def create_embedder(model_name: str = DEFAULT_MODEL, enhanced_similarity: bool = True):
    """
    Quick start function to create a code embedder with enhanced similarity.
    
    Args:
        model_name: Name of the embedding model
        enhanced_similarity: Enable enhanced similarity features
        
    Returns:
        ImprovedCodeEmbedder instance
    """
    embedder = ImprovedCodeEmbedder(model_name=model_name)
    
    if enhanced_similarity:
        # Configure with default weights using positional arguments
        embedder.adjust_similarity_weights(*DEFAULT_SIMILARITY_WEIGHTS)
    
    return embedder 