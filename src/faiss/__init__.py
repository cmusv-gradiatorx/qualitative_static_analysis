"""
FAISS Historical Context System for Java Code

This module provides vector database capabilities for historical Java code context
using StarCoder2 embeddings. It includes simplified code embedding and assignment-specific indexing.

Components:
- OllamaJavaCodeEmbedder: Java code embeddings using Ollama with StarCoder2
- Processor: Extract and process submissions from ZIP files
- FAISS Manager: Traditional single-index FAISS management
- Assignment FAISS Manager: Per-assignment FAISS indices for better efficiency
- Historical Context: Interface for retrieving contextual examples
- RepomixEmbedder: Embedding generator using repomix processing
- RepomixClusterManager: Clustering manager for repomix embeddings

Author: Auto-generated
"""

from .embedder import JavaCodeEmbedder, OllamaJavaCodeEmbedder, EmbedderConfig, create_java_embedder
from .processor import SubmissionProcessor, Submission, RepomixProcessor
from .faiss_manager import FAISSManager
from .repomix_embedder import RepomixEmbedder
from .repomix_cluster_manager import RepomixClusterManager

# FAISS Manager is now always available
FAISS_MANAGER_AVAILABLE = True

try:
    HISTORICAL_CONTEXT_AVAILABLE = True
except ImportError:
    HistoricalContextProvider = None
    HISTORICAL_CONTEXT_AVAILABLE = False

# Import AST analyzer if available
try:
    from .analyzers.ast_java_analyzer import ASTJavaAnalyzer
    AST_ANALYZER_AVAILABLE = True
except ImportError:
    ASTJavaAnalyzer = None
    AST_ANALYZER_AVAILABLE = False

__all__ = [
    # Core embedding and processing
    'JavaCodeEmbedder',  # Backward compatibility alias
    'OllamaJavaCodeEmbedder',  # Main embedder class
    'EmbedderConfig',
    'create_java_embedder',
    'SubmissionProcessor',
    'Submission',
    'RepomixProcessor',
    
    # FAISS management
    'FAISSManager',
    
    # AST analysis
    'ASTJavaAnalyzer',
    'AST_ANALYZER_AVAILABLE',
    
    # Repomix
    'RepomixEmbedder',
    'RepomixClusterManager'
]

# FAISSManager is already in __all__, no need to add it again

if HISTORICAL_CONTEXT_AVAILABLE:
    __all__.append('HistoricalContextProvider')

__version__ = "3.0.0"

# Configuration constants
DEFAULT_MODEL = "bigcode/starCoder2-3b"
DEFAULT_INDEX_TYPE = "flat"
DEFAULT_MAX_LENGTH = 2048

# Quick start functions
def create_assignment_manager(base_path: str, index_type: str = DEFAULT_INDEX_TYPE):
    """
    Quick start function to create an assignment FAISS manager.
    
    Args:
        base_path: Base directory for assignment indices
        index_type: FAISS index type
        
    Returns:
        FAISSManager instance
    """
    return FAISSManager(
        base_index_path=base_path,
        index_type=index_type
    )

def create_embedder(model_name: str = DEFAULT_MODEL, 
                   device: str = None,
                   max_length: int = DEFAULT_MAX_LENGTH):
    """
    Quick start function to create a Java code embedder.
    
    Args:
        model_name: Name of the StarCoder2 model
        device: Device to use ('cpu', 'cuda', or None for auto)
        max_length: Maximum sequence length
        
    Returns:
        OllamaJavaCodeEmbedder instance
    """
    return create_java_embedder(
        model_name=model_name,
        device=device,
        max_length=max_length
    )

# Backward compatibility aliases
HybridCodeEmbedder = OllamaJavaCodeEmbedder
ImprovedCodeEmbedder = OllamaJavaCodeEmbedder 