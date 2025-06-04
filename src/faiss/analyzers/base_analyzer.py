"""
Base Language Analyzer Interface

Abstract base class for language-specific code analyzers.
All language analyzers must implement this interface for consistent analysis.

Author: Auto-generated
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class LanguageAnalyzer(ABC):
    """Abstract base class for language-specific code analyzers"""
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Return list of file extensions this analyzer supports
        
        Returns:
            List of file extensions (e.g., ['.py', '.pyw'])
        """
        pass
    
    @abstractmethod
    def analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Analyze a single file and return structural metrics
        
        Args:
            content: File content as string
            filename: Name of the file being analyzed
            
        Returns:
            Dictionary containing structural metrics for the file
        """
        pass
    
    @abstractmethod
    def extract_dependencies(self, content: str, filename: str, all_files: List[str]) -> List[Tuple[str, str]]:
        """
        Extract dependencies from file content
        
        Args:
            content: File content as string
            filename: Name of the file being analyzed
            all_files: List of all available files in the codebase
            
        Returns:
            List of (from_file, to_file) dependency tuples
        """
        pass 