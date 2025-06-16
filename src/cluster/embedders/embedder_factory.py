"""
Embedder Factory

Factory class for creating different types of embedders with proper configuration.
Provides a unified interface for embedder creation.

Author: Auto-generated
"""

from typing import Dict, Any, Optional, Type
from enum import Enum

from .base_embedder import BaseEmbedder, EmbedderConfig
from .java_embedder import JavaEmbedder
from .repomix_embedder import RepomixEmbedder, RepomixConfig
from .issue_embedder import IssueEmbedder, IssueConfig


class EmbedderType(Enum):
    """Enumeration of available embedder types"""
    JAVA = "java"
    REPOMIX = "repomix"
    ISSUES = "issues"


class EmbedderFactory:
    """
    Factory for creating embedder instances.
    
    Provides a unified interface for creating different types of embedders
    with appropriate configurations.
    """
    
    # Registry of embedder classes
    _EMBEDDER_REGISTRY: Dict[EmbedderType, Type[BaseEmbedder]] = {
        EmbedderType.JAVA: JavaEmbedder,
        EmbedderType.REPOMIX: RepomixEmbedder,
        EmbedderType.ISSUES: IssueEmbedder,
    }
    
    # Default configurations for each embedder type
    _DEFAULT_CONFIGS: Dict[EmbedderType, Type[EmbedderConfig]] = {
        EmbedderType.JAVA: EmbedderConfig,
        EmbedderType.REPOMIX: RepomixConfig,
        EmbedderType.ISSUES: IssueConfig,
    }
    
    @classmethod
    def create_embedder(cls, 
                       embedder_type: str,
                       model_name: str = "starCoder2:3b",
                       task_name: Optional[str] = None,
                       config: Optional[EmbedderConfig] = None,
                       **kwargs) -> BaseEmbedder:
        """
        Create an embedder instance.
        
        Args:
            embedder_type: Type of embedder ('java' or 'repomix')
            model_name: Model name for the embedder
            task_name: Task name for cache organization (e.g., "task4_GildedRoseKata")
            config: Optional pre-configured EmbedderConfig instance
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured embedder instance
            
        Raises:
            ValueError: If embedder_type is not supported
            TypeError: If configuration is invalid
        """
        # Convert string to enum
        try:
            embedder_enum = EmbedderType(embedder_type.lower())
        except ValueError:
            available_types = [e.value for e in EmbedderType]
            raise ValueError(f"Unsupported embedder type: {embedder_type}. Available: {available_types}")
        
        # Get embedder class
        embedder_class = cls._EMBEDDER_REGISTRY[embedder_enum]
        
        # Create configuration if not provided
        if config is None:
            config_class = cls._DEFAULT_CONFIGS[embedder_enum]
            
            # Merge kwargs with defaults
            config_kwargs = {
                'model_name': model_name,
                'task_name': task_name,
                **kwargs
            }
            
            config = config_class(**config_kwargs)
        else:
            # Update config with provided model_name, task_name and kwargs
            if hasattr(config, 'model_name'):
                config.model_name = model_name
            if hasattr(config, 'task_name') and task_name:
                config.task_name = task_name
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create and return embedder instance
        return embedder_class(config)
    
    @classmethod
    def create_java_embedder(cls,
                           model_name: str = "starCoder2:7b",
                           task_name: Optional[str] = None,
                           ollama_base_url: Optional[str] = None,
                           max_length: int = 8192,
                           normalize_embeddings: bool = True,
                           cache_embeddings: bool = True,
                           **kwargs) -> JavaEmbedder:
        """
        Create a Java embedder with specific configuration.
        
        Args:
            model_name: StarCoder2 model name
            task_name: Task name for cache organization
            ollama_base_url: Ollama server URL
            max_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings
            cache_embeddings: Whether to cache embeddings
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured Java embedder
        """
        config_kwargs = {
            'model_name': model_name,
            'task_name': task_name,
            'max_length': max_length,
            'normalize_embeddings': normalize_embeddings,
            'cache_embeddings': cache_embeddings,
            **kwargs
        }
        
        if ollama_base_url:
            config_kwargs['ollama_base_url'] = ollama_base_url
        
        config = EmbedderConfig(**config_kwargs)
        return JavaEmbedder(config)
    
    @classmethod
    def create_repomix_embedder(cls,
                              model_name: str = "starCoder2:3b",
                              task_name: Optional[str] = None,
                              ollama_base_url: Optional[str] = None,
                              max_tokens: int = 128000,
                              use_compression: bool = True,
                              remove_comments: bool = False,
                              ignore_patterns: Optional[list] = None,
                              keep_patterns: Optional[list] = None,
                              **kwargs) -> RepomixEmbedder:
        """
        Create a Repomix embedder with specific configuration.
        
        Args:
            model_name: StarCoder2 model name
            task_name: Task name for cache organization
            ollama_base_url: Ollama server URL
            max_tokens: Maximum token limit for repomix
            use_compression: Whether to use compression
            remove_comments: Whether to remove comments
            ignore_patterns: File patterns to ignore
            keep_patterns: File patterns to keep
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured Repomix embedder
        """
        config_kwargs = {
            'model_name': model_name,
            'task_name': task_name,
            'max_tokens': max_tokens,
            'use_compression': use_compression,
            'remove_comments': remove_comments,
            **kwargs
        }
        
        if ollama_base_url:
            config_kwargs['ollama_base_url'] = ollama_base_url
        if ignore_patterns:
            config_kwargs['ignore_patterns'] = ignore_patterns
        if keep_patterns:
            config_kwargs['keep_patterns'] = keep_patterns
        
        config = RepomixConfig(**config_kwargs)
        return RepomixEmbedder(config)
    
    @classmethod
    def create_issue_embedder(cls,
                            sentence_model: str = "all-MiniLM-L6-v2",
                            task_name: Optional[str] = None,
                            issues_file: Optional[str] = None,
                            max_issues_per_student: int = 50,
                            use_issue_clustering: bool = True,
                            similarity_threshold: float = 0.8,
                            **kwargs) -> IssueEmbedder:
        """
        Create an Issue embedder with specific configuration.
        
        Args:
            sentence_model: Sentence transformer model name
            task_name: Task name for cache organization
            issues_file: Path to JSON file containing student issues
            max_issues_per_student: Maximum issues per student to consider
            use_issue_clustering: Whether to cluster similar issues
            similarity_threshold: Threshold for issue similarity
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured Issue embedder
        """
        config_kwargs = {
            'sentence_model': sentence_model,
            'task_name': task_name,
            'max_issues_per_student': max_issues_per_student,
            'use_issue_clustering': use_issue_clustering,
            'similarity_threshold': similarity_threshold,
            **kwargs
        }
        
        if issues_file:
            config_kwargs['issues_file'] = issues_file
        
        config = IssueConfig(**config_kwargs)
        return IssueEmbedder(config)
    
    @classmethod
    def get_available_embedders(cls) -> list[str]:
        """
        Get list of available embedder types.
        
        Returns:
            List of embedder type strings
        """
        return [embedder_type.value for embedder_type in EmbedderType]
    
    @classmethod
    def get_embedder_info(cls, embedder_type: str) -> Dict[str, Any]:
        """
        Get information about an embedder type.
        
        Args:
            embedder_type: Type of embedder
            
        Returns:
            Dictionary containing embedder information
        """
        try:
            embedder_enum = EmbedderType(embedder_type.lower())
            embedder_class = cls._EMBEDDER_REGISTRY[embedder_enum]
            config_class = cls._DEFAULT_CONFIGS[embedder_enum]
            
            return {
                'type': embedder_type,
                'class_name': embedder_class.__name__,
                'config_class': config_class.__name__,
                'description': embedder_class.__doc__.strip().split('\n')[0] if embedder_class.__doc__ else "No description available"
            }
        except ValueError:
            available_types = [e.value for e in EmbedderType]
            raise ValueError(f"Unknown embedder type: {embedder_type}. Available: {available_types}")
    
    @classmethod
    def register_embedder(cls, 
                         embedder_type: str,
                         embedder_class: Type[BaseEmbedder],
                         config_class: Type[EmbedderConfig] = EmbedderConfig) -> None:
        """
        Register a new embedder type (for extensions).
        
        Args:
            embedder_type: String identifier for the embedder type
            embedder_class: Embedder class that inherits from BaseEmbedder
            config_class: Configuration class for the embedder
            
        Raises:
            TypeError: If embedder_class doesn't inherit from BaseEmbedder
            ValueError: If embedder_type already exists
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise TypeError("embedder_class must inherit from BaseEmbedder")
        
        # Create enum value (this will raise ValueError if it already exists)
        try:
            embedder_enum = EmbedderType(embedder_type.lower())
            raise ValueError(f"Embedder type {embedder_type} already exists")
        except ValueError as e:
            if "already exists" in str(e):
                raise
            # This is expected for new embedder types
            pass
        
        # For dynamic registration, we'd need to extend the enum
        # For now, just update the registry directly
        cls._EMBEDDER_REGISTRY[embedder_type] = embedder_class
        cls._DEFAULT_CONFIGS[embedder_type] = config_class 