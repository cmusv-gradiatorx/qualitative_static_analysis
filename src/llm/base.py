"""
LLM Base Classes

Defines the abstract interface for LLM providers using the Strategy pattern.
This allows for easy addition of new LLM providers without modifying existing code.

Author: Auto-generated
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM implementations must follow,
    enabling the Strategy pattern for pluggable LLM providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Dictionary containing provider-specific configuration
        """
        self.config = config
        self.model = config.get('model')
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the provider-specific configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM for the given prompt.
        
        Args:
            prompt: The input prompt for the LLM
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated response as a string
            
        Raises:
            Exception: If the LLM request fails
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Get the maximum number of tokens supported by this model.
        
        Returns:
            Maximum token limit for the model
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model})"


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConfigError(LLMError):
    """Exception raised for LLM configuration errors."""
    pass


class LLMRequestError(LLMError):
    """Exception raised for LLM request errors."""
    pass 