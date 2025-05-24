"""
LLM Factory

Factory class for creating LLM provider instances using the Factory pattern.
This provides a centralized way to create LLM providers based on configuration.

Author: Auto-generated
"""

from typing import Dict, Any

from .base import LLMProvider, LLMConfigError
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider


class LLMFactory:
    """
    Factory class for creating LLM provider instances.
    
    This class implements the Factory pattern to create appropriate
    LLM provider instances based on the configuration.
    """
    
    # Registry of available providers
    _providers = {
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
        'ollama': OllamaProvider,
    }
    
    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> LLMProvider:
        """
        Create an LLM provider instance based on configuration.
        
        Args:
            config: Configuration dictionary containing provider type and settings
            
        Returns:
            Configured LLM provider instance
            
        Raises:
            LLMConfigError: If provider type is unsupported or configuration is invalid
        """
        provider_type = config.get('provider')
        
        if not provider_type:
            raise LLMConfigError("Provider type must be specified in configuration")
        
        provider_type = provider_type.lower()
        
        if provider_type not in cls._providers:
            available_providers = ', '.join(cls._providers.keys())
            raise LLMConfigError(
                f"Unsupported provider type: {provider_type}. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls._providers[provider_type]
        
        try:
            return provider_class(config)
        except Exception as e:
            raise LLMConfigError(f"Failed to create {provider_type} provider: {str(e)}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available provider types.
        
        Returns:
            List of available provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new provider type.
        
        This allows for dynamic addition of new providers without modifying the factory.
        
        Args:
            name: Name of the provider
            provider_class: Provider class implementing LLMProvider interface
            
        Raises:
            ValueError: If provider_class doesn't implement LLMProvider interface
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(f"Provider class must inherit from LLMProvider")
        
        cls._providers[name.lower()] = provider_class 