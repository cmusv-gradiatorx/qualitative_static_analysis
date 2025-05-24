"""
OpenAI LLM Provider

Implementation of the LLM provider interface for OpenAI models.
Provides integration with OpenAI's API.

Author: Auto-generated
"""

from openai import OpenAI
from typing import Dict, Any
import tiktoken

from .base import LLMProvider, LLMConfigError, LLMRequestError


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.
    
    This class provides integration with OpenAI's models through
    the official OpenAI Python SDK.
    """
    
    # Token limits for different OpenAI models
    MODEL_LIMITS = {
        'gpt-4o': 128000,
        'gpt-4o-mini': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4': 8192,
        'gpt-3.5-turbo': 16385,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider with API key and model configuration."""
        super().__init__(config)
        
        # Initialize the OpenAI client
        client_kwargs = {'api_key': self.config['api_key']}
        if self.config.get('org_id'):
            client_kwargs['organization'] = self.config['org_id']
        
        self.client = OpenAI(**client_kwargs)
        
        # Initialize tokenizer for the specific model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration."""
        if not self.config.get('api_key'):
            raise LLMConfigError("OpenAI API key is required")
        
        if not self.model:
            raise LLMConfigError("OpenAI model name is required")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI's API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
            
        Raises:
            LLMRequestError: If the request fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 4096),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LLMRequestError("Empty response from OpenAI")
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMRequestError(f"OpenAI request failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model-specific tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Exact token count for the model
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to word-based estimation
            return len(text.split()) * 1.3  # Rough approximation
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit for the current OpenAI model."""
        return self.MODEL_LIMITS.get(self.model, 8192) 