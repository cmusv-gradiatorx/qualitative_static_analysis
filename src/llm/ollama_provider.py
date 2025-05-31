"""
Ollama LLM Provider

Implementation of the LLM provider interface for Ollama-hosted models.
Provides integration with local Ollama instances running Llama and other models.

Author: Auto-generated
"""

import requests
import json
from typing import Dict, Any
import tiktoken

from .base import LLMProvider, LLMConfigError, LLMRequestError


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider implementation.
    
    This class provides integration with Ollama for running local LLM models
    like Llama, CodeLlama, and others.
    """
    
    # Approximate token limits for different Ollama models
    MODEL_LIMITS = {
        'llama3.1': 131072,
        'llama3.1:8b': 131072,
        'llama3.1:70b': 131072,
        'llama3:8b': 8192,
        'llama3:70b': 8192,
        'deepseek-r1': 131072,
        'deepseek-r1:14b': 131072,
        'codellama:7b': 16384,
        'codellama:13b': 16384,
        'deepseek-coder': 16384,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider with base URL and model configuration."""
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.generate_url = f"{self.base_url}/api/generate"
        
        # Initialize tokenizer for token counting (approximate)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        super().__init__(config)
        # Test connection to Ollama
        self._test_connection()
    
    def _validate_config(self) -> None:
        """Validate Ollama-specific configuration."""
        if not self.model:
            raise LLMConfigError("Ollama model name is required")
        
        if not self.base_url:
            raise LLMConfigError("Ollama base URL is required")
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise LLMConfigError(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, num_predict, etc.)
            
        Returns:
            Generated response text
            
        Raises:
            LLMRequestError: If the request fails
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.1),
                    "num_predict": kwargs.get('num_predict', 4096),
                    "top_p": kwargs.get('top_p', 0.95),
                    "top_k": kwargs.get('top_k', 40),
                }
            }
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=kwargs.get('timeout', 300)  # 5 minutes default timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'response' not in result:
                raise LLMRequestError("Invalid response format from Ollama")
            
            return result['response']
            
        except requests.RequestException as e:
            raise LLMRequestError(f"Ollama request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise LLMRequestError(f"Failed to parse Ollama response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken as approximation.
        
        Note: This is an approximation since different models use different tokenizers.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to word-based estimation
            return len(text.split()) * 1.3  # Rough approximation
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit for the current Ollama model."""
        return self.MODEL_LIMITS.get(self.model, 8192) 