"""
Gemini LLM Provider

Implementation of the LLM provider interface for Google's Gemini models.
Provides integration with Google's Generative AI SDK.

Author: Auto-generated
"""

import google.generativeai as genai
from typing import Dict, Any, List
import tiktoken
from pathlib import Path
import mimetypes

from .base import LLMProvider, LLMConfigError, LLMRequestError


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider implementation.
    
    This class provides integration with Google's Gemini models through
    the google-generativeai SDK.
    """
    
    # Token limits for different Gemini models
    MODEL_LIMITS = {
        'gemini-1.5-pro': 2097152,
        'gemini-1.5-flash': 1048576,
        'gemini-pro': 32768,
        'gemini-2.5-flash-preview-05-20': 831072,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini provider with API key and model configuration."""
        super().__init__(config)
        
        # Configure the Gemini client
        genai.configure(api_key=self.config['api_key'])
        
        # Initialize the model
        self.client = genai.GenerativeModel(self.model)
        
        # Initialize tokenizer for token counting (approximate)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configure safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
    
    def _validate_config(self) -> None:
        """Validate Gemini-specific configuration."""
        if not self.config.get('api_key'):
            raise LLMConfigError("Gemini API key is required")
        
        if not self.model:
            raise LLMConfigError("Gemini model name is required")
        
        if self.model not in self.MODEL_LIMITS:
            raise LLMConfigError(f"Unsupported Gemini model: {self.model}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_output_tokens, etc.)
            
        Returns:
            Generated response text
            
        Raises:
            LLMRequestError: If the request fails
        """
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.1),
                max_output_tokens=kwargs.get('max_output_tokens', 65536),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 40)
            )
            
            # Generate response
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            if not response.text:
                raise LLMRequestError("Empty response from Gemini")
            
            return response.text
            
        except Exception as e:
            raise LLMRequestError(f"Gemini request failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken as approximation.
        
        Note: This is an approximation since Gemini uses a different tokenizer.
        For production use, consider using Gemini's token counting API.
        
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
        """Get maximum token limit for the current Gemini model."""
        return self.MODEL_LIMITS.get(self.model, 32768)
    
    def supports_multimodal(self) -> bool:
        """
        Check if this Gemini model supports multimodal content.
        
        Returns:
            True - Gemini models support images and PDFs
        """
        return True
    
    def generate_response_with_attachments(self, prompt: str, attachment_paths: List[str], **kwargs) -> str:
        """
        Generate a response using Gemini with image/PDF attachments.
        
        Args:
            prompt: The input prompt
            attachment_paths: List of paths to image/PDF files
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
            
        Raises:
            LLMRequestError: If the request fails
        """
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.1),
                max_output_tokens=kwargs.get('max_output_tokens', 65536),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 40)
            )
            
            # Process attachments
            content_parts = [prompt]
            
            for attachment_path in attachment_paths:
                file_path = Path(attachment_path)
                if not file_path.exists():
                    raise LLMRequestError(f"Attachment file not found: {attachment_path}")
                
                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if not mime_type:
                    # Try to determine based on file extension
                    ext = file_path.suffix.lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    elif ext == '.png':
                        mime_type = 'image/png'
                    elif ext == '.pdf':
                        mime_type = 'application/pdf'
                    else:
                        raise LLMRequestError(f"Unsupported file type: {ext}")
                
                # Upload file to Gemini
                uploaded_file = genai.upload_file(
                    path=str(file_path),
                    mime_type=mime_type
                )
                content_parts.append(uploaded_file)
            
            # Generate response
            response = self.client.generate_content(
                content_parts,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            if not response.text:
                raise LLMRequestError("Empty response from Gemini")
            
            return response.text
            
        except Exception as e:
            raise LLMRequestError(f"Gemini multimodal request failed: {str(e)}") 