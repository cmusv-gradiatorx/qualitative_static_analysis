"""
Configuration Settings

Handles loading and validation of configuration from environment variables.
Uses Singleton pattern to ensure consistent configuration across the application.

Author: Auto-generated
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


class Settings:
    """
    Application settings loaded from environment variables.
    
    This class implements a singleton pattern to ensure configuration
    consistency across the application.
    """
    
    _instance: Optional['Settings'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'Settings':
        """Singleton implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize settings from environment variables."""
        if Settings._initialized:
            return
            
        # Load environment variables
        load_dotenv('config.env')
        
        # LLM Configuration
        self.llm_provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
        
        # Gemini settings
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.openai_org_id = os.getenv('OPENAI_ORG_ID')
        
        # Ollama settings
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        
        # Application paths
        self.input_folder = Path(os.getenv('INPUT_FOLDER', 'input'))
        self.output_folder = Path(os.getenv('OUTPUT_FOLDER', 'output'))
        self.temp_folder = Path(os.getenv('TEMP_FOLDER', 'temp'))
        
        # Repomix configuration (global defaults)
        self.max_tokens = int(os.getenv('MAX_TOKENS', '128000'))
        self.use_compression = os.getenv('USE_COMPRESSION', 'true').lower() == 'true'
        self.remove_comments = os.getenv('REMOVE_COMMENTS', 'false').lower() == 'true'
        
        # Project assignment configuration (only this comes from main config)
        self.project_assignment = os.getenv('PROJECT_ASSIGNMENT', 'functional_programming_milestone_3')
        
        # Assignment-specific settings are loaded dynamically from project config via get_project_config()
        # This includes: max_file_size (for file filtering), enable_semgrep_analysis, semgrep_rules_file, 
        # semgrep_timeout, max_parallel_llm, ignore_patterns, keep_patterns
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Validate configuration
        self._validate_configuration()
        
        Settings._initialized = True
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.input_folder, self.output_folder, self.temp_folder]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self) -> None:
        """Validate that required configuration is present."""
        if self.llm_provider not in ['gemini', 'openai', 'ollama']:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        if self.llm_provider == 'gemini' and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when using Gemini")
        
        if self.llm_provider == 'openai' and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")
    
    def get_llm_config(self) -> dict:
        """Get configuration for the selected LLM provider."""
        if self.llm_provider == 'gemini':
            return {
                'provider': 'gemini',
                'api_key': self.gemini_api_key,
                'model': self.gemini_model
            }
        elif self.llm_provider == 'openai':
            return {
                'provider': 'openai',
                'api_key': self.openai_api_key,
                'model': self.openai_model,
                'org_id': self.openai_org_id
            }
        elif self.llm_provider == 'ollama':
            return {
                'provider': 'ollama',
                'base_url': self.ollama_base_url,
                'model': self.ollama_model
            }
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def get_project_config(self) -> Dict[str, Any]:
        """
        Get project-specific configuration from project config file.
        
        Returns:
            Dictionary containing project-specific settings
        """
        project_config_file = Path(f"config/projects/{self.project_assignment}.json")
        
        # Default configuration if project config doesn't exist
        default_config = {
            "max_file_size": 1000000,  # 1MB default file size limit for filtering (in bytes)
            "ignore_patterns": ["*.pyc", "__pycache__", ".git", "*.log"],
            "keep_patterns": ["*.py", "*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
            "max_parallel_llm": 2,  # Default value
            "enable_semgrep_analysis": False,  # Default value  
            "semgrep_rules_file": "config/semgrep_rules.yaml",  # Default value
            "semgrep_timeout": 300  # Default value
        }
        
        if project_config_file.exists():
            try:
                with open(project_config_file, 'r', encoding='utf-8') as f:
                    project_config = json.load(f)
                    default_config.update(project_config)
            except Exception as e:
                print(f"Warning: Failed to load project config {project_config_file}: {e}")
                print("Using default configuration")
        
        # Validate project-specific settings
        if default_config.get('max_parallel_llm', 1) < 1:
            raise ValueError("max_parallel_llm must be at least 1")
        
        return default_config
    
    def get_prompts_dir(self) -> Path:
        """
        Get the prompts directory for the current project assignment.
        
        Returns:
            Path to the project-specific prompts directory
        """
        return Path(f"prompts/{self.project_assignment}")
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return f"Settings(llm_provider='{self.llm_provider}', max_tokens={self.max_tokens}, project='{self.project_assignment}')" 