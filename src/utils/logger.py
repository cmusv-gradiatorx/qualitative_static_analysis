"""
Logging Utilities

Provides consistent logging configuration across the application.
Uses structured logging with appropriate formatting and levels.

Author: Auto-generated
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up the main application logger.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    logger = logging.getLogger("autograder")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
    else:
        file_handler = logging.FileHandler(logs_dir / "autograder.log")
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"autograder.{name}")


class LogContext:
    """
    Context manager for temporary logging configuration.
    
    Useful for temporarily changing log levels or adding handlers.
    """
    
    def __init__(self, logger: logging.Logger, level: Optional[str] = None):
        """
        Initialize log context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper()) if level else None
        self.original_level = logger.level
    
    def __enter__(self):
        """Enter the context."""
        if self.new_level:
            self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        self.logger.setLevel(self.original_level) 