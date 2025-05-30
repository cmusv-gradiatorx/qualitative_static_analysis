#!/usr/bin/env python3
"""
Autograder Main Application

This is the entry point for the graduate-level software engineering assignment autograder.
It processes codebases using repomix and analyzes them with LLMs for qualitative feedback.

Author: Auto-generated
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.config.settings import Settings
from src.core.autograder import AutoGrader
from src.utils.logger import setup_logger


def main():
    """Main entry point for the autograder application."""
    console = Console()
    logger = setup_logger()
    
    try:
        # Display welcome message
        console.print(Panel.fit(
            "[bold blue]Graduate Assignment AutoGrader[/bold blue]\n"
            "Qualitative analysis using LLMs and repomix",
            title="🎓 AutoGrader",
            border_style="blue"
        ))
        
        # Load configuration
        settings = Settings()
        logger.info("Configuration loaded successfully")
        
        # Initialize autograder
        autograder = AutoGrader(settings)
        
        # Process assignments
        results = autograder.process_assignments()
        
        if results:
            # Count successful and failed results
            successful = [r for r in results if r.get('success', False)]
            failed = [r for r in results if not r.get('success', False)]
            
            if successful:
                console.print(f"\n[green]✅ Successfully processed {len(successful)} assignment(s)[/green]")
                for result in successful:
                    console.print(f"  📄 {result['filename']} -> {result['output_file']}")
            
            if failed:
                console.print(f"\n[red]❌ Failed to process {len(failed)} assignment(s)[/red]")
                for result in failed:
                    console.print(f"  📄 {result['filename']} - Error: {result.get('error', 'Unknown error')}")
        else:
            console.print("\n[yellow]⚠️  No assignments found to process[/yellow]")
            console.print(f"Place ZIP files in the '{settings.input_folder}' directory")
            
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 