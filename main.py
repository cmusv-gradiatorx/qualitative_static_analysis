#!/usr/bin/env python3
"""
Autograder Main Application

This is the entry point for the graduate-level software engineering assignment autograder.
It processes codebases using repomix and analyzes them with LLMs for qualitative feedback.
Enhanced with parallel processing and structured rubric evaluation.

Author: Auto-generated
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
            "[bold blue]Gradiator[/bold blue]\n"
            "Qualitative analysis using LLMs and repomix",
            title="🎓 AutoGrader Enhanced",
            border_style="blue"
        ))
        
        # Load configuration
        settings = Settings()
        logger.info("Configuration loaded successfully")
        
        # Get project configuration for display
        project_config = settings.get_project_config()
        
        # Display configuration info
        console.print(f"\n[cyan]📋 Configuration:[/cyan]")
        console.print(f"  • Project: {settings.project_assignment}")
        console.print(f"  • LLM Provider: {settings.llm_provider}")
        console.print(f"  • Max Parallel LLM: {project_config.get('max_parallel_llm', 2)}")
        console.print(f"  • Prompts Directory: {settings.get_prompts_dir()}")
        
        # Initialize autograder
        autograder = AutoGrader(settings)
        
        # Get status before processing
        status = autograder.get_status()
        console.print(f"\n[cyan]📊 Status:[/cyan]")
        console.print(f"  • Pending assignments: {status['pending_assignments']}")
        console.print(f"  • Semgrep enabled: {status['semgrep_enabled']}")
        
        if status['pending_assignments'] == 0:
            console.print(f"\n[yellow]⚠️  No assignments found to process[/yellow]")
            console.print(f"Place ZIP files in the '{settings.input_folder}' directory")
            return
        
        # Process assignments
        console.print(f"\n[cyan]🚀 Processing {status['pending_assignments']} assignment(s)...[/cyan]")
        results = autograder.process_assignments()
        
        if results:
            # Count successful and failed results
            successful = [r for r in results if r.get('success', False)]
            failed = [r for r in results if not r.get('success', False)]
            
            # Create results table
            table = Table(title="Processing Results")
            table.add_column("Assignment", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Rubric Groups", style="yellow")
            table.add_column("Total Rubrics", style="blue")
            table.add_column("Semgrep Findings", style="red")
            table.add_column("Output Files", style="magenta")
            
            for result in results:
                if result.get('success', False):
                    output_files = result.get('output_files', {})
                    file_list = ", ".join([f.name for f in output_files.values()])
                    
                    table.add_row(
                        result['filename'],
                        "✅ Success",
                        str(result.get('rubric_groups_processed', 'N/A')),
                        str(result.get('total_rubrics', 'N/A')),
                        str(result.get('semgrep_findings', 'N/A')),
                        file_list
                    )
                else:
                    table.add_row(
                        result['filename'],
                        "❌ Failed",
                        "N/A",
                        "N/A",
                        "N/A",
                        f"Error: {result.get('error', 'Unknown')}"
                    )
            
            console.print(table)
            
            if successful:
                console.print(f"\n[green]✅ Successfully processed {len(successful)} assignment(s)[/green]")
                console.print(f"📁 Output directory: {settings.output_folder}")
                
                # Show detailed output for each successful result
                for result in successful:
                    console.print(f"\n[cyan]📄 {result['filename']}:[/cyan]")
                    console.print(f"  • Project: {result.get('project_name', 'Unknown')}")
                    console.print(f"  • Token count: {result.get('token_count', 0):,}")
                    console.print(f"  • Compressed: {'Yes' if result.get('compressed', False) else 'No'}")
                    console.print(f"  • Parallel groups: {result.get('rubric_groups_processed', 0)}")
                    
                    output_files = result.get('output_files', {})
                    for file_type, file_path in output_files.items():
                        console.print(f"  • {file_type}: {file_path.name}")
            
            if failed:
                console.print(f"\n[red]❌ Failed to process {len(failed)} assignment(s)[/red]")
                for result in failed:
                    console.print(f"  📄 {result['filename']} - Error: {result.get('error', 'Unknown error')}")
        else:
            console.print("\n[yellow]⚠️  No results returned[/yellow]")
            
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 