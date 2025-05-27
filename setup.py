#!/usr/bin/env python3
"""
AutoGrader Setup Script

Helps set up the autograder environment and dependencies.
Run this script after installing Python dependencies.

Author: Auto-generated
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm


def main():
    """Main setup function."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]AutoGrader Setup[/bold blue]\n"
        "Setting up your graduate assignment evaluation system",
        title="🎓 Setup",
        border_style="blue"
    ))
    
    # Check Python version
    if sys.version_info < (3, 8):
        console.print("[red]❌ Python 3.8 or higher is required[/red]")
        sys.exit(1)
    
    console.print("[green]✅ Python version check passed[/green]")
    
    # Check if repomix is available
    try:
        result = subprocess.run(['npx', 'repomix', '--version'], 
                               capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            console.print(f"[green]✅ Repomix available: {result.stdout.strip()}[/green]")
        else:
            console.print("[yellow]⚠️  Repomix not found. Will install on first use.[/yellow]")
    except Exception:
        console.print("[yellow]⚠️  Could not check repomix. Ensure Node.js is installed.[/yellow]")
    
    # Create directories
    directories = ['input', 'output', 'temp', 'logs', 'prompts', 'config']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        console.print(f"[green]✅ Created directory: {directory}/[/green]")
    
    # Create template files if they don't exist
    console.print("\n[bold]Creating Template Files[/bold]")
    
    # Create prompts template files
    prompt_files = {
        'prompts/instruction_content.txt': 'Add your assignment instructions here.',
        'prompts/rubric_content.txt': 'Add your assignment rubric here.',
        'prompts/static_instructions.txt': 'Add your assignment static analysis instructions here.'
    }
    
    for file_path, content in prompt_files.items():
        if not Path(file_path).exists():
            Path(file_path).write_text(content, encoding='utf-8')
            console.print(f"[green]✅ Created template: {file_path}[/green]")
        else:
            console.print(f"[yellow]⚠️  File exists: {file_path}[/yellow]")
    
    # Create config template files
    config_files = {
        'config/semgrep_rules.yaml': 'Add your semgrep rules here.'
    }
    
    for file_path, content in config_files.items():
        if not Path(file_path).exists():
            Path(file_path).write_text(content, encoding='utf-8')
            console.print(f"[green]✅ Created template: {file_path}[/green]")
        else:
            console.print(f"[yellow]⚠️  File exists: {file_path}[/yellow]")
    
    # Configuration setup
    console.print("\n[bold]Configuration Setup[/bold]")
    
    if not Path('config.env').exists():
        console.print("[red]❌ config.env not found[/red]")
        sys.exit(1)
    
    # Ask user to configure LLM provider
    console.print("\n[bold]LLM Provider Configuration[/bold]")
    
    providers = {
        '1': 'gemini',
        '2': 'openai', 
        '3': 'ollama'
    }
    
    console.print("Available LLM providers:")
    console.print("1. Google Gemini")
    console.print("2. OpenAI GPT")
    console.print("3. Ollama (Local)")
    
    choice = Prompt.ask("Choose your LLM provider", choices=['1', '2', '3'], default='1')
    selected_provider = providers[choice]
    
    # Update config file
    config_lines = []
    with open('config.env', 'r') as f:
        config_lines = f.readlines()
    
    # Update LLM_PROVIDER line
    for i, line in enumerate(config_lines):
        if line.startswith('LLM_PROVIDER='):
            config_lines[i] = f'LLM_PROVIDER={selected_provider}\n'
            break
    
    with open('config.env', 'w') as f:
        f.writelines(config_lines)
    
    console.print(f"[green]✅ Set LLM provider to: {selected_provider}[/green]")
    
    # Provider-specific setup
    if selected_provider == 'gemini':
        console.print("\n[bold yellow]Gemini Setup Required:[/bold yellow]")
        console.print("1. Go to https://makersuite.google.com/app/apikey")
        console.print("2. Create an API key")
        console.print("3. Update GEMINI_API_KEY in config.env")
        
    elif selected_provider == 'openai':
        console.print("\n[bold yellow]OpenAI Setup Required:[/bold yellow]")
        console.print("1. Go to https://platform.openai.com/api-keys")
        console.print("2. Create an API key") 
        console.print("3. Update OPENAI_API_KEY in config.env")
        
    elif selected_provider == 'ollama':
        console.print("\n[bold yellow]Ollama Setup Required:[/bold yellow]")
        console.print("1. Install Ollama: https://ollama.ai/")
        console.print("2. Run: ollama pull llama3.1:8b")
        console.print("3. Start Ollama server: ollama serve")
    
    # Test configuration
    if Confirm.ask("\nWould you like to test the configuration now?"):
        console.print("\n[bold]Testing Configuration...[/bold]")
        try:
            from src.config.settings import Settings
            settings = Settings()
            console.print("[green]✅ Configuration loaded successfully[/green]")
            
            # Test LLM provider
            from src.llm.factory import LLMFactory
            llm_config = settings.get_llm_config()
            provider = LLMFactory.create_provider(llm_config)
            console.print(f"[green]✅ LLM provider initialized: {provider}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Configuration test failed: {str(e)}[/red]")
            console.print("Please check your API keys in config.env")
    
    # Final instructions
    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "Next steps:\n"
        "1. Update your API keys in config.env\n"
        "2. Place ZIP files in the input/ directory\n"
        "3. Run: python main.py\n"
        "4. Check results in the output/ directory",
        title="🎉 Ready to Go",
        border_style="green"
    ))


if __name__ == "__main__":
    main() 