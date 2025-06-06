#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Assignment-Specific FAISS Indices Script

This script builds separate FAISS indices for each assignment (task folder) to improve
efficiency and provide better contextual results. Each assignment gets its own
dedicated FAISS index using Java code embeddings with Ollama StarCoder2.

Usage:
    python src/faiss/build_assignment_indices.py [options]

Author: Auto-generated
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# Load environment variables from config.env
try:
    from dotenv import load_dotenv
    # Load from config.env in the project root
    config_path = Path(__file__).parent.parent.parent / "config.env"
    if config_path.exists():
        load_dotenv(config_path)
        print(f"[SUCCESS] Loaded configuration from {config_path}")
    else:
        print(f"[WARNING] Config file not found at {config_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed. Install with: pip install python-dotenv")
    print("[WARNING] Will use system environment variables only")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.faiss.processor import MultiFolderProcessor, TaskFolderProcessor
from src.faiss.embedder import create_java_embedder
from src.faiss.assignment_faiss_manager import AssignmentFAISSManager
from src.utils.logger import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build assignment-specific FAISS indices for Java code using Ollama")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/faiss/data",
        help="Directory containing task folders with individual ZIP files"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="src/faiss/assignment_indices",
        help="Output directory for assignment-specific FAISS indices"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="starcoder2:3b",
        help="Ollama model name for Java code embeddings (e.g., starcoder2:3b, qwen2.5-coder:7b)"
    )
    
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        default=True,
        help="Use Ollama API for embeddings (default: True)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama base URL (loads from OLLAMA_BASE_URL env var if not specified)"
    )
    
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length for code tokenization (Ollama supports longer contexts)"
    )
    
    parser.add_argument(
        "--assignments",
        type=str,
        nargs="+",
        default=None,
        help="Specific assignment IDs to process (default: all found task folders)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_requirements():
    """Validate that required packages are available"""
    missing_packages = []
    
    try:
        import faiss
    except ImportError:
        missing_packages.append("faiss-cpu or faiss-gpu")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import requests
    except ImportError:
        missing_packages.append("requests")
    
    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def test_ollama_connection(ollama_base_url: str, model_name: str) -> bool:
    """Test connection to Ollama and check if model is available"""
    try:
        import requests
        
        # Test basic connection
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        if response.status_code != 200:
            print(f"[ERROR] Cannot connect to Ollama at {ollama_base_url}")
            return False
        
        # Check if model is available
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if not any(model_name in name for name in model_names):
            print(f"[WARNING] Model {model_name} not found in Ollama")
            print(f"Available models: {model_names}")
            print(f"To pull the model, run: ollama pull {model_name}")
            return False
        
        print(f"[SUCCESS] Ollama connection successful, model {model_name} available")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing Ollama connection: {e}")
        return False


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting assignment-specific FAISS index build process for Java code with Ollama")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate requirements
    if not validate_requirements():
        sys.exit(1)
    
    # Load Ollama URL from environment if not provided
    ollama_base_url = args.ollama_url
    if ollama_base_url is None:
        ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # Test Ollama connection
    if args.use_ollama:
        logger.info(f"Testing Ollama connection at {ollama_base_url}")
        if not test_ollama_connection(ollama_base_url, args.model_name):
            logger.error("Ollama connection failed. Please check your setup.")
            sys.exit(1)
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    try:
        # Process all task folders to extract submissions
        logger.info(f"Processing task folders in directory: {args.data_dir}")
        multi_processor = MultiFolderProcessor(args.data_dir)
        
        # Extract submissions from all task folders
        submissions_by_assignment = multi_processor.process_all_tasks()
        
        if not submissions_by_assignment:
            logger.error("No submissions found in any task folders")
            sys.exit(1)
        
        logger.info(f"Found submissions for {len(submissions_by_assignment)} assignments")
        
        # Get statistics
        overall_stats = multi_processor.get_overall_statistics(submissions_by_assignment)
        logger.info(f"Total submissions: {overall_stats['total_submissions']}")
        
        for assignment_id, submissions in submissions_by_assignment.items():
            logger.info(f"Assignment '{assignment_id}': {len(submissions)} submissions")
        
        # Filter assignments if specified
        if args.assignments:
            filtered_submissions = {aid: subs for aid, subs in submissions_by_assignment.items() 
                                  if aid in args.assignments}
            if not filtered_submissions:
                logger.error(f"None of the specified assignments found: {args.assignments}")
                sys.exit(1)
            submissions_by_assignment = filtered_submissions
            logger.info(f"Processing specified assignments: {list(submissions_by_assignment.keys())}")
        
        # Initialize Java code embedder with Ollama
        logger.info(f"Initializing Java code embedder with Ollama model: {args.model_name}")
        embedder = create_java_embedder(
            model_name=args.model_name,
            use_ollama=args.use_ollama,
            ollama_base_url=ollama_base_url,
            max_length=args.max_length
        )
        
        logger.info(f"Embedder initialized: {embedder.get_embedding_info()}")
        
        # Initialize assignment FAISS manager
        logger.info(f"Initializing Assignment FAISS manager")
        assignment_manager = AssignmentFAISSManager(
            base_index_path=args.output_path,
            index_type=args.index_type
        )
        
        # Build assignment-specific indices
        logger.info("Building assignment-specific FAISS indices...")
        build_stats = assignment_manager.build_assignment_indices(
            submissions_by_assignment=submissions_by_assignment,
            embedder=embedder
        )
        
        logger.info("Assignment-specific indices build completed successfully!")
        
        # Save comprehensive metadata
        metadata_path = Path(args.output_path) / "build_metadata.json"
        comprehensive_metadata = {
            'build_args': vars(args),
            'overall_stats': overall_stats,
            'build_stats': build_stats,
            'embedding_info': embedder.get_embedding_info(),
            'data_structure': 'task_folders_with_individual_zips',
            'data_directory': args.data_dir,
            'total_submissions': overall_stats['total_submissions'],
            'assignments_built': list(build_stats.keys()),
            'build_type': 'assignment_specific_java_ollama',
            'model_name': args.model_name,
            'ollama_base_url': ollama_base_url,
            'max_length': args.max_length
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)
        logger.info(f"Saved comprehensive build metadata to {metadata_path}")
        
        # Final summary
        print("\n" + "="*80)
        print("[SUCCESS] Assignment-Specific Java FAISS Indices Build Complete!")
        print(f"[MODEL] Using Ollama with {args.model_name}")
        print(f"[PATH] Indices saved to: {args.output_path}")
        print(f"[STATS] Assignments processed: {len(build_stats)}")
        
        total_indexed = sum(stats['valid_embeddings'] for stats in build_stats.values())
        print(f"[TOTAL] Total Java submissions indexed: {total_indexed}")
        
        print(f"[MODEL] Model: {args.model_name}")
        print(f"[URL] Ollama URL: {ollama_base_url}")
        print(f"[DIM] Embedding dimension: {embedder.get_embedding_info()['embedding_dim']}")
        print(f"[TYPE] Index type: {args.index_type}")
        print(f"[LENGTH] Max sequence length: {args.max_length}")
        
        print("\nPer-Assignment Statistics:")
        for assignment_id, stats in build_stats.items():
            print(f"  • {assignment_id}: {stats['valid_embeddings']} Java submissions indexed")
        
        print("="*80)
        
        # Usage examples
        print("\n[USAGE] Usage Examples:")
        print("  # Search within specific assignment:")
        print(f"  from src.faiss.assignment_faiss_manager import AssignmentFAISSManager")
        print(f"  from src.faiss.embedder import create_java_embedder")
        print(f"  manager = AssignmentFAISSManager('{args.output_path}')")
        print("  manager.load_assignment_indices()")
        print(f"  embedder = create_java_embedder(model_name='{args.model_name}')")
        print("  results = manager.search_similar_in_assignment('assignment_id', query_embedding)")
        print("\n  # Search across multiple assignments:")
        print("  results = manager.search_across_assignments(query_embedding, assignment_ids=['id1', 'id2'])")
        
        print(f"\n[EVAL] Quick Start Evaluation:")
        print(f"  python evaluate_embedder_performance.py --model-name {args.model_name}")
        
    except Exception as e:
        logger.error(f"Failed to build assignment indices: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 