"""
Repomix Clustering Training Script

Trains k-means clustering on repomix embeddings of student submissions.
Groups similar submissions together for easier grading.

Author: Auto-generated
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from typing import List, Optional
import glob
import logging

from src.faiss.repomix_cluster_manager import RepomixClusterManager
from src.utils.logger import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train clustering on repomix embeddings of student submissions"
    )
    
    parser.add_argument(
        "assignment_path",
        type=str,
        help="Path to assignment directory containing student submissions"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters to create (default: 5)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="starCoder2:3b",
        help="Ollama model name (default: starCoder2:3b)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128000,
        help="Maximum token limit for repomix processing (default: 128000)"
    )
    
    parser.add_argument(
        "--use-compression",
        action="store_true",
        help="Use compression in repomix processing"
    )
    
    parser.add_argument(
        "--remove-comments",
        action="store_true",
        help="Remove comments during processing"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clustering_results",
        help="Directory to save clustering results (default: clustering_results)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging"
    )
    
    return parser.parse_args()


def find_submission_zips(assignment_path: str) -> List[Path]:
    """
    Find all submission ZIP files in the assignment directory.
    
    Args:
        assignment_path: Path to assignment directory
        
    Returns:
        List of paths to submission ZIP files
    """
    # Look for ZIP files in the assignment directory
    zip_pattern = os.path.join(assignment_path, "*.zip")
    zip_files = glob.glob(zip_pattern)
    
    if not zip_files:
        raise ValueError(f"No ZIP files found in {assignment_path}")
    
    return [Path(f) for f in zip_files]


def main():
    """Main function to train clustering on submissions."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging based on verbose flag
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find submission ZIP files
        zip_paths = find_submission_zips(args.assignment_path)
        logger.info(f"Found {len(zip_paths)} submission ZIP files")
        
        # Initialize cluster manager
        cluster_manager = RepomixClusterManager(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            use_compression=args.use_compression,
            remove_comments=args.remove_comments
        )
        
        # Process submissions
        cluster_manager.process_submissions(zip_paths)
        
        # Train clustering
        results = cluster_manager.train_clustering(
            n_clusters=args.n_clusters,
            random_state=args.random_state
        )
        
        # Save results
        output_file = output_dir / "clustering_results.json"
        cluster_manager.save_clustering_results(results, output_file)
        
        # Print summary
        summary = cluster_manager.get_cluster_summary(results)
        print("\n" + summary)
        
        # Save summary to file
        summary_file = output_dir / "clustering_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Clustering results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise


if __name__ == "__main__":
    main() 