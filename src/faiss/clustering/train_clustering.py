#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Clustering Models for Assignment Grading

This script trains 2-layer clustering models (score-based and issue-based) using the existing
StarCoder2 embeddings and grade mapping data. Provides an alternative to FAISS similarity search
using sklearn clustering algorithms.

Usage:
    python src/faiss/clustering/train_clustering.py --assignment task4_GildedRoseKata

Author: Auto-generated
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    config_path = Path(__file__).parent.parent.parent.parent / "config.env"
    if config_path.exists():
        load_dotenv(config_path)
        print(f"[SUCCESS] Loaded configuration from {config_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.faiss.clustering.cluster_manager import ClusteringManager
from src.faiss.processor import MultiFolderProcessor
from src.faiss.embedder import create_java_embedder
from src.utils.logger import get_logger, setup_logger


def make_json_serializable(obj):
    """Convert NumPy types to Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train clustering models for assignment grading")
    
    parser.add_argument(
        "--assignment",
        type=str,
        required=True,
        help="Assignment ID (e.g., task4_GildedRoseKata)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/faiss/data",
        help="Directory containing task folders with submissions"
    )
    
    parser.add_argument(
        "--grade-mapping",
        type=str,
        default=None,
        help="Path to CSV with grade mapping (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/faiss/clustering/models",
        help="Output directory for trained models"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="starCoder2:15b",
        help="Ollama model name for embeddings"
    )
    

    
    parser.add_argument(
        "--issue-algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan", "hierarchical"],
        help="Algorithm for issue-based clustering"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def find_grade_mapping_csv(assignment_id: str) -> str:
    """Auto-detect grade mapping CSV file"""
    possible_paths = [
        f"src/faiss/grade_mapping/{assignment_id}.csv",
        f"grade_mapping/{assignment_id}.csv",
        f"data/{assignment_id}_grades.csv",
        f"data/grades/{assignment_id}.csv"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError(f"Could not find grade mapping CSV for {assignment_id}. Tried: {possible_paths}")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info(f"Starting clustering training for assignment: {args.assignment}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Find grade mapping file
        if args.grade_mapping:
            grade_mapping_path = args.grade_mapping
        else:
            grade_mapping_path = find_grade_mapping_csv(args.assignment)
        
        logger.info(f"Using grade mapping: {grade_mapping_path}")
        
        # Load submissions from data directory
        logger.info(f"Loading submissions from {args.data_dir}")
        multi_processor = MultiFolderProcessor(args.data_dir)
        submissions_by_assignment = multi_processor.process_all_tasks()
        
        if args.assignment not in submissions_by_assignment:
            logger.error(f"Assignment {args.assignment} not found in data directory")
            logger.info(f"Available assignments: {list(submissions_by_assignment.keys())}")
            sys.exit(1)
        
        submissions = submissions_by_assignment[args.assignment]
        logger.info(f"Found {len(submissions)} submissions for {args.assignment}")
        
        # Initialize clustering manager
        logger.info("Initializing clustering manager...")
        
        # Create embedder with Ollama (default)
        embedder = create_java_embedder(
            model_name=args.model_name,
            use_ollama=True
        )
        
        clustering_manager = ClusteringManager(args.assignment, embedder)
        
        # Load submissions and grades
        logger.info("Loading submissions and grade mapping...")
        clustering_manager.load_submissions_and_grades(submissions, grade_mapping_path)
        
        if len(clustering_manager.submissions) == 0:
            logger.error("No submissions matched with grades. Check student name mapping.")
            sys.exit(1)
        
        # Generate embeddings
        logger.info("Generating StarCoder2 embeddings...")
        clustering_manager.generate_embeddings()
        
        # Train score-based clustering
        logger.info("Training score-based clustering...")
        score_results = clustering_manager.train_score_clustering()
        logger.info(f"Score clustering results: {json.dumps(make_json_serializable(score_results), indent=2)}")
        
        # Train issue-based clustering
        logger.info(f"Training issue-based clustering with {args.issue_algorithm}...")
        issue_results = clustering_manager.train_issue_clustering(args.issue_algorithm)
        logger.info(f"Issue clustering results: {json.dumps(make_json_serializable(issue_results), indent=2)}")
        
        # Save models
        output_path = Path(args.output_dir) / args.assignment
        logger.info(f"Saving models to {output_path}")
        clustering_manager.save_models(str(output_path))
        
        # Generate and save comprehensive report
        report = {
            'assignment_id': args.assignment,
            'training_timestamp': clustering_manager.score_cluster_info,  # This contains timestamp
            'model_config': {
                'model_name': args.model_name,
                'ollama_url': os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'issue_algorithm': args.issue_algorithm
            },
            'data_stats': {
                'total_submissions': len(clustering_manager.submissions),
                'embedding_dimension': clustering_manager.embeddings.shape[1],
                'unique_scores': list(set(sub.score for sub in clustering_manager.submissions))
            },
            'score_clustering': score_results,
            'issue_clustering': issue_results,
            'cluster_summary': clustering_manager.get_cluster_summary()
        }
        
        report_path = output_path / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(make_json_serializable(report), f, indent=2, default=str)
        
        # Print human-readable summary
        print("\n" + "="*80)
        print("[SUCCESS] Clustering Training Complete!")
        print("="*80)
        print(f"[ASSIGNMENT] {args.assignment}")
        print(f"[SUBMISSIONS] {len(clustering_manager.submissions)} submissions processed")
        print(f"[EMBEDDINGS] {clustering_manager.embeddings.shape[1]}-dimensional StarCoder2 embeddings")
        print(f"[MODELS] Saved to: {output_path}")
        
        print(f"\n[SCORE CLUSTERING]")
        print(f"  Clusters: {score_results['n_clusters']}")
        print(f"  Silhouette Score: {score_results['silhouette_score']:.3f}")
        
        # Show predefined ranges instead of unique scores
        if 'score_ranges' in score_results:
            ranges_str = ", ".join([f"{int(r[0]*100)}-{int(r[1]*100)}%" for r in score_results['score_ranges']])
            print(f"  Score Ranges: {ranges_str}")
        
        # Show actual unique scores found in data
        unique_scores = list(set(sub.score for sub in clustering_manager.submissions))
        print(f"  Unique Scores Found: {sorted(unique_scores)}")
        
        print(f"\n[ISSUE CLUSTERING]")
        print(f"  Algorithm: {issue_results['algorithm']}")
        print(f"  Clusters: {issue_results['n_clusters']}")
        print(f"  Silhouette Score: {issue_results['silhouette_score']:.3f}")
        
        print(f"\n[CLUSTER SUMMARY]")
        print(clustering_manager.get_cluster_summary())
        
        print(f"\n[USAGE] To use the trained models:")
        print(f"  from src.faiss.clustering.cluster_manager import ClusteringManager")
        print(f"  manager = ClusteringManager('{args.assignment}')")
        print(f"  manager.load_models('{output_path}')")
        print(f"  result = manager.predict_clusters(new_submission)")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 