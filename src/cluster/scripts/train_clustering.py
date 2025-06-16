#!/usr/bin/env python3
"""
Train Clustering Model Script

Trains clustering models on student submissions using the clean clustering architecture.
Supports both Java and Repomix embedders with various clustering algorithms.

Usage:
    python src/cluster/scripts/train_clustering.py \
        --task-folder data/task1_SOLID \
        --embedder-type java \
        --model-name starCoder2:7b \
        --algorithm kmeans \
        --n-clusters 5 \
        --output-dir models/task1_java_kmeans

Author: Auto-generated
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from config.env
config_path = Path(__file__).parent.parent.parent.parent / 'config.env'
if config_path.exists():
    load_dotenv(config_path)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.cluster.embedders import EmbedderFactory
from src.cluster.processors import SubmissionProcessor
from src.cluster.clustering import ClusterManager
from src.utils.logger import setup_logger, get_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train clustering model on student submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Java embedder with K-means
  python src/cluster/scripts/train_clustering.py src/faiss/data/task1_SOLID \\
      --embedder-type java \\
      --model-name starCoder2:3b \\
      --algorithm kmeans \\
      --n-clusters 5

  # Train Repomix embedder with DBSCAN
  python src/cluster/scripts/train_clustering.py src/faiss/data/task1_SOLID \\
      --embedder-type repomix \\
      --model-name starCoder2:3b \\
      --algorithm dbscan \\
      --eps 0.3 \\
      --min-samples 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "task_folder",
        type=str,
        help="Path to task folder containing student ZIP files"
    )
    
    # Embedder configuration
    parser.add_argument(
        "--embedder-type",
        type=str,
        required=True,
        choices=["java", "repomix", "issues"],
        help="Type of embedder to use"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="starCoder2:3b",
        help="Model name for code embedders (java/repomix) (default: starCoder2:7b)"
    )
    
    # Clustering configuration
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan", "hierarchical", "gmm"],
        help="Clustering algorithm (default: kmeans)"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters (ignored for DBSCAN) (default: 5)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Algorithm-specific parameters
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps parameter (default: 0.5)"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples parameter (default: 2)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained model (auto-generated if not provided)"
    )
    
    # Processing options
    parser.add_argument(
        "--java-only",
        action="store_true",
        help="Only process submissions with Java files"
    )
    
    parser.add_argument(
        "--min-java-files",
        type=int,
        default=1,
        help="Minimum number of Java files required (default: 1)"
    )
    
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        default=True,
        help="Cache embeddings for faster repeated training"
    )
    
    # Issues embedder specific options
    parser.add_argument(
        "--issues-file",
        type=str,
        default=None,
        help="Path to JSON file containing student issues (for issues embedder)"
    )
    
    parser.add_argument(
        "--sentence-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for issues embedder (default: all-MiniLM-L6-v2)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (logs to console only if not provided)"
    )
    
    return parser.parse_args()


def create_output_directory(args) -> Path:
    """Create output directory with auto-generated name if needed"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate output directory name
        task_name = Path(args.task_folder).name
        
        # Use appropriate model name based on embedder type
        if args.embedder_type == "issues":
            model_suffix = args.sentence_model.replace('-', '_').replace(':', '_')
        else:
            model_suffix = args.model_name.replace(':', '_')
        
        # Include number of clusters in the filename
        if args.algorithm == "dbscan":
            cluster_suffix = f"eps{args.eps}_min{args.min_samples}"
        else:
            cluster_suffix = f"{args.n_clusters}clusters"
        
        output_name = f"{task_name}_{args.embedder_type}_{args.algorithm}_{model_suffix}_{cluster_suffix}"
        output_dir = Path("models") / output_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level, args.log_file)
    logger = get_logger(__name__)
    
    logger.info("Starting clustering training")
    logger.info(f"Task folder: {args.task_folder}")
    logger.info(f"Embedder: {args.embedder_type}")
    if args.embedder_type == "issues":
        logger.info(f"Sentence Model: {args.sentence_model}")
    else:
        logger.info(f"Model: {args.model_name}")
    logger.info(f"Algorithm: {args.algorithm}")
    
    try:
        # Validate task folder
        task_folder = Path(args.task_folder)
        if not task_folder.exists():
            logger.error(f"Task folder not found: {task_folder}")
            sys.exit(1)
        
        # Create output directory
        output_dir = create_output_directory(args)
        logger.info(f"Output directory: {output_dir}")
        
        # Extract task name from task folder for cache organization
        task_name = task_folder.name
        logger.info(f"Task name: {task_name}")
        
        # Initialize processor based on embedder type
        if args.embedder_type == "issues":
            # Use issue processor for issues embedder
            if not args.issues_file:
                logger.error("--issues-file is required for issues embedder")
                sys.exit(1)
            
            from src.cluster.processors.issue_processor import IssueProcessor
            logger.info("Initializing issue processor...")
            processor = IssueProcessor(args.issues_file)
            
            # Extract submissions
            logger.info("Extracting issue-based submissions...")
            submissions = processor.extract_submissions(min_issues=1)
        else:
            # Use regular submission processor for code-based embedders
            logger.info("Initializing submission processor...")
            processor = SubmissionProcessor(str(task_folder))
            
            # Extract submissions
            logger.info("Extracting submissions...")
            submissions = processor.extract_submissions(java_only=args.java_only)
        
        if not submissions:
            logger.error("No submissions found or processed")
            sys.exit(1)
        
        # Filter submissions
        if args.min_java_files > 1:
            submissions = processor.filter_submissions(
                submissions, 
                min_java_files=args.min_java_files
            )
        
        logger.info(f"Processing {len(submissions)} submissions")
        
        # Get task statistics
        stats = processor.get_task_statistics(submissions)
        logger.info(f"Task statistics: {stats}")
        
        # Create embedder
        logger.info(f"Creating {args.embedder_type} embedder...")
        embedder_kwargs = {
            'cache_embeddings': args.cache_embeddings,
            'task_name': task_name
        }
        
        # Add embedder-specific parameters
        if args.embedder_type == "repomix":
            embedder_kwargs.update({
                'use_compression': True,
                'remove_comments': False
            })
        elif args.embedder_type == "issues":
            embedder_kwargs.update({
                'sentence_model': args.sentence_model,
                'use_issue_clustering': True,
                'similarity_threshold': 0.8
            })
            if args.issues_file:
                embedder_kwargs['issues_file'] = args.issues_file
        
        # For issues embedder, use the factory method that doesn't require model_name
        if args.embedder_type == "issues":
            embedder = EmbedderFactory.create_issue_embedder(**embedder_kwargs)
        else:
            embedder = EmbedderFactory.create_embedder(
                embedder_type=args.embedder_type,
                model_name=args.model_name,
                **embedder_kwargs
            )
        
        logger.info(f"Embedder info: {embedder.get_embedding_info()}")
        
        # Create cluster manager
        logger.info("Initializing cluster manager...")
        cluster_manager = ClusterManager(embedder)
        
        # Prepare algorithm parameters
        algorithm_kwargs = {}
        if args.algorithm == "dbscan":
            algorithm_kwargs = {
                'eps': args.eps,
                'min_samples': args.min_samples
            }
        
        # Train clustering
        logger.info(f"Training {args.algorithm} clustering...")
        result = cluster_manager.fit(
            submissions=submissions,
            algorithm=args.algorithm,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            **algorithm_kwargs
        )
        
        # Log results
        logger.info(f"Clustering completed:")
        logger.info(f"  - Algorithm: {result.algorithm}")
        logger.info(f"  - Clusters found: {result.n_clusters}")
        logger.info(f"  - Silhouette score: {result.silhouette_score:.3f}")
        logger.info(f"  - Calinski-Harabasz score: {result.calinski_harabasz_score:.3f}")
        if result.inertia is not None:
            logger.info(f"  - Inertia: {result.inertia:.3f}")
        
        # Get cluster analysis
        cluster_analysis = cluster_manager.get_cluster_analysis()
        
        # Log cluster details
        for cluster_id, analysis in cluster_analysis.items():
            students = ", ".join(analysis['student_names'][:3])
            if len(analysis['student_names']) > 3:
                students += f" and {len(analysis['student_names']) - 3} more"
            logger.info(f"  - {cluster_id}: {analysis['size']} students ({students})")
        
        # Save model
        logger.info("Saving model...")
        cluster_manager.save_model(str(output_dir))
        
        # Save training configuration
        training_config = {
            'task_folder': str(task_folder),
            'embedder_type': args.embedder_type,
            'model_name': args.sentence_model if args.embedder_type == "issues" else args.model_name,
            'algorithm': args.algorithm,
            'n_clusters': args.n_clusters,
            'random_state': args.random_state,
            'algorithm_kwargs': algorithm_kwargs,
            'embedder_kwargs': embedder_kwargs,
            'java_only': args.java_only,
            'min_java_files': args.min_java_files,
            'submission_count': len(submissions),
            'task_statistics': stats
        }
        
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 CLUSTERING TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Task: {task_folder.name}")
        print(f"🤖 Embedder: {args.embedder_type} ({args.model_name})")
        print(f"🔍 Algorithm: {args.algorithm}")
        print(f"📊 Submissions: {len(submissions)}")
        print(f"🎯 Clusters: {result.n_clusters}")
        print(f"📈 Silhouette Score: {result.silhouette_score:.3f}")
        
        print(f"\n📂 Model saved to: {output_dir}")
        print(f"📋 Config saved to: {config_path}")
        
        print("\n🔧 Usage Examples:")
        print("# Evaluate the model:")
        print(f"python src/cluster/scripts/evaluate_clustering.py --model-dir {output_dir}")
        
        print("\n# Use in code:")
        print(f"from src.cluster.clustering import ClusterManager")
        print(f"cluster_manager = ClusterManager(embedder)")
        print(f"cluster_manager.load_model('{output_dir}')")
        print(f"cluster_id = cluster_manager.predict(new_submission)")
        
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 