#!/usr/bin/env python3
"""
Build Assignment-Specific FAISS Indices Script

This script builds separate FAISS indices for each assignment (ZIP file) to improve
efficiency and provide better contextual results. Each assignment gets its own
dedicated FAISS index with enhanced similarity support.

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.faiss.processor import SubmissionProcessor
from src.faiss.embedder import HybridCodeEmbedder
from src.faiss.assignment_faiss_manager import AssignmentFAISSManager
from src.utils.logger import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build assignment-specific FAISS indices")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/faiss/data",
        help="Directory containing submission ZIP files"
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
        default="microsoft/codebert-base",
        help="Code embedding model name"
    )
    
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embeddings (cuda/cpu, auto if not specified)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum score to include submissions"
    )
    
    parser.add_argument(
        "--has-feedback",
        action="store_true",
        help="Only include submissions with feedback"
    )
    
    parser.add_argument(
        "--assignments",
        type=str,
        nargs="+",
        default=None,
        help="Specific assignment IDs to process (default: all)"
    )
    
    parser.add_argument(
        "--enhanced-similarity",
        action="store_true",
        default=True,
        help="Enable enhanced similarity calculation"
    )
    
    parser.add_argument(
        "--similarity-weights",
        type=float,
        nargs=4,
        default=[0.5, 0.3, 0.15, 0.05],
        metavar=("SEMANTIC", "STRUCTURAL", "PATTERN", "GRAPH"),
        help="Weights for similarity components: semantic structural pattern graph"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def find_zip_files(data_dir: str) -> Dict[str, str]:
    """Find all ZIP files and map them to assignment IDs"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}
    
    zip_files = list(data_path.glob("*.zip"))
    assignment_map = {}
    
    for zip_file in zip_files:
        # Extract assignment ID from filename (remove _submissions.zip suffix)
        assignment_id = zip_file.stem.replace("_submissions", "")
        assignment_map[assignment_id] = str(zip_file)
    
    return assignment_map


def group_submissions_by_assignment(all_submissions: List) -> Dict[str, List]:
    """Group submissions by assignment ID"""
    submissions_by_assignment = defaultdict(list)
    
    for submission in all_submissions:
        assignment_id = submission.assignment_id
        submissions_by_assignment[assignment_id].append(submission)
    
    return dict(submissions_by_assignment)


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
        import sklearn
    except ImportError:
        missing_packages.append("scikit-learn")
    
    try:
        import networkx
    except ImportError:
        missing_packages.append("networkx")
    
    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting assignment-specific FAISS index build process")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate requirements
    if not validate_requirements():
        sys.exit(1)
    
    # Find all ZIP files and their assignment mappings
    logger.info(f"Looking for ZIP files in directory: {args.data_dir}")
    assignment_zip_map = find_zip_files(args.data_dir)
    
    if not assignment_zip_map:
        logger.error(f"No ZIP files found in directory: {args.data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(assignment_zip_map)} assignments: {list(assignment_zip_map.keys())}")
    
    # Filter assignments if specified
    if args.assignments:
        filtered_map = {aid: path for aid, path in assignment_zip_map.items() 
                       if aid in args.assignments}
        if not filtered_map:
            logger.error(f"None of the specified assignments found: {args.assignments}")
            sys.exit(1)
        assignment_zip_map = filtered_map
        logger.info(f"Processing specified assignments: {list(assignment_zip_map.keys())}")
    
    try:
        # Process submissions from each ZIP file
        all_submissions = []
        assignment_stats = {}
        
        for assignment_id, zip_file in assignment_zip_map.items():
            logger.info(f"Processing submissions from {zip_file} (Assignment: {assignment_id})")
            processor = SubmissionProcessor(zip_file)
            submissions = processor.extract_submissions()
            
            if not submissions:
                logger.warning(f"No submissions found in ZIP file: {zip_file}")
                continue
            
            logger.info(f"Extracted {len(submissions)} submissions from {assignment_id}")
            
            # Get statistics for this assignment
            stats = processor.get_submission_statistics()
            assignment_stats[assignment_id] = {
                'zip_file': Path(zip_file).name,
                'extraction_stats': stats,
                'raw_submission_count': len(submissions)
            }
            
            # Filter submissions if requested
            if args.min_score is not None or args.has_feedback:
                logger.info(f"Applying filters to submissions from {assignment_id}")
                filtered_submissions = processor.filter_submissions(
                    min_score=args.min_score,
                    has_feedback=args.has_feedback,
                    min_files=1
                )
                submissions = filtered_submissions
                logger.info(f"After filtering {assignment_id}: {len(submissions)} submissions")
                assignment_stats[assignment_id]['filtered_submission_count'] = len(submissions)
            
            all_submissions.extend(submissions)
        
        if not all_submissions:
            logger.error("No submissions found in any ZIP files after processing")
            sys.exit(1)
        
        logger.info(f"Total submissions collected: {len(all_submissions)}")
        
        # Group submissions by assignment
        submissions_by_assignment = group_submissions_by_assignment(all_submissions)
        logger.info(f"Grouped submissions into {len(submissions_by_assignment)} assignments")
        
        for assignment_id, submissions in submissions_by_assignment.items():
            logger.info(f"Assignment '{assignment_id}': {len(submissions)} submissions")
        
        # Initialize embedder with enhanced similarity
        logger.info(f"Initializing embedder with model: {args.model_name}")
        embedder = HybridCodeEmbedder(
            model_name=args.model_name,
            device=args.device
        )
        
        # Configure similarity weights
        if args.enhanced_similarity:
            embedder.adjust_similarity_weights(*args.similarity_weights)
            logger.info(f"Enhanced similarity enabled with weights: {args.similarity_weights}")
        
        # Initialize assignment FAISS manager
        logger.info(f"Initializing Assignment FAISS manager")
        assignment_manager = AssignmentFAISSManager(
            base_index_path=args.output_path,
            index_type=args.index_type,
            use_enhanced_similarity=args.enhanced_similarity
        )
        
        # Build assignment-specific indices
        logger.info("Building assignment-specific FAISS indices...")
        build_stats = assignment_manager.build_assignment_indices(
            submissions_by_assignment=submissions_by_assignment,
            embedder=embedder
        )
        
        logger.info("Assignment-specific indices build completed successfully!")
        
        # Save embedder scalers
        scaler_path = Path(args.output_path) / "embedder_scalers.pkl"
        embedder.save_scalers(str(scaler_path))
        logger.info(f"Saved embedder scalers to {scaler_path}")
        
        # Save comprehensive metadata
        metadata_path = Path(args.output_path) / "build_metadata.json"
        comprehensive_metadata = {
            'build_args': vars(args),
            'assignment_stats': assignment_stats,
            'build_stats': build_stats,
            'embedding_info': embedder.get_embedding_info(),
            'assignment_zip_mapping': assignment_zip_map,
            'total_submissions': len(all_submissions),
            'assignments_built': list(build_stats.keys()),
            'build_type': 'assignment_specific',
            'enhanced_similarity_enabled': args.enhanced_similarity,
            'similarity_weights': {
                'semantic': args.similarity_weights[0],
                'structural': args.similarity_weights[1], 
                'pattern': args.similarity_weights[2],
                'graph': args.similarity_weights[3]
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)
        logger.info(f"Saved comprehensive build metadata to {metadata_path}")
        
        # Final summary
        print("\n" + "="*70)
        print("✅ Assignment-Specific FAISS Indices Build Complete!")
        print(f"📁 Indices saved to: {args.output_path}")
        print(f"🎯 Assignments processed: {len(build_stats)}")
        
        total_indexed = sum(stats['valid_embeddings'] for stats in build_stats.values())
        print(f"📊 Total submissions indexed: {total_indexed}")
        
        print(f"🔧 Enhanced similarity: {'Enabled' if args.enhanced_similarity else 'Disabled'}")
        if args.enhanced_similarity:
            print(f"   Weights: Semantic={args.similarity_weights[0]}, Structural={args.similarity_weights[1]}")
            print(f"           Pattern={args.similarity_weights[2]}, Graph={args.similarity_weights[3]}")
        
        print(f"📏 Embedding dimension: {embedder.get_embedding_info()['total_dim']}")
        print(f"📦 Index type: {args.index_type}")
        
        print("\nPer-Assignment Statistics:")
        for assignment_id, stats in build_stats.items():
            print(f"  • {assignment_id}: {stats['valid_embeddings']} submissions indexed")
        
        print("="*70)
        
        # Usage examples
        print("\n📖 Usage Examples:")
        print("  # Search within specific assignment:")
        print(f"  manager = AssignmentFAISSManager('{args.output_path}')")
        print("  manager.load_assignment_indices()")
        print("  results = manager.search_similar_in_assignment('assignment_id', query_embedding)")
        print("\n  # Search across multiple assignments:")
        print("  results = manager.search_across_assignments(query_embedding, assignment_ids=['id1', 'id2'])")
        
    except Exception as e:
        logger.error(f"Failed to build assignment indices: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 