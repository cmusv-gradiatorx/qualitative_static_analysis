#!/usr/bin/env python3
"""
Build FAISS Index Script

This script builds the FAISS index from historical submission data.
Run this script to initialize the historical context system.

Usage:
    python src/faiss/build_index.py [options]

Author: Auto-generated
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from faiss.processor import SubmissionProcessor
from faiss.embedder import HybridCodeEmbedder
from faiss.faiss_manager import FAISSManager
from faiss.historical_context import HistoricalContextProvider
from utils.logger import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FAISS index for historical context")
    
    parser.add_argument(
        "--zip-path",
        type=str,
        default="src/faiss/data/GildedRoseKata_submissions.zip",
        help="Path to submissions ZIP file"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="src/faiss/index",
        help="Output directory for FAISS index"
    )
    
    parser.add_argument(
        "--assignment-id",
        type=str,
        default=None,
        help="Assignment ID (inferred from ZIP filename if not provided)"
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
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (quick validation)"
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
    
    logger.info("Starting FAISS index build process")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate requirements
    if not validate_requirements():
        sys.exit(1)
    
    # Validate input file
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        sys.exit(1)
    
    try:
        # Step 1: Process submissions
        logger.info(f"Processing submissions from {zip_path}")
        processor = SubmissionProcessor(str(zip_path), args.assignment_id)
        submissions = processor.extract_submissions()
        
        if not submissions:
            logger.error("No submissions found in ZIP file")
            sys.exit(1)
        
        logger.info(f"Extracted {len(submissions)} submissions")
        
        # Print submission statistics
        stats = processor.get_submission_statistics()
        logger.info(f"Submission statistics: {stats}")
        
        # Filter submissions if requested
        if args.min_score is not None or args.has_feedback:
            logger.info("Applying filters to submissions")
            filtered_submissions = processor.filter_submissions(
                min_score=args.min_score,
                has_feedback=args.has_feedback,
                min_files=1
            )
            submissions = filtered_submissions
            logger.info(f"After filtering: {len(submissions)} submissions")
        
        if args.test and len(submissions) > 5:
            logger.info("Test mode: Using only first 5 submissions")
            submissions = submissions[:5]
        
        # Step 2: Initialize embedder
        logger.info(f"Initializing embedder with model: {args.model_name}")
        embedder = HybridCodeEmbedder(
            model_name=args.model_name,
            device=args.device
        )
        
        # Step 3: Initialize FAISS manager
        logger.info(f"Initializing FAISS manager with index type: {args.index_type}")
        faiss_manager = FAISSManager(index_type=args.index_type)
        
        # Step 4: Build index
        logger.info("Building FAISS index...")
        build_stats = faiss_manager.build_index(
            submissions=submissions,
            embedder=embedder,
            save_path=args.output_path
        )
        
        logger.info("Index build completed successfully!")
        logger.info(f"Build statistics: {json.dumps(build_stats, indent=2)}")
        
        # Step 5: Save embedder scalers
        scaler_path = Path(args.output_path) / "scalers.pkl"
        embedder.save_scalers(str(scaler_path))
        logger.info(f"Saved embedder scalers to {scaler_path}")
        
        # Step 6: Test the index
        logger.info("Testing the index...")
        test_results = test_index(faiss_manager, embedder, submissions[:3])
        logger.info(f"Test results: {test_results}")
        
        # Step 7: Save additional metadata
        metadata_path = Path(args.output_path) / "build_metadata.json"
        metadata = {
            'build_args': vars(args),
            'build_stats': build_stats,
            'submission_stats': stats,
            'test_results': test_results,
            'embedding_info': embedder.get_embedding_info()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved build metadata to {metadata_path}")
        
        print("\n" + "="*60)
        print("✅ FAISS Index Build Complete!")
        print(f"📁 Index saved to: {args.output_path}")
        print(f"📊 Submissions indexed: {build_stats['valid_embeddings']}")
        print(f"🎯 Assignment ID: {build_stats['assignments']}")
        print(f"📏 Embedding dimension: {build_stats['embedding_dimension']}")
        print("="*60)
        
        # Provide usage example
        print("\n📖 Usage example:")
        print(f"""
from src.faiss import FAISSManager, HybridCodeEmbedder, HistoricalContextProvider

# Load the index
faiss_manager = FAISSManager()
faiss_manager.load_index("{args.output_path}")

# Load embedder
embedder = HybridCodeEmbedder()
embedder.load_scalers("{scaler_path}")

# Create context provider
context = HistoricalContextProvider(faiss_manager, embedder)

# Use in your autograder!
""")
        
    except Exception as e:
        logger.error(f"Failed to build index: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def test_index(faiss_manager: FAISSManager, embedder: HybridCodeEmbedder, 
              test_submissions: list) -> Dict[str, Any]:
    """Test the built index with sample queries"""
    logger = get_logger(__name__)
    
    if not test_submissions:
        return {'status': 'no_test_data'}
    
    try:
        test_results = {
            'status': 'success',
            'tests_performed': len(test_submissions),
            'search_results': []
        }
        
        for i, submission in enumerate(test_submissions):
            logger.info(f"Testing search with submission {i+1}: {submission.file_name}")
            
            # Search for similar submissions
            similar = faiss_manager.search_by_code_similarity(
                code_files=submission.code_files,
                embedder=embedder,
                assignment_id=submission.assignment_id,
                top_k=3
            )
            
            search_result = {
                'query_submission': submission.file_name,
                'similar_count': len(similar),
                'top_similarity': similar[0]['similarity_score'] if similar else 0.0,
                'avg_similarity': sum(s['similarity_score'] for s in similar) / len(similar) if similar else 0.0
            }
            
            test_results['search_results'].append(search_result)
            logger.info(f"Found {len(similar)} similar submissions, top similarity: {search_result['top_similarity']:.3f}")
        
        # Calculate overall statistics
        if test_results['search_results']:
            similarities = [r['top_similarity'] for r in test_results['search_results']]
            test_results['overall_stats'] = {
                'avg_top_similarity': sum(similarities) / len(similarities),
                'min_similarity': min(similarities),
                'max_similarity': max(similarities)
            }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error testing index: {str(e)}")
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    main() 