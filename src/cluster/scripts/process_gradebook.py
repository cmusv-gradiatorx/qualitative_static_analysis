#!/usr/bin/env python3
"""
Process Gradebook Script

Processes student gradebook CSV to create reference clusters based on Task 1 scores.
Creates task1_SOLID_reference_clusters.csv for evaluation purposes.

Usage:
    python src/cluster/scripts/process_gradebook.py \
        --input src/faiss/grade_mapping/2025-06-12T1929_Grades-18664-SV.csv \
        --output src/cluster/metrics/task1_SOLID_reference_clusters.csv

Author: Auto-generated
"""

import argparse
import sys
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from config.env
config_path = Path(__file__).parent.parent.parent.parent / 'config.env'
if config_path.exists():
    load_dotenv(config_path)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.logger import setup_logger, get_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process gradebook CSV to create reference clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input gradebook CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="src/cluster/metrics/task1_SOLID_reference_clusters.csv",
        help="Path to output reference clusters CSV file"
    )
    
    parser.add_argument(
        "--task-column",
        type=str,
        default="Task 1: Refactoring to Adhere to SOLID Principles (758617)",
        help="Name of the task column in CSV"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def convert_name_format(full_name: str) -> str:
    """
    Convert 'Last, First' format to 'lastfirst' format to match submission processing.
    
    Args:
        full_name: Name in 'Last, First' format
        
    Returns:
        Name in 'lastfirst' format (lowercase, no special characters)
    """
    # Handle format like "Cai, Tianjun"
    if ',' in full_name:
        last_name, first_name = full_name.split(',', 1)
        last_name = last_name.strip()
        first_name = first_name.strip()
    else:
        # Fallback: assume "First Last" format
        parts = full_name.strip().split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = ' '.join(parts[1:])
        else:
            first_name = full_name
            last_name = ""
    
    # Create the format used by submission processor: lastnamefirstname (not firstname+lastname)
    combined_name = f"{last_name}{first_name}"
    
    # Remove special characters and convert to lowercase (matching submission processor logic)
    cleaned_name = re.sub(r'[^a-zA-Z0-9]', '', combined_name).lower()
    
    return cleaned_name


def parse_score(score_str: str) -> float:
    """
    Parse score from string, handling various formats.
    
    Args:
        score_str: Score as string
        
    Returns:
        Score as float, -1.0 if invalid
    """
    if not score_str or score_str.strip() in ['', 'N/A', 'null', '-']:
        return -1.0
    
    try:
        return float(score_str.strip())
    except ValueError:
        return -1.0


def create_score_clusters(scores: List[float]) -> Dict[float, int]:
    """
    Create cluster mapping from unique scores.
    
    Args:
        scores: List of unique scores
        
    Returns:
        Dictionary mapping score to cluster ID
    """
    # Sort scores in descending order (highest score gets cluster 0)
    unique_scores = sorted(set(s for s in scores if s >= 0), reverse=True)
    
    # Create mapping from score to cluster ID
    score_to_cluster = {}
    for i, score in enumerate(unique_scores):
        score_to_cluster[score] = i
    
    # Handle invalid scores (assign to special cluster)
    score_to_cluster[-1.0] = len(unique_scores)  # Invalid scores get highest cluster ID
    
    return score_to_cluster


def process_gradebook(input_path: Path, task_column: str, logger) -> List[Tuple[str, float, int]]:
    """
    Process gradebook CSV file.
    
    Args:
        input_path: Path to input CSV file
        task_column: Name of the task column
        logger: Logger instance
        
    Returns:
        List of tuples (student_name, score, cluster_id)
    """
    logger.info(f"Processing gradebook: {input_path}")
    
    students_data = []
    
    with open(input_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Check if required columns exist
        if 'Student' not in reader.fieldnames:
            raise ValueError("'Student' column not found in CSV")
        if task_column not in reader.fieldnames:
            raise ValueError(f"Task column '{task_column}' not found in CSV")
        
        logger.info(f"Found columns: {reader.fieldnames}")
        
        for row in reader:
            student_name_raw = row['Student'].strip()
            task_score_raw = row[task_column].strip()
            
            # Skip empty rows and header rows
            if not student_name_raw or student_name_raw in ['Student', 'Points Possible']:
                continue
            
            # Convert name format
            student_name = convert_name_format(student_name_raw)
            
            # Parse score
            score = parse_score(task_score_raw)
            
            students_data.append((student_name_raw, student_name, score))
            logger.debug(f"Processed: {student_name_raw} -> {student_name}, score: {score}")
    
    logger.info(f"Processed {len(students_data)} students")
    
    # Create clusters based on scores
    scores = [data[2] for data in students_data]
    score_to_cluster = create_score_clusters(scores)
    
    logger.info(f"Created {len(score_to_cluster)} score-based clusters:")
    for score, cluster_id in sorted(score_to_cluster.items(), reverse=True):
        count = sum(1 for s in scores if s == score)
        if score >= 0:
            logger.info(f"  Cluster {cluster_id}: Score {score:.2f} ({count} students)")
        else:
            logger.info(f"  Cluster {cluster_id}: Invalid scores ({count} students)")
    
    # Create final result with cluster assignments
    result = []
    for student_name_raw, student_name, score in students_data:
        cluster_id = score_to_cluster[score]
        result.append((student_name, score, cluster_id))
    
    return result


def save_reference_clusters(output_path: Path, student_data: List[Tuple[str, float, int]], logger):
    """
    Save reference clusters to CSV file.
    
    Args:
        output_path: Path to output CSV file
        student_data: List of (student_name, score, cluster_id) tuples
        logger: Logger instance
    """
    logger.info(f"Saving reference clusters to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['student_name', 'task1_score', 'reference_cluster'])
        
        # Write student data sorted by cluster then by name
        sorted_data = sorted(student_data, key=lambda x: (x[2], x[0]))
        for student_name, score, cluster_id in sorted_data:
            writer.writerow([student_name, score, cluster_id])
    
    logger.info(f"Saved {len(student_data)} student records")


def main():
    """Main processing function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting gradebook processing")
    
    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Process gradebook
        student_data = process_gradebook(input_path, args.task_column, logger)
        
        if not student_data:
            logger.error("No student data found in gradebook")
            sys.exit(1)
        
        # Save reference clusters
        output_path = Path(args.output)
        save_reference_clusters(output_path, student_data, logger)
        
        # Print summary
        cluster_counts = defaultdict(int)
        score_distribution = defaultdict(int)
        
        for _, score, cluster_id in student_data:
            cluster_counts[cluster_id] += 1
            if score >= 0:
                score_distribution[score] += 1
        
        print("\n" + "="*60)
        print("📊 REFERENCE CLUSTERS GENERATED")
        print("="*60)
        print(f"📁 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        print(f"👥 Total students: {len(student_data)}")
        print(f"🎯 Total clusters: {len(cluster_counts)}")
        
        print(f"\n📈 CLUSTER DISTRIBUTION:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / len(student_data)) * 100
            print(f"  Cluster {cluster_id}: {count} students ({percentage:.1f}%)")
        
        print(f"\n🏆 SCORE DISTRIBUTION:")
        for score in sorted(score_distribution.keys(), reverse=True):
            if score >= 0:
                count = score_distribution[score]
                print(f"  Score {score:.1f}: {count} students")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 