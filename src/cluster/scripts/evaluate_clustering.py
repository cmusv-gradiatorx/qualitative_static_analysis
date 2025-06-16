#!/usr/bin/env python3
"""
Evaluate Clustering Model Script

Evaluates trained clustering models on student submissions.
Provides detailed analysis and visualization of clustering results.
Can compare against reference clusters based on actual scores.

Usage:
    python src/cluster/scripts/evaluate_clustering.py \
        --model-dir models/task1_java_kmeans \
        --task-folder data/task1_SOLID \
        --output-dir evaluation_results \
        --reference-clusters src/cluster/metrics/task1_SOLID_reference_clusters.csv

Author: Auto-generated
"""

import argparse
import sys
import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import Counter, defaultdict
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


def create_output_directory(model_dir: Path, config: Dict[str, Any]) -> Path:
    """Create organized output directory with auto-generated name"""
    # Extract task name from task folder
    task_folder = Path(config['task_folder'])
    task_name = task_folder.name
    
    # Extract embedder, algorithm, and model info
    embedder_type = config['embedder_type']
    algorithm = config['algorithm']
    model_name = config['model_name'].replace(':', '_')
    
    # Include number of clusters in the filename
    if algorithm == "dbscan":
        algorithm_kwargs = config.get('algorithm_kwargs', {})
        eps = algorithm_kwargs.get('eps', 0.5)
        min_samples = algorithm_kwargs.get('min_samples', 2)
        cluster_suffix = f"eps{eps}_min{min_samples}"
    else:
        n_clusters = config['n_clusters']
        cluster_suffix = f"{n_clusters}clusters"
    
    # Create output directory name
    output_name = f"{task_name}_{embedder_type}_{algorithm}_{model_name}_{cluster_suffix}_evaluation"
    output_dir = Path("cluster_evaluation_results") / output_name
    
    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate clustering model on student submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a trained model with auto-generated output directory
  python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans

  # Evaluate with custom reference clusters
  python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \\
      --reference-clusters src/cluster/metrics/task1_SOLID_reference_clusters.csv

  # Evaluate on different task folder than training
  python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \\
      --task-folder src/faiss/data/task2_PreparingRefactoring

  # Evaluate with custom output directory
  python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \\
      --output-dir custom_evaluation_results
        """
    )
    
    # Required arguments
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to directory containing trained model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--task-folder",
        type=str,
        default=None,
        help="Path to task folder for re-evaluation (uses original if not provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--reference-clusters",
        type=str,
        default="src/cluster/metrics/task1_SOLID_reference_clusters.csv",
        help="Path to reference clusters CSV file for ground truth comparison"
    )
    
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    return parser.parse_args()


def load_training_config(model_dir: Path) -> Dict[str, Any]:
    """Load training configuration"""
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_reference_clusters(reference_path: Path, logger) -> Dict[str, int]:
    """
    Load reference clusters from CSV file.
    
    Args:
        reference_path: Path to reference clusters CSV
        logger: Logger instance
        
    Returns:
        Dictionary mapping student_name to reference cluster ID
    """
    if not reference_path.exists():
        logger.warning(f"Reference clusters file not found: {reference_path}")
        return {}
    
    reference_clusters = {}
    
    with open(reference_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            student_name = row['student_name'].strip()
            cluster_id = int(row['reference_cluster'])
            reference_clusters[student_name] = cluster_id
    
    logger.info(f"Loaded reference clusters for {len(reference_clusters)} students")
    return reference_clusters


def find_best_cluster_mapping(predicted_labels: np.ndarray, 
                             reference_labels: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Find the best mapping between predicted and reference clusters.
    
    Args:
        predicted_labels: Predicted cluster labels
        reference_labels: Reference (ground truth) cluster labels
        
    Returns:
        Tuple of (cluster_mapping, num_correctly_assigned)
    """
    from collections import defaultdict
    
    # Count student assignments for each (predicted, reference) pair
    assignment_counts = defaultdict(int)
    
    for pred_cluster, ref_cluster in zip(predicted_labels, reference_labels):
        assignment_counts[(pred_cluster, ref_cluster)] += 1
    
    # Find the best mapping using a greedy approach
    # For each predicted cluster, assign it to the reference cluster with most overlap
    pred_clusters = set(predicted_labels)
    ref_clusters = set(reference_labels)
    
    cluster_mapping = {}
    used_ref_clusters = set()
    
    # Sort predicted clusters by size (largest first)
    pred_cluster_sizes = Counter(predicted_labels)
    sorted_pred_clusters = sorted(pred_clusters, key=lambda x: pred_cluster_sizes[x], reverse=True)
    
    for pred_cluster in sorted_pred_clusters:
        # Find the reference cluster with maximum overlap for this predicted cluster
        best_ref_cluster = None
        best_count = 0
        
        for ref_cluster in ref_clusters:
            if ref_cluster in used_ref_clusters:
                continue
            count = assignment_counts[(pred_cluster, ref_cluster)]
            if count > best_count:
                best_count = count
                best_ref_cluster = ref_cluster
        
        if best_ref_cluster is not None:
            cluster_mapping[pred_cluster] = best_ref_cluster
            used_ref_clusters.add(best_ref_cluster)
        else:
            # No available reference cluster, assign to -1 (unmatched)
            cluster_mapping[pred_cluster] = -1
    
    # Count correctly assigned students using this mapping
    correctly_assigned = 0
    for pred_cluster, ref_cluster in zip(predicted_labels, reference_labels):
        if cluster_mapping.get(pred_cluster, -1) == ref_cluster:
            correctly_assigned += 1
    
    return cluster_mapping, correctly_assigned


def calculate_clustering_accuracy(cluster_labels: np.ndarray, 
                                reference_clusters: Dict[str, int],
                                student_names: List[str],
                                logger) -> Dict[str, Any]:
    """
    Calculate clustering accuracy against reference clusters.
    
    Args:
        cluster_labels: Predicted cluster labels
        reference_clusters: Dictionary mapping student names to reference clusters
        student_names: List of student names corresponding to cluster_labels
        logger: Logger instance
        
    Returns:
        Dictionary containing accuracy metrics
    """
    # Filter to only students present in both predicted and reference
    common_students = []
    predicted_labels = []
    reference_labels = []
    
    for i, student_name in enumerate(student_names):
        if student_name in reference_clusters:
            common_students.append(student_name)
            predicted_labels.append(cluster_labels[i])
            reference_labels.append(reference_clusters[student_name])
    
    if not common_students:
        logger.warning("No common students found between predictions and reference")
        return {
            'accuracy': 0.0,
            'num_common_students': 0,
            'num_correctly_assigned': 0,
            'num_misassigned': 0,
            'cluster_mapping': {},
            'misassigned_students': []
        }
    
    predicted_labels = np.array(predicted_labels)
    reference_labels = np.array(reference_labels)
    
    # Find best cluster mapping
    cluster_mapping, correctly_assigned = find_best_cluster_mapping(predicted_labels, reference_labels)
    
    # Calculate accuracy
    accuracy = correctly_assigned / len(common_students) if common_students else 0.0
    
    # Find misassigned students
    misassigned_students = []
    for i, (student_name, pred_cluster, ref_cluster) in enumerate(
        zip(common_students, predicted_labels, reference_labels)
    ):
        mapped_cluster = cluster_mapping.get(pred_cluster, -1)
        if mapped_cluster != ref_cluster:
            misassigned_students.append({
                'student_name': student_name,
                'predicted_cluster': int(pred_cluster),
                'mapped_cluster': mapped_cluster,
                'reference_cluster': int(ref_cluster)
            })
    
    return {
        'accuracy': accuracy,
        'num_common_students': len(common_students),
        'num_correctly_assigned': correctly_assigned,
        'num_misassigned': len(misassigned_students),
        'cluster_mapping': {int(k): int(v) for k, v in cluster_mapping.items()},
        'misassigned_students': misassigned_students,
        'common_students': common_students
    }


def create_embedder_from_config(config: Dict[str, Any]) -> Any:
    """Create embedder from training configuration"""
    embedder_kwargs = config.get('embedder_kwargs', {})
    
    return EmbedderFactory.create_embedder(
        embedder_type=config['embedder_type'],
        model_name=config['model_name'],
        **embedder_kwargs
    )


def analyze_cluster_distribution(cluster_labels: np.ndarray, 
                               student_names: list) -> Dict[str, Any]:
    """Analyze cluster distribution"""
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    # Calculate distribution statistics
    total_students = len(cluster_labels)
    cluster_sizes = {int(cluster): int(count) for cluster, count in zip(unique_clusters, counts)}
    
    # Find largest and smallest clusters
    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
    smallest_cluster = min(cluster_sizes, key=cluster_sizes.get)
    
    # Calculate statistics
    mean_size = np.mean(counts)
    std_size = np.std(counts)
    size_coefficient_of_variation = std_size / mean_size if mean_size > 0 else 0
    
    return {
        'n_clusters': len(unique_clusters),
        'total_students': total_students,
        'cluster_sizes': cluster_sizes,
        'largest_cluster': {
            'id': largest_cluster,
            'size': cluster_sizes[largest_cluster],
            'percentage': (cluster_sizes[largest_cluster] / total_students) * 100
        },
        'smallest_cluster': {
            'id': smallest_cluster,
            'size': cluster_sizes[smallest_cluster],
            'percentage': (cluster_sizes[smallest_cluster] / total_students) * 100
        },
        'mean_cluster_size': float(mean_size),
        'std_cluster_size': float(std_size),
        'size_coefficient_of_variation': float(size_coefficient_of_variation),
        'size_balance_score': 1.0 - size_coefficient_of_variation  # Higher is better
    }


def generate_markdown_summary(evaluation_report: Dict[str, Any], 
                             detailed_report: List[Dict[str, Any]],
                             output_dir: Path) -> str:
    """
    Generate a markdown summary of evaluation results.
    
    Args:
        evaluation_report: Complete evaluation report
        detailed_report: Detailed cluster breakdown
        output_dir: Output directory for saving files
        
    Returns:
        Markdown content as string
    """
    model_info = evaluation_report['model_info']
    metrics = evaluation_report['clustering_metrics']
    distribution = evaluation_report['distribution_analysis']
    accuracy = evaluation_report.get('accuracy_analysis', {})
    
    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Clustering Evaluation Results

**Generated:** {timestamp}  
**Model Directory:** `{model_info['model_dir']}`  
**Task:** {model_info['task_folder'].split('/')[-1]}

## 📊 Performance Overview

| Metric | Value | Status |
|--------|-------|--------|"""
    
    # Add performance metrics with status indicators
    silhouette = metrics['silhouette_score']
    silhouette_status = "🟢 Good" if silhouette > 0.3 else "🟡 Fair" if silhouette > 0.1 else "🔴 Poor"
    markdown += f"\n| **Silhouette Score** | {silhouette:.3f} | {silhouette_status} |"
    
    calinski = metrics['calinski_harabasz_score']
    calinski_status = "🟢 Good" if calinski > 20 else "🟡 Fair" if calinski > 5 else "🔴 Poor"
    markdown += f"\n| **Calinski-Harabasz** | {calinski:.1f} | {calinski_status} |"
    
    balance = distribution['size_balance_score']
    balance_status = "🟢 Balanced" if balance > 0.7 else "🟡 Moderate" if balance > 0.3 else "🔴 Imbalanced"
    markdown += f"\n| **Balance Score** | {balance:.3f} | {balance_status} |"
    
    if accuracy:
        acc_score = accuracy['accuracy']
        acc_status = "🟢 Excellent" if acc_score > 0.8 else "🟡 Good" if acc_score > 0.6 else "🟠 Fair" if acc_score > 0.4 else "🔴 Poor"
        markdown += f"\n| **Accuracy vs Scores** | {acc_score:.1%} | {acc_status} |"
    
    markdown += f"""

## 🎯 Model Configuration

- **Embedder:** {model_info['embedder_type']}
- **Model:** {model_info['model_name']}
- **Algorithm:** {model_info['algorithm']}
- **Clusters Requested:** {model_info['n_clusters_requested']}
- **Clusters Found:** {metrics['n_clusters']}
- **Total Submissions:** {evaluation_report['dataset_info']['total_submissions']}

"""
    
    # Add accuracy analysis if available
    if accuracy:
        markdown += f"""## 📈 Accuracy Analysis

- **Students Evaluated:** {accuracy['num_common_students']}
- **Correctly Assigned:** {accuracy['num_correctly_assigned']} students
- **Misassigned:** {accuracy['num_misassigned']} students
- **Overall Accuracy:** {accuracy['accuracy']:.1%}

### Cluster Mapping
| Predicted Cluster | → | Reference Cluster |
|-------------------|---|-------------------|"""
        
        for pred_cluster, ref_cluster in sorted(accuracy['cluster_mapping'].items()):
            ref_display = f"Cluster {ref_cluster}" if ref_cluster != -1 else "Unmatched"
            markdown += f"\n| Cluster {pred_cluster} | → | {ref_display} |"
        
        markdown += "\n\n"
    
    # Add cluster breakdown
    markdown += f"""## 🏆 Cluster Breakdown

| Cluster | Size | Percentage | Top Students |
|---------|------|------------|--------------|"""
    
    for cluster_info in detailed_report[:8]:  # Show top 8 clusters
        cluster_id = cluster_info['cluster_id'].replace('cluster_', '')
        size = cluster_info['size']
        percentage = cluster_info['percentage']
        
        # Get top 3 students
        students = cluster_info['students']
        if len(students) <= 3:
            students_str = ", ".join(students)
        else:
            students_str = ", ".join(students[:3]) + f" +{len(students)-3} more"
        
        markdown += f"\n| **{cluster_id}** | {size} | {percentage:.1f}% | {students_str} |"
    
    if len(detailed_report) > 8:
        remaining = len(detailed_report) - 8
        markdown += f"\n| ... | ... | ... | *{remaining} more clusters* |"
    
    markdown += f"""

## 📊 Distribution Statistics

- **Largest Cluster:** {distribution['largest_cluster']['size']} students ({distribution['largest_cluster']['percentage']:.1f}%)
- **Smallest Cluster:** {distribution['smallest_cluster']['size']} students ({distribution['smallest_cluster']['percentage']:.1f}%)
- **Mean Cluster Size:** {distribution['mean_cluster_size']:.1f} students
- **Size Variation:** {distribution['size_coefficient_of_variation']:.3f}

## 📁 Files Generated

- [`evaluation_report.json`]({output_dir.name}/evaluation_report.json) - Complete evaluation metrics
- [`detailed_cluster_report.json`]({output_dir.name}/detailed_cluster_report.json) - Detailed cluster analysis"""
    
    if accuracy and accuracy.get('misassigned_students'):
        markdown += f"\n- [`misassigned_students.csv`]({output_dir.name}/misassigned_students.csv) - List of misassigned students"
    
    markdown += f"""

## 🔧 Quick Commands

```bash
# View detailed report
cat {output_dir}/evaluation_report.json | jq

# Check misassigned students"""
    
    if accuracy and accuracy.get('misassigned_students'):
        markdown += f"\nhead {output_dir}/misassigned_students.csv"
    
    markdown += f"""

# Compare with other evaluations
ls cluster_evaluation_results/
```

---
*Generated by the Clustering Evaluation System*
"""
    
    return markdown


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level, args.log_file)
    logger = get_logger(__name__)
    
    logger.info("Starting clustering evaluation")
    
    try:
        # Validate model directory
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
        
        # Load training configuration
        logger.info("Loading training configuration...")
        config = load_training_config(model_dir)
        
        # Load reference clusters
        reference_clusters = {}
        reference_path = Path(args.reference_clusters)
        if reference_path.exists():
            logger.info("Loading reference clusters...")
            reference_clusters = load_reference_clusters(reference_path, logger)
        else:
            logger.warning(f"Reference clusters file not found: {reference_path}")
        
        # Create output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using custom output directory: {output_dir}")
        else:
            output_dir = create_output_directory(model_dir, config)
            logger.info(f"Auto-generated output directory: {output_dir}")
        
        # Create embedder
        logger.info("Creating embedder...")
        embedder = create_embedder_from_config(config)
        
        # Create cluster manager and load model
        logger.info("Loading trained model...")
        cluster_manager = ClusterManager(embedder)
        cluster_manager.load_model(str(model_dir))
        
        # Determine task folder
        if args.task_folder:
            task_folder = Path(args.task_folder)
        else:
            task_folder = Path(config['task_folder'])
        
        logger.info(f"Evaluating on task folder: {task_folder}")
        
        # Process submissions
        logger.info("Processing submissions...")
        processor = SubmissionProcessor(str(task_folder))
        submissions = processor.extract_submissions(java_only=config.get('java_only', False))
        
        if config.get('min_java_files', 1) > 1:
            submissions = processor.filter_submissions(
                submissions, 
                min_java_files=config['min_java_files']
            )
        
        logger.info(f"Evaluating {len(submissions)} submissions")
        
        # Get evaluation metrics
        logger.info("Calculating evaluation metrics...")
        metrics = cluster_manager.evaluate_clustering()
        
        # Get cluster analysis
        cluster_analysis = cluster_manager.get_cluster_analysis(submissions)
        
        # Analyze cluster distribution
        student_names = [sub.student_name for sub in submissions]
        distribution_analysis = analyze_cluster_distribution(
            cluster_manager.clustering_result.cluster_labels,
            student_names
        )
        
        # Calculate accuracy against reference clusters
        accuracy_analysis = {}
        if reference_clusters:
            logger.info("Calculating accuracy against reference clusters...")
            accuracy_analysis = calculate_clustering_accuracy(
                cluster_manager.clustering_result.cluster_labels,
                reference_clusters,
                student_names,
                logger
            )
        
        # Create comprehensive evaluation report
        evaluation_report = {
            'model_info': {
                'model_dir': str(model_dir),
                'task_folder': str(task_folder),
                'embedder_type': config['embedder_type'],
                'model_name': config['model_name'],
                'algorithm': config['algorithm'],
                'n_clusters_requested': config['n_clusters'],
                'training_timestamp': config.get('fit_timestamp', 'unknown')
            },
            'dataset_info': {
                'total_submissions': len(submissions),
                'task_statistics': processor.get_task_statistics(submissions)
            },
            'clustering_metrics': metrics,
            'distribution_analysis': distribution_analysis,
            'cluster_analysis': cluster_analysis,
            'accuracy_analysis': accuracy_analysis,
            'model_performance': {
                'clusters_found': metrics['n_clusters'],
                'silhouette_score': metrics['silhouette_score'],
                'calinski_harabasz_score': metrics['calinski_harabasz_score'],
                'size_balance_score': distribution_analysis['size_balance_score']
            }
        }
        
        # Add accuracy metrics to model performance if available
        if accuracy_analysis:
            evaluation_report['model_performance'].update({
                'accuracy_vs_scores': accuracy_analysis['accuracy'],
                'num_misassigned': accuracy_analysis['num_misassigned'],
                'num_common_students': accuracy_analysis['num_common_students']
            })
        
        # Save evaluation report
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        # Create detailed cluster report
        detailed_report = []
        for cluster_id, analysis in cluster_analysis.items():
            cluster_info = {
                'cluster_id': cluster_id,
                'size': analysis['size'],
                'percentage': (analysis['size'] / len(submissions)) * 100,
                'students': analysis['student_names'],
                'statistics': {
                    'avg_distance_to_center': analysis['avg_distance_to_center'],
                    'max_distance_to_center': analysis['max_distance_to_center'],
                    'compactness': analysis['compactness'],
                    'avg_java_files': analysis['java_files_avg'],
                    'avg_total_files': analysis['total_files_avg']
                }
            }
            detailed_report.append(cluster_info)
        
        # Sort clusters by size
        detailed_report.sort(key=lambda x: x['size'], reverse=True)
        
        # Save detailed report
        detailed_path = output_dir / "detailed_cluster_report.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        # Save misassigned students report if available
        if accuracy_analysis and accuracy_analysis['misassigned_students']:
            misassigned_path = output_dir / "misassigned_students.csv"
            with open(misassigned_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['student_name', 'predicted_cluster', 'mapped_cluster', 'reference_cluster'])
                for student in accuracy_analysis['misassigned_students']:
                    writer.writerow([
                        student['student_name'],
                        student['predicted_cluster'],
                        student['mapped_cluster'],
                        student['reference_cluster']
                    ])
        
        # Print summary
        print("\n" + "="*80)
        print("📊 CLUSTERING EVALUATION RESULTS")
        print("="*80)
        print(f"📁 Task: {task_folder.name}")
        print(f"🤖 Model: {config['embedder_type']} + {config['algorithm']} ({config['model_name']})")
        print(f"📈 Submissions: {len(submissions)}")
        print(f"🎯 Clusters: {metrics['n_clusters']}")
        print(f"📊 Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"📈 Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f}")
        print(f"⚖️  Balance Score: {distribution_analysis['size_balance_score']:.3f}")
        
        # Print accuracy results if available
        if accuracy_analysis:
            print(f"\n🎯 ACCURACY VS ACTUAL SCORES:")
            print(f"✅ Correctly assigned: {accuracy_analysis['num_correctly_assigned']}/{accuracy_analysis['num_common_students']} students")
            print(f"🎯 Accuracy: {accuracy_analysis['accuracy']:.1%}")
            print(f"❌ Misassigned: {accuracy_analysis['num_misassigned']} students")
            
            if accuracy_analysis['cluster_mapping']:
                print(f"\n🔗 CLUSTER MAPPING (Predicted → Reference):")
                for pred_cluster, ref_cluster in sorted(accuracy_analysis['cluster_mapping'].items()):
                    if ref_cluster != -1:
                        print(f"  Cluster {pred_cluster} → Reference Cluster {ref_cluster}")
                    else:
                        print(f"  Cluster {pred_cluster} → Unmatched")
        
        print(f"\n🏆 CLUSTER BREAKDOWN:")
        for i, cluster_info in enumerate(detailed_report[:5]):  # Show top 5 clusters
            print(f"  {cluster_info['cluster_id']}: {cluster_info['size']} students ({cluster_info['percentage']:.1f}%)")
            if len(cluster_info['students']) <= 3:
                students_str = ", ".join(cluster_info['students'])
            else:
                students_str = ", ".join(cluster_info['students'][:3]) + f" + {len(cluster_info['students']) - 3} more"
            print(f"    👥 {students_str}")
        
        if len(detailed_report) > 5:
            print(f"    ... and {len(detailed_report) - 5} more clusters")
        
        print(f"\n📂 EVALUATION RESULTS SAVED:")
        print(f"📁 Directory: {output_dir}")
        print(f"📋 Main Report: {report_path.name}")
        print(f"📊 Detailed Report: {detailed_path.name}")
        print(f"📄 Quick Summary: evaluation_summary.md")
        if accuracy_analysis and accuracy_analysis['misassigned_students']:
            print(f"❌ Misassigned Students: misassigned_students.csv")
        
        print(f"\n🔧 USAGE EXAMPLES:")
        print(f"# View quick summary:")
        print(f"cat {output_dir}/evaluation_summary.md")
        print(f"# View main report:")
        print(f"cat {report_path}")
        print(f"# Analyze misassigned students:")
        if accuracy_analysis and accuracy_analysis['misassigned_students']:
            print(f"head {output_dir}/misassigned_students.csv")
        print(f"# Compare with other evaluations:")
        print(f"ls cluster_evaluation_results/")
        print("="*80)
        
        # Generate markdown summary
        markdown_summary = generate_markdown_summary(evaluation_report, detailed_report, output_dir)
        
        # Save markdown summary
        markdown_path = output_dir / "evaluation_summary.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_summary)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 