 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Clustering vs FAISS Performance

This script evaluates the clustering-based grading system and compares it with the
traditional FAISS similarity search approach for providing grading context.

Usage:
    python src/faiss/clustering/evaluate_clustering.py --assignment task4_GildedRoseKata

Author: Auto-generated
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load environment variables
try:
    from dotenv import load_dotenv
    config_path = Path(__file__).parent.parent.parent.parent / "config.env"
    if config_path.exists():
        load_dotenv(config_path)
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.faiss.clustering.cluster_manager import ClusteringManager
from src.faiss.assignment_faiss_manager import AssignmentFAISSManager
from src.faiss.processor import MultiFolderProcessor
from src.faiss.embedder import create_java_embedder
from src.utils.logger import get_logger, setup_logger


class ClusteringEvaluator:
    """Evaluate clustering performance and compare with FAISS"""
    
    def __init__(self, assignment_id: str, clustering_manager: ClusteringManager, 
                 faiss_manager: AssignmentFAISSManager = None, embedder=None):
        """
        Initialize evaluator.
        
        Args:
            assignment_id: Assignment identifier
            clustering_manager: Trained clustering manager
            faiss_manager: Optional FAISS manager for comparison
            embedder: Code embedder
        """
        self.assignment_id = assignment_id
        self.clustering_manager = clustering_manager
        self.faiss_manager = faiss_manager
        self.embedder = embedder
        self.logger = get_logger(__name__)
    
    def evaluate_score_prediction(self, test_split: float = 0.3) -> Dict[str, Any]:
        """
        Evaluate how well clustering predicts student scores.
        
        Args:
            test_split: Fraction of data to use for testing
            
        Returns:
            Evaluation metrics for score prediction
        """
        if not self.clustering_manager.submissions:
            raise ValueError("No submissions available for evaluation")
        
        submissions = self.clustering_manager.submissions
        embeddings = self.clustering_manager.embeddings
        scores = self.clustering_manager.scores
        
        # Split data
        indices = np.arange(len(submissions))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=42, stratify=scores
        )
        
        train_embeddings = embeddings[train_indices]
        test_embeddings = embeddings[test_indices]
        train_scores = scores[train_indices]
        test_scores = scores[test_indices]
        
        self.logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")
        
        # Retrain clustering on training data only
        temp_manager = ClusteringManager(self.assignment_id, self.embedder)
        temp_manager.submissions = [submissions[i] for i in train_indices]
        temp_manager.embeddings = train_embeddings
        temp_manager.scores = train_scores
        temp_manager.issue_vectors = self.clustering_manager.issue_vectors[train_indices]
        
        # Train score clustering
        temp_manager.train_score_clustering()
        
        # Predict on test set
        scaled_test_embeddings = temp_manager.scaler.transform(test_embeddings)
        predicted_clusters = temp_manager.score_clusterer.predict(scaled_test_embeddings)
        
        # Map cluster assignments to predicted score ranges
        predicted_score_ranges = []
        actual_scores = []
        
        for i, cluster_id in enumerate(predicted_clusters):
            cluster_info = temp_manager.score_cluster_info.get(f"cluster_{cluster_id}", {})
            predicted_range = cluster_info.get('score_range', [0, 1])
            avg_predicted_score = cluster_info.get('avg_score', 0.5)
            
            predicted_score_ranges.append(avg_predicted_score)
            actual_scores.append(test_scores[i])
        
        # Calculate evaluation metrics
        predicted_scores = np.array(predicted_score_ranges)
        actual_scores = np.array(actual_scores)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predicted_scores - actual_scores))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((predicted_scores - actual_scores) ** 2))
        
        # Accuracy within score bands (0.1 tolerance)
        within_band_accuracy = np.mean(np.abs(predicted_scores - actual_scores) <= 0.1)
        
        # Grade classification accuracy (convert scores to letter grades)
        actual_grades = self._scores_to_grades(actual_scores)
        predicted_grades = self._scores_to_grades(predicted_scores)
        grade_accuracy = accuracy_score(actual_grades, predicted_grades)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'within_band_accuracy': float(within_band_accuracy),
            'grade_accuracy': float(grade_accuracy),
            'grade_classification_report': classification_report(actual_grades, predicted_grades, output_dict=True),
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            'test_score_distribution': {
                str(score): int(count) for score, count in zip(*np.unique(actual_scores, return_counts=True))
            }
        }
    
    def _scores_to_grades(self, scores: np.ndarray) -> List[str]:
        """Convert numeric scores to letter grades"""
        grades = []
        for score in scores:
            if score >= 0.9:
                grades.append('A')
            elif score >= 0.8:
                grades.append('B')
            elif score >= 0.7:
                grades.append('C')
            elif score >= 0.6:
                grades.append('D')
            else:
                grades.append('F')
        return grades
    
    def evaluate_issue_detection(self) -> Dict[str, Any]:
        """Evaluate how well clustering detects common issues"""
        if not hasattr(self.clustering_manager, 'issue_cluster_info'):
            return {'error': 'Issue clustering not available'}
        
        issue_analysis = {}
        
        # Analyze each issue cluster
        for cluster_id, cluster_info in self.clustering_manager.issue_cluster_info.items():
            dominant_issues = cluster_info['dominant_issues']
            cluster_size = cluster_info['size']
            avg_score = cluster_info['avg_score']
            
            issue_analysis[cluster_id] = {
                'size': cluster_size,
                'avg_score': avg_score,
                'issues_detected': dominant_issues,
                'issue_coherence': len(dominant_issues) > 0,  # Whether cluster has clear issues
                'score_consistency': True  # You could add variance calculation here
            }
        
        # Overall metrics
        total_clusters = len(issue_analysis)
        clusters_with_clear_issues = sum(1 for info in issue_analysis.values() if info['issue_coherence'])
        
        return {
            'total_issue_clusters': total_clusters,
            'clusters_with_clear_issues': clusters_with_clear_issues,
            'issue_detection_rate': clusters_with_clear_issues / max(total_clusters, 1),
            'cluster_analysis': issue_analysis
        }
    
    def compare_with_faiss(self, n_queries: int = 10) -> Dict[str, Any]:
        """
        Compare clustering approach with FAISS similarity search.
        
        Args:
            n_queries: Number of random queries to test
            
        Returns:
            Comparison metrics between clustering and FAISS
        """
        if not self.faiss_manager:
            return {'error': 'FAISS manager not available for comparison'}
        
        submissions = self.clustering_manager.submissions
        if len(submissions) < n_queries:
            n_queries = len(submissions)
        
        # Randomly sample query submissions
        query_indices = np.random.choice(len(submissions), n_queries, replace=False)
        
        comparison_results = []
        
        for i, query_idx in enumerate(query_indices):
            query_submission = submissions[query_idx]
            query_embedding = self.clustering_manager.embeddings[query_idx]
            
            self.logger.debug(f"Comparing query {i+1}/{n_queries}: {query_submission.file_name}")
            
            # Get clustering prediction
            cluster_result = self.clustering_manager.predict_clusters(query_submission)
            
            # Get FAISS similar submissions
            faiss_results = self.faiss_manager.search_similar_in_assignment(
                assignment_id=self.assignment_id,
                query_embedding=query_embedding,
                top_k=5,
                embedder=self.embedder
            )
            
            # Analyze results
            query_score = query_submission.score
            
            # Clustering accuracy
            predicted_range = cluster_result.get('predicted_score_range', [0, 1])
            clustering_score_accuracy = (predicted_range[0] <= query_score <= predicted_range[1])
            
            # FAISS score similarity
            faiss_scores = [result['submission'].score for result in faiss_results[:3]]
            faiss_score_similarity = np.mean([abs(score - query_score) for score in faiss_scores]) if faiss_scores else 1.0
            
            comparison_results.append({
                'query_file': query_submission.file_name,
                'actual_score': query_score,
                'clustering_prediction': cluster_result,
                'clustering_score_accuracy': clustering_score_accuracy,
                'faiss_similar_scores': faiss_scores,
                'faiss_score_mae': faiss_score_similarity,
                'faiss_results_count': len(faiss_results)
            })
        
        # Aggregate comparison metrics
        clustering_accuracies = [r['clustering_score_accuracy'] for r in comparison_results]
        faiss_maes = [r['faiss_score_mae'] for r in comparison_results if r['faiss_score_mae'] is not None]
        
        return {
            'n_queries': n_queries,
            'clustering_score_accuracy': np.mean(clustering_accuracies),
            'faiss_average_score_mae': np.mean(faiss_maes) if faiss_maes else None,
            'detailed_comparisons': comparison_results,
            'summary': {
                'clustering_wins': sum(1 for r in comparison_results if r['clustering_score_accuracy']),
                'average_faiss_mae': np.mean(faiss_maes) if faiss_maes else None
            }
        }
    
    def generate_grading_context_examples(self, n_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Generate examples of grading context that would be provided to LLM.
        
        Args:
            n_examples: Number of examples to generate
            
        Returns:
            List of grading context examples
        """
        submissions = self.clustering_manager.submissions
        examples = []
        
        # Sample diverse submissions
        sample_indices = np.random.choice(len(submissions), min(n_examples, len(submissions)), replace=False)
        
        for idx in sample_indices:
            submission = submissions[idx]
            
            # Get clustering context
            cluster_result = self.clustering_manager.predict_clusters(submission)
            
            # Format grading context
            context = {
                'student_submission': submission.file_name,
                'actual_score': submission.score,
                'actual_feedback': submission.feedback[:200] + "...",
                'clustering_context': {
                    'predicted_score_range': cluster_result.get('predicted_score_range', []),
                    'score_cluster_info': cluster_result.get('score_cluster_info', {}),
                    'issue_cluster_info': cluster_result.get('issue_cluster_info', {}),
                    'common_issues_in_cluster': cluster_result.get('issue_cluster_info', {}).get('dominant_issues', [])
                },
                'llm_prompt_context': self._format_llm_context(cluster_result)
            }
            
            examples.append(context)
        
        return examples
    
    def _format_llm_context(self, cluster_result: Dict[str, Any]) -> str:
        """Format clustering results as LLM prompt context"""
        score_info = cluster_result.get('score_cluster_info', {})
        issue_info = cluster_result.get('issue_cluster_info', {})
        
        context = "**CLUSTERING-BASED GRADING CONTEXT:**\n\n"
        
        # Score context
        predicted_range = cluster_result.get('predicted_score_range', [0, 1])
        avg_score = score_info.get('avg_score', 0)
        context += f"**Predicted Score Range:** {predicted_range[0]:.2f} - {predicted_range[1]:.2f}\n"
        context += f"**Similar Submissions Average:** {avg_score:.2f}\n\n"
        
        # Issue context
        common_issues = issue_info.get('dominant_issues', [])
        if common_issues:
            context += f"**Common Issues in Similar Submissions:**\n"
            for issue in common_issues:
                context += f"- {issue.replace('_', ' ').title()}\n"
        
        # Sample feedback
        sample_feedback = issue_info.get('sample_feedback', '')
        if sample_feedback:
            context += f"\n**Sample Feedback from Similar Submissions:**\n{sample_feedback}\n"
        
        context += "\n**Instructions:** Use this context to calibrate your grading and ensure consistency with similar submissions."
        
        return context


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Evaluate clustering vs FAISS performance")
    
    parser.add_argument("--assignment", type=str, required=True, help="Assignment ID")
    parser.add_argument("--clustering-models", type=str, required=True, help="Path to trained clustering models")
    parser.add_argument("--faiss-indices", type=str, default=None, help="Path to FAISS indices for comparison")
    parser.add_argument("--data-dir", type=str, default="src/faiss/data", help="Data directory")
    parser.add_argument("--output", type=str, default="clustering_evaluation.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    logger.info(f"Starting evaluation for assignment: {args.assignment}")
    
    try:
        # Load embedder
        embedder = create_java_embedder(model_name="starcoder2:15b", use_ollama=True)
        
        # Load clustering manager
        clustering_manager = ClusteringManager(args.assignment, embedder)
        
        # Load submissions and grades
        multi_processor = MultiFolderProcessor(args.data_dir)
        submissions_by_assignment = multi_processor.process_all_tasks()
        submissions = submissions_by_assignment[args.assignment]
        
        grade_mapping_path = f"src/faiss/grade_mapping/{args.assignment}.csv"
        clustering_manager.load_submissions_and_grades(submissions, grade_mapping_path)
        clustering_manager.generate_embeddings()
        
        # Load trained models
        clustering_manager.load_models(args.clustering_models)
        
        # Initialize evaluator
        faiss_manager = None
        if args.faiss_indices:
            faiss_manager = AssignmentFAISSManager(args.faiss_indices)
            faiss_manager.load_assignment_indices()
        
        evaluator = ClusteringEvaluator(args.assignment, clustering_manager, faiss_manager, embedder)
        
        # Run evaluations
        logger.info("Evaluating score prediction...")
        score_eval = evaluator.evaluate_score_prediction()
        
        logger.info("Evaluating issue detection...")
        issue_eval = evaluator.evaluate_issue_detection()
        
        faiss_comparison = {}
        if faiss_manager:
            logger.info("Comparing with FAISS...")
            faiss_comparison = evaluator.compare_with_faiss()
        
        logger.info("Generating grading context examples...")
        context_examples = evaluator.generate_grading_context_examples()
        
        # Compile results
        evaluation_results = {
            'assignment_id': args.assignment,
            'evaluation_timestamp': str(np.datetime64('now')),
            'score_prediction_evaluation': score_eval,
            'issue_detection_evaluation': issue_eval,
            'faiss_comparison': faiss_comparison,
            'grading_context_examples': context_examples,
            'cluster_summary': clustering_manager.get_cluster_summary()
        }
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("[CLUSTERING EVALUATION RESULTS]")
        print("="*80)
        print(f"Assignment: {args.assignment}")
        print(f"Results saved to: {args.output}")
        
        print(f"\n[SCORE PREDICTION]")
        print(f"  MAE: {score_eval['mae']:.3f}")
        print(f"  RMSE: {score_eval['rmse']:.3f}")
        print(f"  Within-band accuracy: {score_eval['within_band_accuracy']:.2%}")
        print(f"  Grade accuracy: {score_eval['grade_accuracy']:.2%}")
        
        print(f"\n[ISSUE DETECTION]")
        print(f"  Issue clusters: {issue_eval['total_issue_clusters']}")
        print(f"  Clusters with clear issues: {issue_eval['clusters_with_clear_issues']}")
        print(f"  Issue detection rate: {issue_eval['issue_detection_rate']:.2%}")
        
        if faiss_comparison:
            print(f"\n[FAISS COMPARISON]")
            print(f"  Clustering score accuracy: {faiss_comparison['clustering_score_accuracy']:.2%}")
            print(f"  FAISS average MAE: {faiss_comparison.get('faiss_average_score_mae', 'N/A')}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()