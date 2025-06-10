#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java Code Retrieval Evaluation Script

This script evaluates the effectiveness of the Java code embedding and retrieval system
by testing self-retrieval rates and other metrics across assignments.

Key Metrics:
- Self-Retrieval Rate: How often a submission retrieves itself as the top result
- Assignment Coherence: How well submissions from the same assignment cluster together  
- Cross-Assignment Discrimination: How well it distinguishes between different assignments
- Average Similarity Scores: Quality of similarity rankings

Author: Auto-generated
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime

# Load environment variables from config.env
try:
    from dotenv import load_dotenv
    config_path = Path(__file__).parent.parent.parent / "config.env"
    if config_path.exists():
        load_dotenv(config_path)
        print(f"[SUCCESS] Loaded configuration from {config_path}")
    else:
        print(f"[WARNING] Config file not found at {config_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed. Install with: pip install python-dotenv")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.faiss.embedder import create_java_embedder
from src.faiss.faiss_manager import FAISSManager
from src.faiss.processor import MultiFolderProcessor
from src.utils.logger import get_logger, setup_logger


class RetrievalEvaluator:
    """Evaluates the effectiveness of Java code retrieval system"""
    
    def __init__(self, assignment_manager: FAISSManager, embedder, logger):
        self.assignment_manager = assignment_manager
        self.embedder = embedder
        self.logger = logger
        self.results = {}
        
    def evaluate_assignment(self, assignment_id: str, submissions: List[Any], 
                          sample_size: int = 5, top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate retrieval effectiveness for a specific assignment.
        
        Args:
            assignment_id: ID of the assignment
            submissions: List of submission objects
            sample_size: Number of random submissions to sample as queries
            top_k: Number of top results to retrieve for each query
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating assignment: {assignment_id}")
        
        if len(submissions) < sample_size:
            self.logger.warning(f"Assignment {assignment_id} has only {len(submissions)} submissions, using all")
            sample_size = len(submissions)
        
        # Randomly sample submissions for evaluation
        sampled_submissions = random.sample(submissions, sample_size)
        
        metrics = {
            'assignment_id': assignment_id,
            'total_submissions': len(submissions),
            'sampled_submissions': sample_size,
            'self_retrieval_hits': 0,
            'self_retrieval_rate': 0.0,
            'same_assignment_in_top_k': 0,
            'same_assignment_rate': 0.0,
            'avg_similarity_scores': [],
            'detailed_results': []
        }
        
        for i, query_submission in enumerate(sampled_submissions):
            self.logger.debug(f"Processing query {i+1}/{sample_size}: {query_submission.file_name}")
            
            try:
                # Generate embedding for the query submission
                query_embedding = self.embedder.embed_codebase(query_submission.code_files)
                
                # Search for similar submissions within the same assignment
                results = self.assignment_manager.search_similar_in_assignment(
                    assignment_id=assignment_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    embedder=self.embedder
                )
                
                if not results:
                    self.logger.warning(f"No results for query: {query_submission.file_name}")
                    continue
                
                # Extract query student ID for comparison
                query_student_id = self._extract_student_id(query_submission.file_name)
                
                # Analyze results
                query_result = {
                    'query_submission': query_submission.file_name,
                    'query_student_id': query_student_id,
                    'query_assignment': assignment_id,
                    'similar_submissions': [],
                    'self_retrieval': False,
                    'self_retrieval_rank': -1,
                    'same_assignment_count': 0,
                    'avg_similarity': 0.0
                }
                
                similarity_scores = []
                
                # Process similar results
                for rank, result in enumerate(results, 1):
                    similar_student_id = self._extract_student_id(result['submission'].file_name)
                    similarity_score = result['similarity_score']
                    similarity_scores.append(similarity_score)
                    
                    similar_info = {
                        'rank': rank,
                        'submission_name': result['submission'].file_name,
                        'student_id': similar_student_id,
                        'similarity_score': round(similarity_score, 4),
                        'assignment_id': result['submission'].assignment_id,
                        'is_self': similar_student_id == query_student_id,
                        'same_assignment': result['submission'].assignment_id == assignment_id,
                        'grade': getattr(result['submission'], 'score', None),
                        'feedback_snippet': getattr(result['submission'], 'feedback', '')[:100] + "..." if hasattr(result['submission'], 'feedback') and result['submission'].feedback else None
                    }
                    
                    query_result['similar_submissions'].append(similar_info)
                    
                    # Check for self-retrieval (rank 1)
                    if rank == 1 and similar_student_id == query_student_id:
                        query_result['self_retrieval'] = True
                        query_result['self_retrieval_rank'] = 1
                        metrics['self_retrieval_hits'] += 1
                    
                    # Check if self appears anywhere in results
                    if similar_student_id == query_student_id and query_result['self_retrieval_rank'] == -1:
                        query_result['self_retrieval_rank'] = rank
                    
                    # Count same assignment submissions
                    if result['submission'].assignment_id == assignment_id:
                        query_result['same_assignment_count'] += 1
                        metrics['same_assignment_in_top_k'] += 1
                
                # Calculate average similarity for this query
                if similarity_scores:
                    query_result['avg_similarity'] = np.mean(similarity_scores)
                    metrics['avg_similarity_scores'].extend(similarity_scores)
                
                metrics['detailed_results'].append(query_result)
                
            except Exception as e:
                self.logger.error(f"Error processing query {query_submission.file_name}: {e}")
                continue
        
        # Calculate final metrics
        if sample_size > 0:
            metrics['self_retrieval_rate'] = metrics['self_retrieval_hits'] / sample_size
            metrics['same_assignment_rate'] = metrics['same_assignment_in_top_k'] / (sample_size * top_k)
        
        if metrics['avg_similarity_scores']:
            metrics['avg_similarity'] = np.mean(metrics['avg_similarity_scores'])
            metrics['std_similarity'] = np.std(metrics['avg_similarity_scores'])
            metrics['min_similarity'] = np.min(metrics['avg_similarity_scores'])
            metrics['max_similarity'] = np.max(metrics['avg_similarity_scores'])
        
        self.logger.info(f"Assignment {assignment_id} - Self-retrieval rate: {metrics['self_retrieval_rate']:.2%}")
        
        return metrics
    

    
    def run_full_evaluation(self, submissions_by_assignment: Dict[str, List[Any]], 
                           sample_size: int = 5, top_k: int = 10,
                           target_assignments: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete evaluation across specified assignments.
        
        Args:
            submissions_by_assignment: All submissions grouped by assignment
            sample_size: Number of submissions to sample per assignment
            top_k: Number of top results to retrieve
            target_assignments: Specific assignments to evaluate (None for all)
            
        Returns:
            Complete evaluation results in clustering-style format
        """
        self.logger.info("Starting full retrieval evaluation")
        
        # Filter assignments if specified
        if target_assignments:
            submissions_by_assignment = {
                aid: subs for aid, subs in submissions_by_assignment.items() 
                if aid in target_assignments
            }
        
        # Determine evaluation type
        evaluation_type = 'single_assignment' if len(submissions_by_assignment) == 1 else 'multi_assignment'
        assignment_id = list(submissions_by_assignment.keys())[0] if len(submissions_by_assignment) == 1 else None
        
        overall_results = {
            'assignment_id': assignment_id if evaluation_type == 'single_assignment' else 'multi_assignment',
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_type': evaluation_type,
            'model_info': self.embedder.get_embedding_info(),
            'evaluation_params': {
                'sample_size': sample_size,
                'top_k': top_k,
                'target_assignments': target_assignments or list(submissions_by_assignment.keys())
            },
            'retrieval_performance_evaluation': {},
            'grading_context_examples': [],
            'assignment_results': {}
        }
        
        # Evaluate each assignment individually
        assignment_metrics = []
        all_grading_examples = []
        
        for assignment_id, submissions in submissions_by_assignment.items():
            metrics = self.evaluate_assignment(assignment_id, submissions, sample_size, top_k)
            
            # Generate grading context examples
            self.logger.info(f"Generating grading context examples for {assignment_id}...")
            grading_examples = self.generate_grading_context_examples(
                assignment_id, submissions, n_examples=min(3, len(submissions))
            )
            
            # Add assignment_id to each example
            for example in grading_examples:
                example['assignment_id'] = assignment_id
            
            all_grading_examples.extend(grading_examples)
            overall_results['assignment_results'][assignment_id] = metrics
            assignment_metrics.append(metrics)
        
        # Calculate retrieval performance evaluation (similar to clustering's score prediction)
        if assignment_metrics:
            retrieval_scores = []
            predicted_scores = []
            
            for example in all_grading_examples:
                actual_score = example.get('actual_score')
                predicted_score = example['retrieval_context'].get('predicted_score_from_retrievals')
                
                if actual_score is not None and predicted_score is not None:
                    retrieval_scores.append(actual_score)
                    predicted_scores.append(predicted_score)
            
            if retrieval_scores and predicted_scores:
                retrieval_scores = np.array(retrieval_scores)
                predicted_scores = np.array(predicted_scores)
                
                mae = np.mean(np.abs(retrieval_scores - predicted_scores))
                rmse = np.sqrt(np.mean((retrieval_scores - predicted_scores) ** 2))
                within_band_accuracy = np.mean(np.abs(retrieval_scores - predicted_scores) <= 0.1)
                
                overall_results['retrieval_performance_evaluation'] = {
                    'score_prediction_mae': float(mae),
                    'score_prediction_rmse': float(rmse),
                    'within_band_accuracy': float(within_band_accuracy),
                    'n_predictions': len(retrieval_scores),
                    'avg_similarity_across_assignments': np.mean([m.get('avg_similarity', 0) for m in assignment_metrics if 'avg_similarity' in m])
                }
        
        # Add grading context examples to results
        overall_results['grading_context_examples'] = all_grading_examples
        
        # Calculate overall metrics
        if assignment_metrics:
            overall_results['overall_metrics'] = {
                'self_retrieval_rate': np.mean([m['self_retrieval_rate'] for m in assignment_metrics]),
                'same_assignment_rate': np.mean([m['same_assignment_rate'] for m in assignment_metrics]),
                'avg_similarity': np.mean([m.get('avg_similarity', 0) for m in assignment_metrics if 'avg_similarity' in m]),
                'assignments_evaluated': len(assignment_metrics),
                'total_queries_processed': sum(m['sampled_submissions'] for m in assignment_metrics),
                'total_submissions': sum(m['total_submissions'] for m in assignment_metrics)
            }
        
        return overall_results

    def _extract_student_id(self, file_name: str) -> str:
        """Extract student ID from submission file name"""
        # Try to extract student ID from filename patterns like "username_id_timestamp_file.zip"
        if '_' in file_name:
            parts = file_name.split('_')
            if len(parts) >= 1:
                return parts[0]  # First part is usually the student identifier
        
        # Fallback: return the file name without extension
        return file_name.replace('.zip', '').replace('.java', '')

    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a format similar to clustering evaluation"""
        print("\n" + "="*80)
        print("[RETRIEVAL EVALUATION RESULTS]")
        print("="*80)
        
        assignment_id = results['assignment_id']
        print(f"Assignment: {assignment_id}")
        print(f"Results saved to: retrieval_evaluation_results.json")
        
        # Retrieval Performance Evaluation (similar to clustering's score prediction)
        if 'retrieval_performance_evaluation' in results and results['retrieval_performance_evaluation']:
            perf_eval = results['retrieval_performance_evaluation']
            print(f"\n[SCORE PREDICTION FROM RETRIEVALS]")
            print(f"  MAE: {perf_eval['score_prediction_mae']:.3f}")
            print(f"  RMSE: {perf_eval['score_prediction_rmse']:.3f}")
            print(f"  Within-band accuracy: {perf_eval['within_band_accuracy']:.2%}")
            print(f"  Predictions made: {perf_eval['n_predictions']}")
            print(f"  Avg similarity: {perf_eval['avg_similarity_across_assignments']:.3f}")
        
        # Overall retrieval metrics
        if 'overall_metrics' in results:
            overall = results['overall_metrics']
            print(f"\n[RETRIEVAL PERFORMANCE]")
            print(f"  Self-retrieval rate: {overall['self_retrieval_rate']:.2%}")
            print(f"  Same assignment rate: {overall['same_assignment_rate']:.2%}")
            print(f"  Average similarity: {overall['avg_similarity']:.3f}")
            print(f"  Total queries processed: {overall['total_queries_processed']}")
        
        # Grading Context Examples (similar to clustering examples)
        if 'grading_context_examples' in results and results['grading_context_examples']:
            print(f"\n[GRADING CONTEXT EXAMPLES]")
            print("-" * 80)
            
            for i, example in enumerate(results['grading_context_examples'][:3], 1):
                print(f"\nExample {i}: Student {example['student_submission']}")
                print(f"  Actual Score: {example['actual_score']:.2f}" if example['actual_score'] else "  Actual Score: N/A")
                print(f"  Actual Feedback: {example['actual_feedback']}")
                
                retrieval_ctx = example['retrieval_context']
                top_similar = retrieval_ctx['top_similar_submission']
                
                print(f"\n  Top Retrieved Similar Submission:")
                print(f"    Student: {top_similar['student_name']}")
                print(f"    Similarity Score: {top_similar['similarity_score']}")
                print(f"    Actual Score: {top_similar['actual_score']:.2f}" if top_similar['actual_score'] else "    Actual Score: N/A")
                print(f"    Actual Feedback: {top_similar['actual_feedback']}")
                
                if retrieval_ctx['predicted_score_from_retrievals']:
                    print(f"\n  Model Predictions Based on Retrievals:")
                    print(f"    Predicted Score: {retrieval_ctx['predicted_score_from_retrievals']:.3f}")
                    print(f"    Score Variance: {retrieval_ctx['retrieval_statistics']['score_variance']:.4f}")
                    print(f"    Retrievals with grades: {retrieval_ctx['retrieval_statistics']['n_retrievals_with_grades']}")
                
                print(f"\n  LLM Context That Would Be Provided:")
                context_lines = example['llm_prompt_context'].split('\n')[:8]  # Show first 8 lines
                for line in context_lines:
                    print(f"    {line}")
                if len(example['llm_prompt_context'].split('\n')) > 8:
                    print("    [... context truncated ...]")
                
                if i < len(results['grading_context_examples'][:3]):
                    print(f"\n{'-' * 40}")
        
        # Per-assignment detailed results for multi-assignment evaluation
        if results['evaluation_type'] == 'multi_assignment' and len(results['assignment_results']) > 1:
            print(f"\n[PER-ASSIGNMENT RESULTS]")
            for assignment_id, metrics in results['assignment_results'].items():
                print(f"\n• {assignment_id}:")
                print(f"  - Self-Retrieval Rate: {metrics['self_retrieval_rate']:.2%}")
                print(f"  - Same Assignment Rate: {metrics['same_assignment_rate']:.2%}")
                print(f"  - Average Similarity: {metrics.get('avg_similarity', 0):.3f}")
                print(f"  - Queries: {metrics['sampled_submissions']}")
        
        print("="*80)

    def generate_grading_context_examples(self, assignment_id: str, submissions: List[Any], 
                                         n_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Generate examples of grading context that would be provided to LLM using FAISS retrieval.
        
        Args:
            assignment_id: Assignment identifier
            submissions: List of submission objects
            n_examples: Number of examples to generate
            
        Returns:
            List of grading context examples
        """
        if len(submissions) < n_examples:
            n_examples = len(submissions)
        
        # Sample diverse submissions
        sampled_submissions = random.sample(submissions, n_examples)
        examples = []
        
        for submission in sampled_submissions:
            try:
                # Generate embedding for the query submission
                query_embedding = self.embedder.embed_codebase(submission.code_files)
                
                # Search for similar submissions
                results = self.assignment_manager.search_similar_in_assignment(
                    assignment_id=assignment_id,
                    query_embedding=query_embedding,
                    top_k=5,
                    embedder=self.embedder
                )
                
                if not results:
                    continue
                
                # Get the top retrieval result (most similar submission)
                top_result = results[0]
                top_submission = top_result['submission']
                
                # Extract student name
                student_name = self._extract_student_id(submission.file_name)
                
                # Calculate predicted score based on top retrievals
                retrieval_scores = [r['submission'].score for r in results[:3] if hasattr(r['submission'], 'score') and r['submission'].score is not None]
                predicted_score = np.mean(retrieval_scores) if retrieval_scores else None
                
                # Get sample feedback from top retrievals
                sample_feedbacks = [r['submission'].feedback for r in results[:3] if hasattr(r['submission'], 'feedback') and r['submission'].feedback]
                sample_feedback = sample_feedbacks[1] if sample_feedbacks else "No feedback available"  # first one is itslef
                
                # Format grading context
                context = {
                    'student_submission': student_name,
                    'actual_score': getattr(submission, 'score', None),
                    'actual_feedback': getattr(submission, 'feedback', "No feedback available")[:200] + "...",
                    'retrieval_context': {
                        'top_similar_submission': {
                            'student_name': self._extract_student_id(top_submission.file_name),
                            'similarity_score': round(top_result['similarity_score'], 4),
                            'actual_score': getattr(top_submission, 'score', None),
                            'actual_feedback': getattr(top_submission, 'feedback', "No feedback available")[:200] + "...",
                        },
                        'predicted_score_from_retrievals': round(predicted_score, 3) if predicted_score else None,
                        'retrieval_based_feedback_sample': sample_feedback[:200] + "...",
                        'similarity_scores': [round(r['similarity_score'], 4) for r in results[:3]],
                        'all_similar_scores': retrieval_scores[:5] if retrieval_scores else [],
                        'retrieval_statistics': {
                            'avg_similarity': round(np.mean([r['similarity_score'] for r in results]), 4),
                            'score_variance': round(np.var(retrieval_scores), 4) if len(retrieval_scores) > 1 else 0,
                            'n_retrievals_with_grades': len(retrieval_scores)
                        }
                    },
                    'llm_prompt_context': self._format_retrieval_llm_context(results, predicted_score)
                }
                
                examples.append(context)
                
            except Exception as e:
                self.logger.error(f"Error generating context for {submission.file_name}: {e}")
                continue
        
        return examples
    
    def _format_retrieval_llm_context(self, results: List[Dict], predicted_score: float = None) -> str:
        """Format retrieval results as LLM prompt context"""
        if not results:
            return "No similar submissions found."
        
        context = "**RETRIEVAL-BASED GRADING CONTEXT:**\n\n"
        
        # Score prediction from similar submissions
        if predicted_score:
            context += f"**Predicted Score Range:** {int(predicted_score*100)}% (±5%)\n"
            context += f"**Similar Submissions Average:** {int(predicted_score*100)}% of total points\n\n"
        
        # Top similar submissions
        context += f"**Top {min(3, len(results))} Most Similar Submissions:**\n"
        for i, result in enumerate(results[:3], 1):
            submission = result['submission']
            similarity = result['similarity_score']
            score = getattr(submission, 'score', None)
            student_name = submission.file_name.split('_')[0] if '_' in submission.file_name else submission.file_name
            
            context += f"{i}. Student: {student_name} | Similarity: {similarity:.3f}"
            if score is not None:
                context += f" | Score: {int(score*100)}%"
            context += "\n"
        
        # Sample feedback from most similar submission
        if hasattr(results[0]['submission'], 'feedback') and results[0]['submission'].feedback:
            context += f"\n**Sample Feedback from Most Similar Submission:**\n"
            context += f"{results[0]['submission'].feedback[:200]}...\n"
        
        context += "\n**Instructions:** Use this context to calibrate your grading and ensure consistency with similar submissions."
        if predicted_score:
            context += f" This submission should receive approximately {int(predicted_score*100)}% of the total points based on similar submissions."
        
        return context


def load_submissions_from_data(data_dir: str, target_assignments: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    """Load submissions from the data directory"""
    logger = get_logger(__name__)
    logger.info(f"Loading submissions from {data_dir}")
    
    multi_processor = MultiFolderProcessor(data_dir)
    submissions_by_assignment = multi_processor.process_all_tasks()
    
    if target_assignments:
        submissions_by_assignment = {
            aid: subs for aid, subs in submissions_by_assignment.items() 
            if aid in target_assignments
        }
    
    logger.info(f"Loaded {len(submissions_by_assignment)} assignments")
    for aid, subs in submissions_by_assignment.items():
        logger.info(f"  {aid}: {len(subs)} submissions")
    
    return submissions_by_assignment


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Evaluate Java code retrieval effectiveness")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/faiss/data",
        help="Directory containing task folders with submissions"
    )
    
    parser.add_argument(
        "--indices-path",
        type=str,
        default="src/faiss/assignment_indices",
        help="Path to assignment FAISS indices"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="starCoder2:15b",
        help="Ollama model name for embeddings"
    )
    
    parser.add_argument(
        "--assignment",
        type=str,
        default=None,
        help="Specific assignment to evaluate (default: all assignments)"
    )
    
    parser.add_argument(
        "--grade-mapping-dir",
        type=str,
        default="src/faiss/grade_mapping",
        help="Directory containing grade mapping CSV files"
    )
    
    parser.add_argument(
        "--with-grades",
        action="store_true",
        help="Enable grade mapping integration for enriched evaluation"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of submissions to sample per assignment for evaluation"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve for each query"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="retrieval_evaluation_results.json",
        help="Output file for evaluation results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("Starting Java code retrieval evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load Ollama configuration
        ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Create Java embedder
        logger.info(f"Creating Java embedder with model: {args.model_name}")
        embedder = create_java_embedder(
            model_name=args.model_name,
            use_ollama=True,
            ollama_base_url=ollama_base_url
        )
        
        # Load assignment indices
        logger.info(f"Loading assignment indices from {args.indices_path}")
        assignment_manager = FAISSManager(
            base_index_path=args.indices_path,
            index_type="flat"
        )
        
        if not assignment_manager.load_assignment_indices():
            logger.error("Failed to load assignment indices. Please build them first.")
            sys.exit(1)
        
        available_assignments = assignment_manager.get_available_assignments()
        logger.info(f"Available assignments: {available_assignments}")
        
        # Determine target assignments
        target_assignments = None
        if args.assignment:
            if args.assignment not in available_assignments:
                logger.error(f"Assignment '{args.assignment}' not found. Available: {available_assignments}")
                sys.exit(1)
            target_assignments = [args.assignment]
            logger.info(f"Evaluating single assignment: {args.assignment}")
        else:
            logger.info("Evaluating all available assignments")
        
        # Load submissions from data directory
        submissions_by_assignment = load_submissions_from_data(args.data_dir, target_assignments)
        
        if not submissions_by_assignment:
            logger.error("No submissions found")
            sys.exit(1)
        
        # Enrich with grade mapping if requested
        if args.with_grades:
            logger.info(f"Enriching submissions with grade mapping from {args.grade_mapping_dir}")
            enriched_submissions = {}
            grade_mapping_path = Path(args.grade_mapping_dir)
            
            for assignment_id, submissions in submissions_by_assignment.items():
                csv_file = grade_mapping_path / f"{assignment_id}.csv"
                
                if csv_file.exists():
                    try:
                        enriched_submissions[assignment_id] = assignment_manager.load_grades_for_assignment(
                            assignment_id, submissions, str(csv_file)
                        )
                        logger.info(f"Enriched {assignment_id} with grades from {csv_file}")
                    except Exception as e:
                        logger.error(f"Failed to load grades for {assignment_id}: {e}")
                        enriched_submissions[assignment_id] = submissions
                else:
                    logger.warning(f"No grade mapping found for {assignment_id}")
                    enriched_submissions[assignment_id] = submissions
            
            submissions_by_assignment = enriched_submissions
        
        # Run evaluation
        evaluator = RetrievalEvaluator(assignment_manager, embedder, logger)
        results = evaluator.run_full_evaluation(
            submissions_by_assignment=submissions_by_assignment,
            sample_size=args.sample_size,
            top_k=args.top_k,
            target_assignments=target_assignments
        )
        
        # Save results to file
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        evaluator.print_results(results)
        
        print(f"[SUCCESS] Evaluation completed! Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 