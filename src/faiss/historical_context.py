"""
Historical Context Provider

This module provides historical context integration for the autograder,
allowing LLMs to consider similar past submissions when evaluating new ones.

Author: Auto-generated
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from .faiss_manager import FAISSManager
from .embedder import HybridCodeEmbedder
from .processor import Submission
from ..utils.logger import get_logger


class HistoricalContextProvider:
    """Provides historical context for LLM evaluations"""
    
    def __init__(self, faiss_manager: FAISSManager, embedder: HybridCodeEmbedder):
        """
        Initialize the historical context provider.
        
        Args:
            faiss_manager: FAISS manager instance with trained index
            embedder: Hybrid code embedder instance
        """
        self.faiss_manager = faiss_manager
        self.embedder = embedder
        self.logger = get_logger(__name__)
        
        # Check if FAISS manager is ready
        if not faiss_manager.is_trained:
            self.logger.warning("FAISS manager is not trained. Historical context will be limited.")
    
    def get_similar_submissions_context(self, code_files: Dict[str, str], 
                                      assignment_id: str,
                                      top_k: int = 3,
                                      min_similarity: float = 0.3) -> Dict[str, Any]:
        """
        Get context from similar historical submissions.
        
        Args:
            code_files: Current submission code files
            assignment_id: Assignment identifier
            top_k: Number of similar submissions to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary containing historical context information
        """
        if not self.faiss_manager.is_trained:
            return {
                'available': False,
                'reason': 'Historical index not available',
                'similar_submissions': []
            }
        
        try:
            # Search for similar submissions
            similar_results = self.faiss_manager.search_by_code_similarity(
                code_files=code_files,
                embedder=self.embedder,
                assignment_id=assignment_id,
                top_k=top_k
            )
            
            # Filter by minimum similarity
            filtered_results = [
                result for result in similar_results 
                if result['similarity_score'] >= min_similarity
            ]
            
            if not filtered_results:
                return {
                    'available': True,
                    'similar_submissions': [],
                    'message': f'No sufficiently similar submissions found (threshold: {min_similarity})'
                }
            
            # Format context information
            context_submissions = []
            for result in filtered_results:
                submission_context = {
                    'similarity_score': result['similarity_score'],
                    'score': result['score'],
                    'feedback_summary': self._summarize_feedback(result['feedback']),
                    'feedback_full': result['feedback'],
                    'file_names': list(result['submission'].code_files.keys()),
                    'metadata': result.get('metadata', {})
                }
                context_submissions.append(submission_context)
            
            return {
                'available': True,
                'similar_submissions': context_submissions,
                'assignment_id': assignment_id,
                'search_performed': True
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical context: {str(e)}")
            return {
                'available': False,
                'reason': f'Error during search: {str(e)}',
                'similar_submissions': []
            }
    
    def _summarize_feedback(self, feedback: str, max_length: int = 200) -> str:
        """Summarize feedback to key points"""
        if not feedback.strip():
            return "No feedback available"
        
        # Simple summarization - take first sentences up to max_length
        sentences = feedback.split('.')
        summary = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(summary + sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        
        if len(summary.strip()) == 0:
            # Fallback: truncate original feedback
            summary = feedback[:max_length] + "..." if len(feedback) > max_length else feedback
        
        return summary.strip()
    
    def create_historical_context_prompt(self, similar_submissions: List[Dict[str, Any]], 
                                       max_examples: int = 3) -> str:
        """
        Create a prompt section with historical context.
        
        Args:
            similar_submissions: List of similar submission contexts
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted prompt text with historical context
        """
        if not similar_submissions:
            return ""
        
        # Limit to max_examples
        examples = similar_submissions[:max_examples]
        
        prompt = "**HISTORICAL CONTEXT - Similar Past Submissions:**\n\n"
        prompt += "Consider these similar submissions from the same assignment as reference points for scoring and feedback:\n\n"
        
        for i, example in enumerate(examples, 1):
            similarity = example['similarity_score']
            score = example['score']
            feedback_summary = example['feedback_summary']
            
            prompt += f"**Example {i} (Similarity: {similarity:.2f}):**\n"
            prompt += f"- **Score Received:** {score:.1f}\n"
            prompt += f"- **Key Feedback Points:** {feedback_summary}\n"
            
            # Add metadata if available
            if example.get('metadata'):
                metadata = example['metadata']
                if 'complexity' in metadata:
                    prompt += f"- **Complexity Level:** {metadata['complexity']}\n"
                if 'approach' in metadata:
                    prompt += f"- **Approach Used:** {metadata['approach']}\n"
            
            prompt += "\n"
        
        prompt += "**Instructions for Using Historical Context:**\n"
        prompt += "- Use these examples as calibration points for consistent scoring\n"
        prompt += "- Consider similar strengths and weaknesses in your evaluation\n"
        prompt += "- Maintain fairness while learning from past evaluation patterns\n"
        prompt += "- Do not directly copy feedback but use insights for guidance\n\n"
        
        return prompt
    
    def get_assignment_statistics(self, assignment_id: str) -> Dict[str, Any]:
        """Get statistics for a specific assignment from historical data"""
        if not self.faiss_manager.is_trained:
            return {'available': False}
        
        try:
            assignment_submissions = self.faiss_manager.get_assignment_submissions(assignment_id)
            
            if not assignment_submissions:
                return {
                    'available': False,
                    'reason': f'No historical data for assignment {assignment_id}'
                }
            
            scores = [sub.score for sub in assignment_submissions if sub.score > 0]
            feedback_lengths = [len(sub.feedback) for sub in assignment_submissions if sub.feedback.strip()]
            
            stats = {
                'available': True,
                'total_submissions': len(assignment_submissions),
                'submissions_with_scores': len(scores),
                'score_statistics': {
                    'mean': sum(scores) / len(scores) if scores else 0,
                    'min': min(scores) if scores else 0,
                    'max': max(scores) if scores else 0,
                    'count': len(scores)
                },
                'feedback_statistics': {
                    'avg_feedback_length': sum(feedback_lengths) / len(feedback_lengths) if feedback_lengths else 0,
                    'submissions_with_feedback': len(feedback_lengths)
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting assignment statistics: {str(e)}")
            return {'available': False, 'reason': str(e)}
    
    def analyze_submission_uniqueness(self, code_files: Dict[str, str], 
                                    assignment_id: str,
                                    similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze how unique a submission is compared to historical ones.
        
        Args:
            code_files: Current submission code files
            assignment_id: Assignment identifier
            similarity_threshold: Threshold for considering submissions too similar
            
        Returns:
            Analysis of submission uniqueness
        """
        if not self.faiss_manager.is_trained:
            return {'available': False, 'reason': 'Historical index not available'}
        
        try:
            # Get all similar submissions (higher top_k for uniqueness analysis)
            similar_results = self.faiss_manager.search_by_code_similarity(
                code_files=code_files,
                embedder=self.embedder,
                assignment_id=assignment_id,
                top_k=10
            )
            
            # Analyze similarity distribution
            similarities = [result['similarity_score'] for result in similar_results]
            
            analysis = {
                'available': True,
                'max_similarity': max(similarities) if similarities else 0.0,
                'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0,
                'potentially_plagiarized': any(sim > similarity_threshold for sim in similarities),
                'similarity_threshold': similarity_threshold,
                'similar_count': len(similarities),
                'high_similarity_submissions': [
                    {
                        'similarity': result['similarity_score'],
                        'file_name': result['file_name'],
                        'score': result['score']
                    }
                    for result in similar_results 
                    if result['similarity_score'] > similarity_threshold
                ]
            }
            
            # Add uniqueness assessment
            if analysis['max_similarity'] > similarity_threshold:
                analysis['uniqueness_assessment'] = 'LOW - Highly similar to existing submission(s)'
            elif analysis['max_similarity'] > 0.6:
                analysis['uniqueness_assessment'] = 'MEDIUM - Some similarity to existing submissions'
            else:
                analysis['uniqueness_assessment'] = 'HIGH - Unique approach compared to historical submissions'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing submission uniqueness: {str(e)}")
            return {'available': False, 'reason': str(e)}
    
    def get_context_for_prompt_manager(self, code_files: Dict[str, str], 
                                     assignment_id: str,
                                     include_stats: bool = True,
                                     include_examples: bool = True,
                                     max_examples: int = 3) -> str:
        """
        Get formatted historical context for integration with PromptManager.
        
        Args:
            code_files: Current submission code files
            assignment_id: Assignment identifier
            include_stats: Whether to include assignment statistics
            include_examples: Whether to include similar examples
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted historical context string for prompt inclusion
        """
        context_parts = []
        
        # Add assignment statistics if requested
        if include_stats:
            stats = self.get_assignment_statistics(assignment_id)
            if stats.get('available'):
                stats_text = f"**Assignment Historical Statistics:**\n"
                stats_text += f"- Total Past Submissions: {stats['total_submissions']}\n"
                stats_text += f"- Submissions with Scores: {stats['submissions_with_scores']}\n"
                
                if stats['score_statistics']['count'] > 0:
                    score_stats = stats['score_statistics']
                    stats_text += f"- Average Score: {score_stats['mean']:.1f}\n"
                    stats_text += f"- Score Range: {score_stats['min']:.1f} - {score_stats['max']:.1f}\n"
                
                stats_text += "\n"
                context_parts.append(stats_text)
        
        # Add similar examples if requested
        if include_examples:
            similar_context = self.get_similar_submissions_context(
                code_files, assignment_id, top_k=max_examples
            )
            
            if similar_context.get('available') and similar_context['similar_submissions']:
                examples_text = self.create_historical_context_prompt(
                    similar_context['similar_submissions'], max_examples
                )
                context_parts.append(examples_text)
        
        # Combine all context parts
        if context_parts:
            full_context = "".join(context_parts)
            full_context += "---\n\n"  # Separator from main prompt content
            return full_context
        else:
            return ""
    
    def save_context_analysis(self, code_files: Dict[str, str], assignment_id: str, 
                            output_path: str):
        """Save detailed context analysis to file for debugging"""
        try:
            # Get comprehensive context information
            similar_context = self.get_similar_submissions_context(code_files, assignment_id, top_k=10)
            assignment_stats = self.get_assignment_statistics(assignment_id)
            uniqueness_analysis = self.analyze_submission_uniqueness(code_files, assignment_id)
            
            analysis = {
                'assignment_id': assignment_id,
                'timestamp': self.logger.handlers[0].formatter.formatTime() if self.logger.handlers else '',
                'similar_submissions_context': similar_context,
                'assignment_statistics': assignment_stats,
                'uniqueness_analysis': uniqueness_analysis,
                'faiss_index_stats': self.faiss_manager.get_statistics()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            self.logger.info(f"Context analysis saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving context analysis: {str(e)}")
    
    @staticmethod
    def is_available() -> bool:
        """Check if historical context is available"""
        try:
            from .faiss_manager import FAISS_AVAILABLE
            return FAISS_AVAILABLE
        except ImportError:
            return False 