"""
Issue Processor

Processor for handling student issues data for clustering.
Creates submission objects from issue data for compatibility with existing clustering system.

Author: Auto-generated
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ...utils.logger import get_logger


@dataclass
class IssueSubmission:
    """Data class representing a student's issues as a submission"""
    student_name: str
    issues: List[str]
    issue_count: int
    metadata: Dict[str, Any]
    
    def get_file_count(self) -> int:
        """Get total number of issues (for compatibility)"""
        return self.issue_count
    
    def get_java_file_count(self) -> int:
        """Get number of issues (for compatibility)"""
        return self.issue_count
    
    def has_java_files(self) -> bool:
        """Check if submission has issues (for compatibility)"""
        return self.issue_count > 0


class IssueProcessor:
    """
    Processes student issues data for clustering.
    
    Adapts issue data to work with the existing clustering architecture.
    """
    
    def __init__(self, issues_file: str):
        """
        Initialize processor for issue data.
        
        Args:
            issues_file: Path to JSON file containing student issues
        """
        self.issues_file = Path(issues_file)
        self.logger = get_logger(__name__)
        
        if not self.issues_file.exists():
            raise FileNotFoundError(f"Issues file not found: {self.issues_file}")
        
        # Load issues data
        self.issues_data = self._load_issues_data()
        
        self.logger.info(f"Initialized issue processor with {len(self.issues_data['student_issues'])} students")
    
    def _load_issues_data(self) -> Dict[str, Any]:
        """Load issues data from JSON file"""
        try:
            with open(self.issues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded issues data for {data['metadata']['total_students']} students")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to load issues data: {e}")
            raise
    
    def extract_submissions(self, 
                          min_issues: int = 1,
                          max_issues: int = None) -> List[IssueSubmission]:
        """
        Extract all submissions from the issues data.
        
        Args:
            min_issues: Minimum number of issues required per student
            max_issues: Maximum number of issues per student (None = no limit)
            
        Returns:
            List of IssueSubmission objects
        """
        submissions = []
        
        for student_name, issues in self.issues_data['student_issues'].items():
            # Filter by minimum issues
            if len(issues) < min_issues:
                self.logger.debug(f"Skipping {student_name} - only {len(issues)} issues")
                continue
            
            # Limit maximum issues if specified
            processed_issues = issues
            if max_issues and len(issues) > max_issues:
                processed_issues = issues[:max_issues]
                self.logger.debug(f"Truncated {student_name} issues from {len(issues)} to {max_issues}")
            
            # Create metadata
            metadata = {
                'original_issue_count': len(issues),
                'processed_issue_count': len(processed_issues),
                'issues_file': str(self.issues_file),
                'unique_issues': len(set(processed_issues))
            }
            
            submission = IssueSubmission(
                student_name=student_name,
                issues=processed_issues,
                issue_count=len(processed_issues),
                metadata=metadata
            )
            
            submissions.append(submission)
            self.logger.debug(f"Processed submission: {student_name} with {len(processed_issues)} issues")
        
        self.logger.info(f"Successfully processed {len(submissions)} issue-based submissions")
        return submissions
    
    def get_task_statistics(self, submissions: List[IssueSubmission]) -> Dict[str, Any]:
        """
        Get statistics about processed submissions.
        
        Args:
            submissions: List of submissions
            
        Returns:
            Dictionary with statistics
        """
        if not submissions:
            return {
                'issues_file': str(self.issues_file),
                'total_submissions': 0,
                'avg_issues_per_submission': 0,
                'total_issues': 0,
                'unique_issues_per_student': 0
            }
        
        total_issues = sum(sub.issue_count for sub in submissions)
        all_issues = []
        unique_issues_per_student = []
        
        for sub in submissions:
            all_issues.extend(sub.issues)
            unique_issues_per_student.append(sub.metadata['unique_issues'])
        
        return {
            'issues_file': str(self.issues_file),
            'total_submissions': len(submissions),
            'avg_issues_per_submission': total_issues / len(submissions),
            'median_issues_per_submission': sorted([sub.issue_count for sub in submissions])[len(submissions)//2],
            'max_issues_per_submission': max(sub.issue_count for sub in submissions),
            'min_issues_per_submission': min(sub.issue_count for sub in submissions),
            'total_issues': total_issues,
            'total_unique_issues': len(set(all_issues)),
            'avg_unique_issues_per_student': sum(unique_issues_per_student) / len(unique_issues_per_student),
            'student_names': [sub.student_name for sub in submissions]
        }
    
    def filter_submissions(self, submissions: List[IssueSubmission], 
                         min_issues: int = 1,
                         max_issues: Optional[int] = None,
                         student_names: Optional[List[str]] = None) -> List[IssueSubmission]:
        """
        Filter submissions based on criteria.
        
        Args:
            submissions: List of submissions to filter
            min_issues: Minimum number of issues required
            max_issues: Maximum number of issues allowed
            student_names: Optional list of student names to include
            
        Returns:
            Filtered list of submissions
        """
        filtered = []
        
        for submission in submissions:
            # Check minimum issues requirement
            if submission.issue_count < min_issues:
                continue
            
            # Check maximum issues requirement
            if max_issues and submission.issue_count > max_issues:
                continue
            
            # Check student name filter
            if student_names and submission.student_name not in student_names:
                continue
            
            filtered.append(submission)
        
        self.logger.info(f"Filtered {len(submissions)} -> {len(filtered)} submissions")
        return filtered
    
    def analyze_issue_patterns(self, submissions: List[IssueSubmission]) -> Dict[str, Any]:
        """
        Analyze patterns in the issues across submissions.
        
        Args:
            submissions: List of submissions
            
        Returns:
            Dictionary with issue pattern analysis
        """
        from collections import defaultdict
        
        issue_counts = defaultdict(int)
        student_issue_matrix = {}
        
        # Count issue frequencies
        for submission in submissions:
            student_issue_matrix[submission.student_name] = submission.issues
            for issue in submission.issues:
                issue_counts[issue] += 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate similarity metrics
        total_unique_issues = len(issue_counts)
        most_common_issues = dict(sorted_issues[:20])
        
        # Calculate issue co-occurrence
        issue_cooccurrence = defaultdict(int)
        for submission in submissions:
            issues_set = set(submission.issues)
            for issue1 in issues_set:
                for issue2 in issues_set:
                    if issue1 != issue2:
                        pair = tuple(sorted([issue1, issue2]))
                        issue_cooccurrence[pair] += 1
        
        top_cooccurrences = sorted(issue_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_submissions': len(submissions),
            'total_unique_issues': total_unique_issues,
            'most_common_issues': most_common_issues,
            'issue_distribution': {
                'mean_frequency': sum(issue_counts.values()) / len(issue_counts),
                'median_frequency': sorted(issue_counts.values())[len(issue_counts)//2],
                'max_frequency': max(issue_counts.values()),
                'min_frequency': min(issue_counts.values())
            },
            'top_issue_cooccurrences': [
                {'issues': list(pair), 'frequency': freq} 
                for pair, freq in top_cooccurrences
            ]
        } 