#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering-Based Grading System

This module provides an alternative to FAISS similarity search by using clustering
to categorize student submissions based on scores and common issues. Uses the existing
StarCoder2 embeddings but applies sklearn clustering algorithms instead of vector search.

Author: Auto-generated
"""

import numpy as np
import pandas as pd
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import re

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from ..processor import Submission, MultiFolderProcessor
from ..embedder import create_java_embedder
from ...utils.logger import get_logger


class IssueExtractor:
    """Extract issue categories from instructor feedback text"""
    
    def __init__(self):
        self.issue_patterns = {
            'magic_numbers': [
                r'magic number', r'sellIn < 0', r'sellIn < 11', r'TEN_DAYS',
                r'SELLIN_THRESHOLD', r'hard.?coded', r'constant'
            ],
            'conceptual_errors': [
                r'MIN_QUALITY', r'distinct concerns', r'separate concerns',
                r'wrong constant', r'conceptual', r'misunderstand'
            ],
            'class_adherence': [
                r'class', r'refactoring we did', r'follow.*pattern',
                r'taught', r'lesson', r'lecture'
            ],
            'compilation_errors': [
                r'not compile', r'compilation', r'missing.*method',
                r'syntax error', r'build error'
            ],
            'code_quality': [
                r'readable', r'maintainable', r'clean', r'structure',
                r'organization', r'style'
            ],
            'missing_implementation': [
                r'missing', r'incomplete', r'not implemented',
                r'no.*method', r'absent'
            ]
        }
    
    def extract_issues(self, feedback: str) -> Dict[str, bool]:
        """Extract binary issue indicators from feedback text"""
        feedback_lower = feedback.lower()
        issues = {}
        
        for issue_type, patterns in self.issue_patterns.items():
            issues[issue_type] = any(
                re.search(pattern, feedback_lower) for pattern in patterns
            )
        
        return issues
    
    def get_issue_vector(self, feedback: str) -> np.ndarray:
        """Convert feedback to binary issue vector"""
        issues = self.extract_issues(feedback)
        return np.array([1 if issues[key] else 0 for key in self.issue_patterns.keys()])
    
    def get_issue_names(self) -> List[str]:
        """Get ordered list of issue category names"""
        return list(self.issue_patterns.keys())


class ClusteringManager:
    """Manages 2-layer clustering for assignment grading"""
    
    def __init__(self, assignment_id: str, embedder=None):
        """
        Initialize clustering manager for a specific assignment.
        
        Args:
            assignment_id: Assignment identifier
            embedder: Java code embedder (will create if None)
        """
        self.assignment_id = assignment_id
        self.embedder = embedder
        self.logger = get_logger(__name__)
        
        # Clustering models
        self.score_clusterer = None
        self.issue_clusterer = None
        self.scaler = StandardScaler()
        
        # Data storage
        self.submissions = []
        self.embeddings = None
        self.scores = None
        self.issue_vectors = None
        
        # Issue extraction
        self.issue_extractor = IssueExtractor()
        
        # Cluster metadata
        self.score_cluster_info = {}
        self.issue_cluster_info = {}
        
        self.logger.info(f"Clustering manager initialized for assignment: {assignment_id}")
    
    def load_submissions_and_grades(self, submissions: List[Submission], 
                                  grade_mapping_csv: str) -> None:
        """
        Load submissions and grade mapping data.
        
        Args:
            submissions: List of submission objects with code
            grade_mapping_csv: Path to CSV with student_name, grade, feedback
        """
        self.logger.info(f"Loading submissions and grades for {self.assignment_id}")
        
        # Load grade mapping with proper CSV parsing settings
        try:
            grade_df = pd.read_csv(
                grade_mapping_csv,
                skipinitialspace=True,  # Skip whitespace after delimiter
                quotechar='"',          # Handle quoted strings properly
                escapechar='\\',        # Handle escape characters
                on_bad_lines='warn'     # Warn about bad lines but continue
            )
            grade_df.columns = grade_df.columns.str.strip()  # Remove whitespace
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {grade_mapping_csv}: {e}")
            # Try alternative parsing method
            try:
                self.logger.info("Trying alternative CSV parsing...")
                grade_df = pd.read_csv(
                    grade_mapping_csv,
                    sep=',',
                    skipinitialspace=True,
                    engine='python'  # Python engine is more forgiving
                )
                grade_df.columns = grade_df.columns.str.strip()
            except Exception as e2:
                self.logger.error(f"Alternative CSV parsing also failed: {e2}")
                raise ValueError(f"Cannot parse CSV file {grade_mapping_csv}. Please check the file format.")
        
        # Log the first few rows to understand the data structure
        self.logger.debug(f"CSV columns: {list(grade_df.columns)}")
        self.logger.debug(f"First 3 rows:\n{grade_df.head(3)}")
        
        # Create grade mapping dictionary
        grade_mapping = {}
        skipped_rows = 0
        
        for idx, row in grade_df.iterrows():
            try:
                student_name = str(row['student_name']).strip()
                
                # Handle various grade formats and validation
                grade_str = str(row['grade']).strip()
                
                # Skip empty or invalid rows
                if not student_name or student_name.lower() in ['nan', 'none', '']:
                    self.logger.debug(f"Skipping row {idx}: empty student name")
                    skipped_rows += 1
                    continue
                
                # Try to convert grade to float, with error handling
                try:
                    # Handle common non-numeric cases
                    if grade_str.lower() in ['nan', 'none', '', 'n/a']:
                        self.logger.warning(f"Skipping row {idx}: no grade for {student_name}")
                        skipped_rows += 1
                        continue
                    
                    # Remove any non-numeric characters and try conversion
                    grade_cleaned = ''.join(c for c in grade_str if c.isdigit() or c in '.-')
                    if not grade_cleaned:
                        raise ValueError(f"No numeric content in grade: '{grade_str}'")
                    
                    grade = float(grade_cleaned)
                    
                    # Validate grade range (assuming 0-1 scale)
                    if grade < 0 or grade > 1:
                        self.logger.warning(f"Grade {grade} for {student_name} outside expected range [0,1]. Using as-is.")
                    
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Cannot convert grade '{grade_str}' to float for student {student_name} (row {idx}): {e}")
                    skipped_rows += 1
                    continue
                
                feedback = str(row['feedback']).strip()
                
                # Handle missing feedback
                if feedback.lower() in ['nan', 'none', '']:
                    feedback = "No feedback provided"
                
                grade_mapping[student_name] = {'grade': grade, 'feedback': feedback}
                self.logger.debug(f"Loaded grade {grade} for student {student_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                self.logger.debug(f"Row data: {dict(row)}")
                skipped_rows += 1
                continue
        
        # Log statistics about grade loading
        self.logger.info(f"Loaded grades for {len(grade_mapping)} students")
        if skipped_rows > 0:
            self.logger.warning(f"Skipped {skipped_rows} invalid rows in CSV")
        
        # Filter submissions to only those with grades
        matched_submissions = []
        unmatched_students = []
        
        for submission in submissions:
            # Extract student name from submission file name
            student_name = self._extract_student_name(submission.file_name)
            
            if student_name in grade_mapping:
                submission.score = grade_mapping[student_name]['grade']
                submission.feedback = grade_mapping[student_name]['feedback']
                matched_submissions.append(submission)
            else:
                unmatched_students.append(student_name)
        
        # Log detailed matching information
        if unmatched_students:
            self.logger.warning(f"No grades found for {len(unmatched_students)} students:")
            for student in unmatched_students[:10]:  # Show first 10
                self.logger.warning(f"  - {student}")
            if len(unmatched_students) > 10:
                self.logger.warning(f"  ... and {len(unmatched_students) - 10} more")
        
        self.submissions = matched_submissions
        self.logger.info(f"Successfully matched {len(matched_submissions)} submissions with grades")
        
        if len(matched_submissions) == 0:
            available_names = list(grade_mapping.keys())[:10]
            self.logger.error("No submissions matched with grades!")
            self.logger.error(f"Available student names in CSV: {available_names}")
            self.logger.error(f"Sample submission file names: {[s.file_name for s in submissions[:5]]}")
            raise ValueError("No submissions could be matched with grades. Check student name extraction logic.")
    
    def _extract_student_name(self, file_name: str) -> str:
        """Extract student name from submission file name"""
        # Remove common file extensions and path separators
        base_name = file_name.replace('.zip', '').replace('.rar', '')
        base_name = base_name.split('/')[-1].split('\\')[-1]
        
        # Extract first part (usually student name)
        if '_' in base_name:
            return base_name.split('_')[0]
        return base_name
    
    def generate_embeddings(self) -> None:
        """Generate StarCoder2 embeddings for all submissions"""
        if not self.embedder:
            self.logger.info("Creating Java embedder...")
            self.embedder = create_java_embedder(
                model_name="starcoder2:15b",
                use_ollama=True
            )
        
        self.logger.info(f"Generating embeddings for {len(self.submissions)} submissions")
        
        embeddings = []
        scores = []
        issue_vectors = []
        
        for i, submission in enumerate(self.submissions):
            try:
                self.logger.debug(f"Processing {i+1}/{len(self.submissions)}: {submission.file_name}")
                
                # Generate code embedding
                embedding = self.embedder.embed_codebase(submission.code_files)
                embeddings.append(embedding)
                
                # Extract score
                scores.append(submission.score)
                
                # Extract issue vector from feedback
                issue_vector = self.issue_extractor.get_issue_vector(submission.feedback)
                issue_vectors.append(issue_vector)
                
            except Exception as e:
                self.logger.error(f"Failed to process {submission.file_name}: {e}")
                continue
        
        self.embeddings = np.vstack(embeddings)
        self.scores = np.array(scores)
        self.issue_vectors = np.vstack(issue_vectors)
        
        self.logger.info(f"Generated embeddings: {self.embeddings.shape}")
        self.logger.info(f"Score distribution: {np.unique(self.scores, return_counts=True)}")
    
    def train_score_clustering(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Train score-based clustering model.
        
        Args:
            n_clusters: Number of clusters (auto-detect from unique scores if None)
        """
        if self.embeddings is None:
            raise ValueError("Must generate embeddings first")
        
        # Determine number of clusters from unique scores
        unique_scores = np.unique(self.scores)
        if n_clusters is None:
            n_clusters = len(unique_scores)
        
        self.logger.info(f"Training score clustering with {n_clusters} clusters")
        
        # Scale embeddings for better clustering
        scaled_embeddings = self.scaler.fit_transform(self.embeddings)
        
        # Train K-means clustering
        self.score_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        score_cluster_labels = self.score_clusterer.fit_predict(scaled_embeddings)
        
        # Analyze cluster composition
        cluster_analysis = self._analyze_score_clusters(score_cluster_labels)
        
        # Calculate clustering metrics
        if n_clusters > 1:
            silhouette = silhouette_score(scaled_embeddings, score_cluster_labels)
        else:
            silhouette = 0.0
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'cluster_analysis': cluster_analysis,
            'unique_scores': unique_scores.tolist()
        }
        
        self.logger.info(f"Score clustering trained - Silhouette: {silhouette:.3f}")
        return results
    
    def train_issue_clustering(self, algorithm: str = 'kmeans') -> Dict[str, Any]:
        """
        Train issue-based clustering model.
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        """
        if self.issue_vectors is None:
            raise ValueError("Must generate embeddings first")
        
        self.logger.info(f"Training issue clustering with {algorithm}")
        
        # Combine code embeddings with issue vectors for richer clustering
        scaled_embeddings = self.scaler.transform(self.embeddings)
        combined_features = np.hstack([scaled_embeddings, self.issue_vectors * 10])  # Weight issue features
        
        if algorithm == 'kmeans':
            # Use number of distinct issue patterns
            n_clusters = min(6, len(self.submissions) // 3)  # 6 issue types, but limit by data size
            self.issue_clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            
        elif algorithm == 'dbscan':
            self.issue_clusterer = DBSCAN(eps=0.5, min_samples=2)
            
        elif algorithm == 'hierarchical':
            n_clusters = min(6, len(self.submissions) // 3)
            self.issue_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        issue_cluster_labels = self.issue_clusterer.fit_predict(combined_features)
        
        # Analyze issue clusters
        cluster_analysis = self._analyze_issue_clusters(issue_cluster_labels)
        
        # Calculate metrics
        if len(np.unique(issue_cluster_labels)) > 1:
            silhouette = silhouette_score(combined_features, issue_cluster_labels)
        else:
            silhouette = 0.0
        
        results = {
            'algorithm': algorithm,
            'n_clusters': len(np.unique(issue_cluster_labels)),
            'silhouette_score': silhouette,
            'cluster_analysis': cluster_analysis
        }
        
        self.logger.info(f"Issue clustering trained - Silhouette: {silhouette:.3f}")
        return results
    
    def _analyze_score_clusters(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze score-based cluster composition"""
        analysis = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_scores = self.scores[cluster_mask]
            cluster_submissions = [self.submissions[i] for i in np.where(cluster_mask)[0]]
            
            analysis[f"cluster_{cluster_id}"] = {
                'size': int(np.sum(cluster_mask)),
                'score_range': [float(np.min(cluster_scores)), float(np.max(cluster_scores))],
                'avg_score': float(np.mean(cluster_scores)),
                'dominant_score': float(np.bincount([int(s*100) for s in cluster_scores]).argmax() / 100),
                'student_names': [self._extract_student_name(sub.file_name) for sub in cluster_submissions[:5]]
            }
        
        self.score_cluster_info = analysis
        return analysis
    
    def _analyze_issue_clusters(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze issue-based cluster composition"""
        analysis = {}
        issue_names = self.issue_extractor.get_issue_names()
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # DBSCAN noise
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_issues = self.issue_vectors[cluster_mask]
            cluster_submissions = [self.submissions[i] for i in np.where(cluster_mask)[0]]
            
            # Calculate issue frequencies in this cluster
            issue_frequencies = np.mean(cluster_issues, axis=0)
            dominant_issues = [issue_names[i] for i, freq in enumerate(issue_frequencies) if freq > 0.5]
            
            analysis[f"cluster_{cluster_id}"] = {
                'size': int(np.sum(cluster_mask)),
                'dominant_issues': dominant_issues,
                'issue_frequencies': {issue_names[i]: float(freq) for i, freq in enumerate(issue_frequencies)},
                'avg_score': float(np.mean([sub.score for sub in cluster_submissions])),
                'sample_feedback': cluster_submissions[0].feedback[:200] + "..." if cluster_submissions else ""
            }
        
        self.issue_cluster_info = analysis
        return analysis
    
    def predict_clusters(self, submission: Submission) -> Dict[str, Any]:
        """
        Predict cluster assignments for a new submission.
        
        Args:
            submission: New submission to classify
            
        Returns:
            Dictionary with cluster predictions and metadata
        """
        if not self.score_clusterer or not self.issue_clusterer:
            raise ValueError("Models must be trained first")
        
        # Generate embedding for new submission
        if not self.embedder:
            self.embedder = create_java_embedder(model_name="starcoder2:15b", use_ollama=True)
        
        embedding = self.embedder.embed_codebase(submission.code_files)
        scaled_embedding = self.scaler.transform(embedding.reshape(1, -1))
        
        # Predict score cluster
        score_cluster = self.score_clusterer.predict(scaled_embedding)[0]
        
        # For issue clustering, we need to estimate issue vector
        # In practice, you might want to use the LLM to pre-analyze the code
        # For now, we'll use a zero vector and let the code embedding dominate
        dummy_issue_vector = np.zeros((1, len(self.issue_extractor.get_issue_names())))
        combined_features = np.hstack([scaled_embedding, dummy_issue_vector * 10])
        
        if hasattr(self.issue_clusterer, 'predict'):
            issue_cluster = self.issue_clusterer.predict(combined_features)[0]
        else:
            # For DBSCAN, we need to fit_predict (not ideal for new data)
            issue_cluster = -1  # Unknown
        
        return {
            'score_cluster': int(score_cluster),
            'issue_cluster': int(issue_cluster),
            'score_cluster_info': self.score_cluster_info.get(f"cluster_{score_cluster}", {}),
            'issue_cluster_info': self.issue_cluster_info.get(f"cluster_{issue_cluster}", {}),
            'predicted_score_range': self.score_cluster_info.get(f"cluster_{score_cluster}", {}).get('score_range', [0, 1])
        }
    
    def save_models(self, save_dir: str) -> None:
        """Save trained clustering models and metadata"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn models
        if self.score_clusterer:
            joblib.dump(self.score_clusterer, save_path / "score_clusterer.pkl")
        if self.issue_clusterer:
            joblib.dump(self.issue_clusterer, save_path / "issue_clusterer.pkl")
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        # Save metadata
        metadata = {
            'assignment_id': self.assignment_id,
            'n_submissions': len(self.submissions),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None,
            'score_cluster_info': self.score_cluster_info,
            'issue_cluster_info': self.issue_cluster_info,
            'issue_names': self.issue_extractor.get_issue_names(),
            'train_timestamp': datetime.now().isoformat()
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Models saved to {save_path}")
    
    def load_models(self, save_dir: str) -> None:
        """Load trained clustering models and metadata"""
        save_path = Path(save_dir)
        
        # Load sklearn models
        if (save_path / "score_clusterer.pkl").exists():
            self.score_clusterer = joblib.load(save_path / "score_clusterer.pkl")
        if (save_path / "issue_clusterer.pkl").exists():
            self.issue_clusterer = joblib.load(save_path / "issue_clusterer.pkl")
        self.scaler = joblib.load(save_path / "scaler.pkl")
        
        # Load metadata
        with open(save_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.score_cluster_info = metadata['score_cluster_info']
            self.issue_cluster_info = metadata['issue_cluster_info']
        
        self.logger.info(f"Models loaded from {save_path}")
    
    def get_cluster_summary(self) -> str:
        """Get human-readable summary of clusters for LLM context"""
        summary = f"**Clustering Summary for {self.assignment_id}**\n\n"
        
        # Score clusters
        summary += "**Score-Based Clusters:**\n"
        for cluster_id, info in self.score_cluster_info.items():
            avg_score = info['avg_score']
            size = info['size']
            score_range = info['score_range']
            summary += f"- {cluster_id}: {size} submissions, avg score {avg_score:.2f} (range: {score_range[0]:.2f}-{score_range[1]:.2f})\n"
        
        summary += "\n**Issue-Based Clusters:**\n"
        for cluster_id, info in self.issue_cluster_info.items():
            size = info['size']
            issues = ", ".join(info['dominant_issues'])
            summary += f"- {cluster_id}: {size} submissions, common issues: {issues}\n"
        
        return summary 