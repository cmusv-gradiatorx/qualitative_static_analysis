"""
Semgrep Static Code Analyzer

This module provides static code analysis using Semgrep rules.
Implements the Strategy pattern for different analysis approaches.

Author: Auto-generated
"""

import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger


@dataclass
class SemgrepFinding:
    """Data class representing a Semgrep finding."""
    rule_id: str
    message: str
    severity: str
    file_path: str
    line: int
    column: int
    code_snippet: str


class SemgrepAnalyzer:
    """
    Static code analyzer using Semgrep rules.
    
    This class implements the Strategy pattern for pluggable static analysis.
    It processes code using configurable Semgrep rules and returns structured findings.
    """
    
    def __init__(self, rules_file: Path, timeout: int = 300):
        """
        Initialize the Semgrep analyzer.
        
        Args:
            rules_file: Path to the Semgrep rules YAML file
            timeout: Timeout for Semgrep analysis in seconds
        """
        self.rules_file = rules_file
        self.timeout = timeout
        self.logger = get_logger(__name__)
        
        # Validate rules file exists
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Semgrep rules file not found: {self.rules_file}")
        
        self.logger.info(f"SemgrepAnalyzer initialized with rules: {self.rules_file}")
    
    def analyze_codebase(self, zip_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Analyze a codebase using Semgrep rules.
        
        Args:
            zip_path: Path to the ZIP file containing the codebase
            temp_dir: Temporary directory for extraction and analysis
            
        Returns:
            Dictionary containing analysis results and findings
            
        Raises:
            Exception: If analysis fails
        """
        self.logger.info(f"Starting Semgrep analysis for: {zip_path.name}")
        
        try:
            # Extract ZIP file
            extracted_dir = self._extract_codebase(zip_path, temp_dir)
            
            # Run Semgrep analysis
            findings = self._run_semgrep_analysis(extracted_dir)
            
            # Process and categorize findings
            processed_findings = self._process_findings(findings, extracted_dir)
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(processed_findings)
            
            self.logger.info(f"Semgrep analysis complete. Found {len(processed_findings)} issues")
            
            return {
                'success': True,
                'findings_count': len(processed_findings),
                'findings': processed_findings,
                'summary': summary,
                'project_name': zip_path.stem
            }
            
        except Exception as e:
            self.logger.error(f"Semgrep analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'findings_count': 0,
                'findings': [],
                'summary': {},
                'project_name': zip_path.stem
            }
    
    def _extract_codebase(self, zip_path: Path, temp_dir: Path) -> Path:
        """
        Extract ZIP file to temporary directory.
        
        Args:
            zip_path: Path to ZIP file
            temp_dir: Temporary directory for extraction
            
        Returns:
            Path to extracted directory
        """
        import zipfile
        
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Extracting {zip_path.name} to {extracted_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        
        return extracted_dir
    
    def _run_semgrep_analysis(self, codebase_dir: Path) -> List[Dict[str, Any]]:
        """
        Run Semgrep analysis on the extracted codebase.
        
        Args:
            codebase_dir: Directory containing the extracted codebase
            
        Returns:
            List of raw Semgrep findings
            
        Raises:
            subprocess.CalledProcessError: If Semgrep execution fails
            json.JSONDecodeError: If Semgrep output is invalid JSON
        """
        self.logger.info("Running Semgrep analysis...")
        
        # Prepare Semgrep command
        cmd = [
            'semgrep',
            '--config', str(self.rules_file),
            '--json',
            '--no-git-ignore',
            '--timeout', str(self.timeout),
            str(codebase_dir)
        ]
        
        self.logger.debug(f"Semgrep command: {' '.join(cmd)}")
        
        try:
            # Run Semgrep
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False  # Don't raise exception on non-zero exit code
            )
            
            # Log command output for debugging
            if result.stderr:
                self.logger.debug(f"Semgrep stderr: {result.stderr}")
            
            # Parse JSON output
            if result.stdout:
                findings_data = json.loads(result.stdout)
                findings = findings_data.get('results', [])
                self.logger.info(f"Semgrep found {len(findings)} potential issues")
                return findings
            else:
                self.logger.info("Semgrep analysis completed with no findings")
                return []
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Semgrep analysis timed out after {self.timeout} seconds")
            raise Exception(f"Semgrep analysis timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Semgrep JSON output: {str(e)}")
            raise Exception(f"Invalid Semgrep output: {str(e)}")
        except FileNotFoundError:
            self.logger.error("Semgrep not found. Please install semgrep: pip install semgrep")
            raise Exception("Semgrep not installed. Please install with: pip install semgrep")
    
    def _process_findings(self, raw_findings: List[Dict[str, Any]], 
                         codebase_dir: Path) -> List[SemgrepFinding]:
        """
        Process raw Semgrep findings into structured objects.
        
        Args:
            raw_findings: Raw findings from Semgrep
            codebase_dir: Base directory of the analyzed codebase
            
        Returns:
            List of processed SemgrepFinding objects
        """
        processed_findings = []
        
        for finding in raw_findings:
            try:
                # Extract finding details
                rule_id = finding.get('check_id', 'unknown')
                message = finding.get('extra', {}).get('message', 'No message')
                severity = finding.get('extra', {}).get('severity', 'INFO').upper()
                
                # Extract location information
                path_info = finding.get('path', '')
                line = finding.get('start', {}).get('line', 0)
                column = finding.get('start', {}).get('col', 0)
                
                # Get relative path
                try:
                    file_path = Path(path_info).relative_to(codebase_dir)
                except ValueError:
                    file_path = Path(path_info)
                
                # Extract code snippet
                code_snippet = finding.get('extra', {}).get('lines', '')
                if not code_snippet:
                    # Try to read the actual code snippet from the file
                    code_snippet = self._extract_code_snippet(
                        codebase_dir / file_path, line
                    )
                
                processed_finding = SemgrepFinding(
                    rule_id=rule_id,
                    message=message,
                    severity=severity,
                    file_path=str(file_path),
                    line=line,
                    column=column,
                    code_snippet=code_snippet
                )
                
                processed_findings.append(processed_finding)
                
            except Exception as e:
                self.logger.warning(f"Failed to process finding: {str(e)}")
                continue
        
        return processed_findings
    
    def _extract_code_snippet(self, file_path: Path, line_number: int, 
                            context_lines: int = 2) -> str:
        """
        Extract code snippet from file around the specified line.
        
        Args:
            file_path: Path to the source file
            line_number: Line number of the finding
            context_lines: Number of context lines to include
            
        Returns:
            Code snippet as string
        """
        try:
            if not file_path.exists():
                return "Code snippet not available"
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start_line = max(0, line_number - context_lines - 1)
            end_line = min(len(lines), line_number + context_lines)
            
            snippet_lines = []
            for i in range(start_line, end_line):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{prefix}{i+1:4}: {lines[i].rstrip()}")
            
            return '\n'.join(snippet_lines)
            
        except Exception as e:
            self.logger.debug(f"Failed to extract code snippet: {str(e)}")
            return "Code snippet not available"
    
    def _generate_analysis_summary(self, findings: List[SemgrepFinding]) -> Dict[str, Any]:
        """
        Generate analysis summary from findings.
        
        Args:
            findings: List of processed findings
            
        Returns:
            Dictionary containing analysis summary
        """
        summary = {
            'total_findings': len(findings),
            'by_severity': {},
            'by_rule': {},
            'by_file': {}
        }
        
        # Count by severity
        for finding in findings:
            severity = finding.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        # Count by rule
        for finding in findings:
            rule_id = finding.rule_id
            summary['by_rule'][rule_id] = summary['by_rule'].get(rule_id, 0) + 1
        
        # Count by file
        for finding in findings:
            file_path = finding.file_path
            summary['by_file'][file_path] = summary['by_file'].get(file_path, 0) + 1
        
        # Note: Category classification is now handled by the LLM through prompt instructions
        # This keeps the analyzer generic and configurable
        
        return summary 