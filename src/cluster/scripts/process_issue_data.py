#!/usr/bin/env python3
"""
Process Issue Data Script

Processes all JSON files in output/json directory to extract issues for each student.
Creates a mapping of student_name to their issues for clustering purposes.

Usage:
    python src/cluster/scripts/process_issue_data.py \
        --input-dir output/json \
        --output-file src/cluster/data/student_issues.json

Author: Auto-generated
"""

import argparse
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Set
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
        description="Process JSON files to extract student issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process JSON files for default assignment (task4_GildedRoseKata)
  python src/cluster/scripts/process_issue_data.py

  # Process JSON files for a specific assignment
  python src/cluster/scripts/process_issue_data.py --assignment task1_SOLID

  # Process with custom input and output paths
  python src/cluster/scripts/process_issue_data.py \\
      --assignment task4_GildedRoseKata \\
      --input-dir output/custom_task/json \\
      --output-file custom_issues.json
        """
    )
    
    parser.add_argument(
        "--assignment",
        type=str,
        default="task4_GildedRoseKata",
        help="Assignment name (default: task4_GildedRoseKata)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing JSON evaluation files (defaults to output/{assignment}/json)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for student issues mapping (defaults to src/cluster/data/issues/{assignment}_student_issues.json)"
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


def extract_student_name_from_filename(filename: str) -> str:
    """
    Extract student name from JSON filename.
    
    Args:
        filename: JSON filename like 'caitianjun_111836_11501289_SolidPrinciples-1_evaluation_20250613_161818.json'
        
    Returns:
        Student name
    """
    # Remove extension and extract first part (student name)
    base_name = filename.replace('.json', '')
    
    # Extract name before first underscore with numbers
    match = re.match(r'^([a-zA-Z]+)', base_name)
    if match:
        return match.group(1)
    
    # Fallback: split by underscore and take first part
    parts = base_name.split('_')
    if parts:
        return re.sub(r'[^a-zA-Z0-9]', '', parts[0]).lower()
    
    return base_name


def normalize_issue_text(issue: str) -> str:
    """
    Normalize issue text to group similar issues together.
    
    Args:
        issue: Raw issue text
        
    Returns:
        Normalized issue text
    """
    if not issue or not isinstance(issue, str):
        return ""
    
    # Start with the original text
    normalized = issue.strip()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Normalize quotes and backticks
    normalized = re.sub(r'[`""''″‴]', '"', normalized)
    
    # Standardize parentheses patterns
    # Convert "name (method)" to "name: method"
    normalized = re.sub(r'\s*\(\s*([^)]+)\s*\)', r': \1', normalized)
    
    # Standardize "for method" patterns
    normalized = re.sub(r'\s+for\s+[`"]*([^`"]+)[`"]*', r': \1', normalized)
    
    # Remove trailing parentheses that don't contain content
    normalized = re.sub(r'\(\s*\)\s*$', '', normalized)
    
    # Standardize method name patterns
    # Remove quotes around method names in standard positions
    normalized = re.sub(r':\s*["`]([^"`]+)["`]', r': \1', normalized)
    normalized = re.sub(r'name:\s*["`]([^"`]+)["`]', r'name: \1', normalized)
    
    # Standardize common prefixes
    prefixes_map = {
        'Misleading test method name': 'Misleading test name',
        'Misleading test case name': 'Misleading test name',
        'Test method name misleading': 'Misleading test name',
        'Test case name misleading': 'Misleading test name',
        'Misnamed test case': 'Misleading test name',
        'Misnamed test method': 'Misleading test name',
        'Poorly named test': 'Misleading test name',
    }
    
    for old_prefix, new_prefix in prefixes_map.items():
        if normalized.startswith(old_prefix):
            normalized = normalized.replace(old_prefix, new_prefix, 1)
            break
    
    # Standardize specific method names first (more specific patterns)
    method_patterns = {
        r'.*forNormalProductAtEndOfDaySellinAndQualityOfItemAreLowered.*': 'Misleading test name: forNormalProductAtEndOfDaySellinAndQualityOfItemAreLowered',
        r'.*updateQualityForExpiredItems.*': 'Misleading method names: updateQualityForExpiredItems',
        r'.*updateQualityForItemsThatAgeWell.*': 'Misleading method names: updateQualityForItemsThatAgeWell',
        r'.*foo\(\).*test.*': 'Placeholder test',
        r'.*foo.*test.*': 'Placeholder test',
    }
    
    for pattern, replacement in method_patterns.items():
        if re.match(pattern, normalized, re.IGNORECASE):
            # Only apply if it matches the context
            if 'forNormalProductAtEndOfDaySellinAndQualityOfItemAreLowered' in pattern and \
               ('misleading' in normalized.lower() and 'test' in normalized.lower()):
                normalized = replacement
                break
            elif ('updateQualityForExpiredItems' in pattern or 'updateQualityForItemsThatAgeWell' in pattern) and \
                 'method' in normalized.lower():
                normalized = replacement
                break
            elif 'foo' in pattern and ('test' in normalized.lower() or 'placeholder' in normalized.lower()):
                normalized = replacement
                break
    
    # Standardize common patterns (more general)
    patterns_map = {
        r'No use of parameterized tests.*': 'No use of parameterized tests',
        r'No parameterized tests.*': 'No use of parameterized tests',
        r'Lack of parameterized tests.*': 'No use of parameterized tests',
        r'Missing.*parameterized.*': 'No use of parameterized tests',
        
        r'Placeholder test.*': 'Placeholder test',
        r'.*placeholder test.*': 'Placeholder test',
        r'Trivial.*test.*': 'Placeholder test',
        r'Useless.*test.*': 'Placeholder test',
        r'Ineffective.*test.*': 'Placeholder test',
        
        r'Missing method definition.*': 'Missing method definition',
        r'Undefined method call.*': 'Undefined method call',
        
        r'Redundant test (structures?|methods?).*': 'Redundant test structures',
        r'High.*redundancy.*test.*': 'Redundant test structures',
        r'Repetitive test.*': 'Redundant test structures',
    }
    
    for pattern, replacement in patterns_map.items():
        if re.match(pattern, normalized, re.IGNORECASE):
            # For parameterized tests, preserve specific context if it's meaningful
            if 'parameterized' in pattern and any(x in normalized.lower() for x in ['backstage', 'aged brie', 'normal item', 'sulfuras']):
                continue
            normalized = replacement
            break
    
    # Final cleanup
    normalized = re.sub(r'\s*[:]\s*$', '', normalized)  # Remove trailing colons
    normalized = re.sub(r'\s+', ' ', normalized)  # Remove extra spaces again
    normalized = normalized.strip()
    
    return normalized


def process_json_file(json_path: Path, logger) -> Dict[str, List[str]]:
    """
    Process a single JSON file to extract issues.
    
    Args:
        json_path: Path to JSON file
        logger: Logger instance
        
    Returns:
        Dictionary with student_name and their issues
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract student name from filename
        student_name = extract_student_name_from_filename(json_path.name)
        
        # Extract issues from the JSON
        issues = data.get('issues', [])
        
        # Clean, normalize, and deduplicate issues
        normalized_issues = []
        for issue in issues:
            if isinstance(issue, str) and issue.strip():
                normalized_issue = normalize_issue_text(issue)
                if normalized_issue:  # Only add non-empty normalized issues
                    normalized_issues.append(normalized_issue)
        
        # Remove duplicates while preserving order
        cleaned_issues = []
        seen_issues = set()
        for issue in normalized_issues:
            if issue not in seen_issues:
                cleaned_issues.append(issue)
                seen_issues.add(issue)
        
        duplicate_count = len(normalized_issues) - len(cleaned_issues)
        if duplicate_count > 0:
            logger.debug(f"Removed {duplicate_count} duplicate issues for {student_name}")
        
        logger.debug(f"Extracted {len(cleaned_issues)} unique issues for {student_name}")
        
        return {student_name: cleaned_issues}
        
    except Exception as e:
        logger.error(f"Failed to process {json_path.name}: {e}")
        return {}


def analyze_issue_patterns(all_issues: List[str], logger) -> Dict[str, int]:
    """
    Analyze patterns in issues to understand the data.
    
    Args:
        all_issues: List of all issues across students
        logger: Logger instance
        
    Returns:
        Dictionary of issue patterns and their frequencies
    """
    issue_counts = defaultdict(int)
    
    for issue in all_issues:
        issue_counts[issue] += 1
    
    # Sort by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top 15 most common issues after normalization:")
    for issue, count in sorted_issues[:15]:
        logger.info(f"  {count:2d}x: {issue}")
    
    # Show statistics
    total_issues = len(all_issues)
    unique_issues = len(issue_counts)
    logger.info(f"\nIssue consolidation statistics:")
    logger.info(f"  Total issues: {total_issues}")
    logger.info(f"  Unique issues: {unique_issues}")
    logger.info(f"  Consolidation ratio: {total_issues/unique_issues:.2f}:1")
    
    return dict(sorted_issues)


def main():
    """Main processing function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level, args.log_file)
    logger = get_logger(__name__)
    
    logger.info("Starting issue data processing")
    
    try:
        # Set default paths based on assignment if not provided
        if args.input_dir is None:
            input_dir = Path("output") / args.assignment / "json"
        else:
            input_dir = Path(args.input_dir)
            
        if args.output_file is None:
            output_file = Path("src/cluster/data/issues") / f"{args.assignment}_student_issues.json"
        else:
            output_file = Path(args.output_file)
        
        # Validate input directory
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        if not json_files:
            logger.error("No JSON files found")
            sys.exit(1)
        
        # Process all JSON files
        student_issues = {}
        all_issues = []
        normalization_examples = []
        total_raw_issues = 0
        
        for json_file in json_files:
            # Count raw issues for statistics
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                raw_issues = data.get('issues', [])
                total_raw_issues += len([issue for issue in raw_issues if isinstance(issue, str) and issue.strip()])
            except:
                pass
            
            file_results = process_json_file(json_file, logger)
            student_issues.update(file_results)
            
            # Collect all issues for analysis
            for issues in file_results.values():
                all_issues.extend(issues)
            
            # Collect some normalization examples for debugging (first file only)
            if not normalization_examples and args.verbose:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    original_issues = data.get('issues', [])[:5]  # First 5 issues
                    for orig in original_issues:
                        if isinstance(orig, str) and orig.strip():
                            normalized = normalize_issue_text(orig)
                            if orig.strip() != normalized:
                                normalization_examples.append((orig.strip(), normalized))
                except:
                    pass
        
        logger.info(f"Processed {len(student_issues)} students")
        logger.info(f"Raw issues found: {total_raw_issues}")
        logger.info(f"Final unique issues: {len(all_issues)}")
        logger.info(f"Duplicates removed: {total_raw_issues - len(all_issues)}")
        
        # Show normalization examples if verbose
        if normalization_examples and args.verbose:
            logger.info(f"\nNormalization examples:")
            for orig, norm in normalization_examples[:5]:
                logger.info(f"  Original: {orig}")
                logger.info(f"  Normalized: {norm}")
                logger.info(f"")
        
        # Analyze issue patterns
        issue_patterns = analyze_issue_patterns(all_issues, logger)
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_data = {
            'metadata': {
                'assignment': args.assignment,
                'total_students': len(student_issues),
                'raw_issues_found': total_raw_issues,
                'total_issues': len(all_issues),
                'duplicates_removed': total_raw_issues - len(all_issues),
                'unique_issues': len(set(all_issues)),
                'input_directory': str(input_dir),
                'processing_timestamp': 'generated_timestamp'
            },
            'student_issues': student_issues,
            'issue_statistics': {
                'most_common_issues': dict(list(issue_patterns.items())[:20]),
                'total_unique_issues': len(set(all_issues))
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 ISSUE DATA PROCESSING COMPLETED")
        print("="*80)
        print(f"📋 Assignment: {args.assignment}")
        print(f"📁 Input directory: {input_dir}")
        print(f"📂 Output file: {output_file}")
        print(f"👥 Students processed: {len(student_issues)}")
        print(f"📊 Raw issues found: {total_raw_issues}")
        print(f"📋 Total unique issues: {len(all_issues)}")
        print(f"🗑️ Duplicates removed: {total_raw_issues - len(all_issues)}")
        print(f"🔍 Issue consolidation ratio: {total_raw_issues/len(all_issues) if len(all_issues) > 0 else 0:.2f}:1")
        
        # Show sample of student issues
        print(f"\n📝 SAMPLE STUDENT ISSUES:")
        for i, (student, issues) in enumerate(list(student_issues.items())[:5]):
            print(f"  {student}: {len(issues)} issues")
            for issue in issues[:2]:  # Show first 2 issues
                print(f"    - {issue}")
            if len(issues) > 2:
                print(f"    ... and {len(issues) - 2} more")
        
        if len(student_issues) > 5:
            print(f"  ... and {len(student_issues) - 5} more students")
        
        print(f"\n🔧 NEXT STEPS:")
        print("# Use the generated data for clustering:")
        print(f"python src/cluster/scripts/train_clustering.py \\")
        print(f"    dummy_folder \\")
        print(f"    --embedder-type issues \\")
        print(f"    --issues-file {output_file}")
        
        print("="*80)
        
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