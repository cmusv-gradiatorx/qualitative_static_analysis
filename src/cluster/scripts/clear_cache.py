#!/usr/bin/env python3
"""
Clear Clustering Cache Script

Clears cached embeddings for the clustering system to force regeneration.
Can clear cache for specific embedders or all embedders.

Usage:
    python src/cluster/scripts/clear_cache.py --all
    python src/cluster/scripts/clear_cache.py --embedder-type java
    python src/cluster/scripts/clear_cache.py --cache-dir .cache/cluster

Author: Auto-generated
"""

import argparse
import sys
import shutil
import os
from pathlib import Path
from typing import Optional
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
        description="Clear cached embeddings for the clustering system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear all cached embeddings
  python src/cluster/scripts/clear_cache.py --all

  # Clear cache for specific embedder type
  python src/cluster/scripts/clear_cache.py --embedder-type java

  # Clear cache for specific embedder type
  python src/cluster/scripts/clear_cache.py --embedder-type repomix

  # Clear cache in specific directory
  python src/cluster/scripts/clear_cache.py --cache-dir .cache/cluster --all

  # Clear cache for specific student
  python src/cluster/scripts/clear_cache.py --student-name john_doe
        """
    )
    
    # Main options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all cached embeddings"
    )
    
    parser.add_argument(
        "--embedder-type",
        type=str,
        choices=["java", "repomix"],
        help="Clear cache for specific embedder type only"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory path (uses environment default if not provided)"
    )
    
    parser.add_argument(
        "--student-name",
        type=str,
        help="Clear cache for specific student only"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Clear cache for specific model only"
    )
    
    # Confirmation
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Logging
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


def get_cache_directory(cache_dir: Optional[str] = None) -> Path:
    """Get the cache directory path"""
    if cache_dir:
        return Path(cache_dir)
    
    # Use environment variable or default
    cache_dir_env = os.environ.get('CLUSTER_CACHE_DIR', '.cache/cluster')
    return Path(cache_dir_env)


def clear_all_cache(cache_dir: Path, logger) -> int:
    """Clear all cached embeddings"""
    if not cache_dir.exists():
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return 0
    
    files_removed = 0
    for item in cache_dir.rglob('*'):
        if item.is_file():
            try:
                item.unlink()
                files_removed += 1
                logger.debug(f"Removed: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    # Remove empty directories
    for item in cache_dir.rglob('*'):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
                logger.debug(f"Removed empty directory: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {item}: {e}")
    
    return files_removed


def clear_embedder_cache(cache_dir: Path, embedder_type: str, logger) -> int:
    """Clear cache for specific embedder type"""
    embedder_cache_dirs = {
        'java': 'JavaEmbedder',
        'repomix': 'RepomixEmbedder'
    }
    
    embedder_dir_name = embedder_cache_dirs.get(embedder_type.lower())
    if not embedder_dir_name:
        logger.error(f"Unknown embedder type: {embedder_type}")
        return 0
    
    embedder_cache_dir = cache_dir / embedder_dir_name
    
    if not embedder_cache_dir.exists():
        logger.info(f"Cache directory for {embedder_type} does not exist: {embedder_cache_dir}")
        return 0
    
    files_removed = 0
    for item in embedder_cache_dir.rglob('*'):
        if item.is_file():
            try:
                item.unlink()
                files_removed += 1
                logger.debug(f"Removed: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    # Remove the embedder directory if empty
    if not any(embedder_cache_dir.iterdir()):
        try:
            embedder_cache_dir.rmdir()
            logger.debug(f"Removed empty directory: {embedder_cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove directory {embedder_cache_dir}: {e}")
    
    return files_removed


def clear_student_cache(cache_dir: Path, student_name: str, logger) -> int:
    """Clear cache for specific student"""
    files_removed = 0
    
    # Search for files containing the student name
    for item in cache_dir.rglob('*'):
        if item.is_file() and student_name in item.name:
            try:
                item.unlink()
                files_removed += 1
                logger.debug(f"Removed: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    return files_removed


def clear_model_cache(cache_dir: Path, model_name: str, logger) -> int:
    """Clear cache for specific model"""
    files_removed = 0
    
    # Convert model name to filename format (replace : with _)
    model_filename = model_name.replace(':', '_')
    
    # Search for files containing the model name
    for item in cache_dir.rglob('*'):
        if item.is_file() and model_filename in item.name:
            try:
                item.unlink()
                files_removed += 1
                logger.debug(f"Removed: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    return files_removed


def get_cache_statistics(cache_dir: Path) -> dict:
    """Get statistics about cached files"""
    if not cache_dir.exists():
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'embedder_breakdown': {}
        }
    
    total_files = 0
    total_size = 0
    embedder_breakdown = {}
    
    for item in cache_dir.rglob('*'):
        if item.is_file():
            total_files += 1
            total_size += item.stat().st_size
            
            # Get embedder type from path
            embedder_name = None
            for part in item.parts:
                if 'Embedder' in part:
                    embedder_name = part
                    break
            
            if embedder_name:
                if embedder_name not in embedder_breakdown:
                    embedder_breakdown[embedder_name] = {'files': 0, 'size_mb': 0}
                embedder_breakdown[embedder_name]['files'] += 1
                embedder_breakdown[embedder_name]['size_mb'] += item.stat().st_size / (1024 * 1024)
    
    return {
        'total_files': total_files,
        'total_size_mb': total_size / (1024 * 1024),
        'embedder_breakdown': embedder_breakdown
    }


def main():
    """Main cache clearing function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level, args.log_file)
    logger = get_logger(__name__)
    
    logger.info("Starting cache clearing operation")
    
    try:
        # Get cache directory
        cache_dir = get_cache_directory(args.cache_dir)
        logger.info(f"Cache directory: {cache_dir}")
        
        # Get cache statistics before clearing
        stats_before = get_cache_statistics(cache_dir)
        
        if stats_before['total_files'] == 0:
            print("🗂️  No cached files found")
            logger.info("No cached files to clear")
            return
        
        # Show current cache status
        print(f"\n📊 CURRENT CACHE STATUS")
        print(f"📁 Cache directory: {cache_dir}")
        print(f"📄 Total files: {stats_before['total_files']}")
        print(f"💾 Total size: {stats_before['total_size_mb']:.2f} MB")
        
        if stats_before['embedder_breakdown']:
            print(f"\n🤖 BY EMBEDDER TYPE:")
            for embedder, data in stats_before['embedder_breakdown'].items():
                print(f"  {embedder}: {data['files']} files ({data['size_mb']:.2f} MB)")
        
        # Confirmation unless --force is used
        if not args.force:
            if args.all:
                confirm_msg = "Are you sure you want to clear ALL cached embeddings?"
            elif args.embedder_type:
                confirm_msg = f"Are you sure you want to clear cache for {args.embedder_type} embedder?"
            elif args.student_name:
                confirm_msg = f"Are you sure you want to clear cache for student '{args.student_name}'?"
            elif args.model_name:
                confirm_msg = f"Are you sure you want to clear cache for model '{args.model_name}'?"
            else:
                print("❌ Error: Must specify --all, --embedder-type, --student-name, or --model-name")
                sys.exit(1)
            
            response = input(f"\n{confirm_msg} [y/N]: ").lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled")
                return
        
        # Perform clearing operation
        files_removed = 0
        
        if args.all:
            logger.info("Clearing all cached embeddings...")
            files_removed = clear_all_cache(cache_dir, logger)
        elif args.embedder_type:
            logger.info(f"Clearing cache for {args.embedder_type} embedder...")
            files_removed = clear_embedder_cache(cache_dir, args.embedder_type, logger)
        elif args.student_name:
            logger.info(f"Clearing cache for student {args.student_name}...")
            files_removed = clear_student_cache(cache_dir, args.student_name, logger)
        elif args.model_name:
            logger.info(f"Clearing cache for model {args.model_name}...")
            files_removed = clear_model_cache(cache_dir, args.model_name, logger)
        else:
            print("❌ Error: Must specify --all, --embedder-type, --student-name, or --model-name")
            sys.exit(1)
        
        # Show results
        print(f"\n✅ CACHE CLEARING COMPLETED")
        print(f"🗑️  Files removed: {files_removed}")
        
        if files_removed > 0:
            # Get new statistics
            stats_after = get_cache_statistics(cache_dir)
            space_freed = stats_before['total_size_mb'] - stats_after['total_size_mb']
            print(f"💾 Space freed: {space_freed:.2f} MB")
            
            if stats_after['total_files'] > 0:
                print(f"📄 Remaining files: {stats_after['total_files']}")
        
        logger.info(f"Cache clearing completed successfully. {files_removed} files removed.")
        
    except KeyboardInterrupt:
        logger.info("Cache clearing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cache clearing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 