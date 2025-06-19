# =====================================

# src/utils.py
"""
Utility functions for logging and file operations
"""

import logging
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
        
    Returns:
        logger: Configured logger
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('patent_stock_matching.log')
        ]
    )
    
    return logging.getLogger(__name__)

def create_output_directory(directory_path):
    """
    Create output directory if it doesn't exist
    
    Args:
        directory_path (Path): Path to create
    """
    directory_path.mkdir(parents=True, exist_ok=True)
