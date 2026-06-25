"""
Logging utilities for SE Word Embeddings Pipeline
Provides centralized logging configuration and management
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""

    logging_config = config.get('logging', {})

    # Get logging parameters
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    log_file = logging_config.get('file')
    console_output = logging_config.get('console', True)
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    max_file_size = logging_config.get('max_file_size', '10MB')
    backup_count = logging_config.get('backup_count', 5)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse max file size
        size_bytes = _parse_size(max_file_size)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initialization
    logger = logging.getLogger('se_embeddings')
    logger.info("=" * 60)
    logger.info("SE Word Embeddings Logging System Initialized")
    logger.info(f"Log Level: {logging_config.get('level', 'INFO')}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Console Output: {console_output}")
    logger.info("=" * 60)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes"""

    size_str = size_str.upper().strip()

    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)
