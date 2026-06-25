"""
Configuration Manager for SE Word Embeddings Pipeline
Handles loading and validation of YAML configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    def _validate_config(self):
        """Validate required configuration sections"""

        required_sections = ['project', 'system', 'paths', 'logging', 'models']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate project section
        project = self.config.get('project', {})
        if not project.get('name'):
            raise ValueError("Project name is required")

        # Validate models section
        models = self.config.get('models', {})
        if 'word2vec' not in models or 'modernbert' not in models:
            raise ValueError("Both word2vec and modernbert model configurations are required")

    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration"""
        return self.config

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section"""
        return self.config.get(section, {})

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value using dot notation"""

        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
