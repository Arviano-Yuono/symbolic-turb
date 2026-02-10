"""
This module contains configuration classes for the SR-TurbulenceModeling project.
"""

import json
import os


class Config:
    """Simple config class that reads configuration files and stores it."""

    def __init__(self, config_path: str = "configs/config.json"):
        with open(os.path.join(os.path.dirname(__file__), config_path), "r") as f:
            self.config = json.load(f)

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return f"Config({self.config})"

    def __getitem__(self, key: str):
        return self.get_config(key)

    def get_config(self, key: str):
        if hasattr(self.config, key):
            return self.config[key]
        else:
            raise KeyError(f"Key '{key}' not found in config")
