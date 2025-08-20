"""Configuration loader for Datarus Pipeline"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file and environment variables"""
    load_dotenv()
    
    config_file = Path(config_path)
    if not config_file.exists():
        example_config = config_file.parent / "config.yaml.example"
        if example_config.exists():
            config_file = example_config
        else:
            # Return default config if no file found
            return {
                "model": {
                    "endpoint": "http://localhost:8000/v1/chat/completions",
                    "path": "/mnt/dellagih/misc/models/Datarus-R1-14B-preview-v4-ckpt3-Exp-Calibration-f9674/",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream": False
                },
                "docker": {
                    "image": "jupyter/tensorflow-notebook",
                    "remote_server_ip": "localhost",
                    "jupyter_port": 8881
                },
                "pipeline": {
                    "max_iterations": 30,
                    "execution_timeout": 60
                },
                "development": {
                    "debug": False
                }
            }
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    

    return config
