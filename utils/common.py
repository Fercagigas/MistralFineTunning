"""
Common utilities for PEFT fine-tuning of Mistral-7B-Instruct-v0.2
"""

import os
import yaml
import random
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Callable
from functools import wraps
from huggingface_hub import login
from transformers import set_seed as transformers_set_seed


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        log_file: Optional path to a log file
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_yaml_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the YAML configuration file
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    
    # Set deterministic behavior for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def login_huggingface(token: Optional[str] = None) -> None:
    """
    Login to Hugging Face Hub using the provided token or from environment variable.
    
    Args:
        token: Hugging Face token (default: None, will use HF_TOKEN environment variable)
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
        
    if not token:
        raise ValueError(
            "Hugging Face token not provided. Either pass it as an argument or "
            "set the HF_TOKEN environment variable."
        )
    
    login(token=token, write_permission=True)


def get_timestamp() -> str:
    """
    Get current timestamp in a format suitable for filenames.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_dir(base_dir: Union[str, Path], model_name: str, method: str) -> Path:
    """
    Create and return an output directory for model checkpoints.
    
    Args:
        base_dir: Base directory for outputs
        model_name: Name of the model
        method: Fine-tuning fine-tuning method (e.g., 'qlora', 'adapter', 'ptuning')
        
    Returns:
        Path: Path to the created output directory
    """
    timestamp = get_timestamp()
    model_name_short = model_name.split('/')[-1]
    output_dir = Path(base_dir) / f"{model_name_short}_{method}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def exception_handler(func: Callable) -> Callable:
    """
    Decorator to handle exceptions in functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices (CPU, GPU).
    
    Returns:
        Dict[str, Any]: Device information
    """
    device_info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return device_info


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of {all_params:,})")
