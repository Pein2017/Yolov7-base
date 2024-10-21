# utils/config.py

from dataclasses import dataclass
from typing import List

import yaml
from dacite import from_dict


@dataclass
class Args:
    # Data-related parameters
    pos_dir: str
    neg_dir: str
    classes_file: str
    balance_ratio: float
    resample_method: str
    num_classes: int

    # Training parameters
    num_epochs: int
    valid_epoch: int
    log_after_epoch: int
    batch_size: int
    lr: float
    optimizer: str
    gradient_clip_val: float
    scheduler_type: str

    # Model architecture parameters
    num_layers: int
    attn_heads: int
    dropout: float
    embedding_dim: int
    fc_hidden_dims: List[int]

    # Evaluation and optimization parameters
    selection_criterion: str
    early_stop_metric: str
    patience: int
    mode: str

    # Hardware and environment parameters
    device_id: int
    seed: int

    # Logging and storage parameters
    ltn_log_dir: str
    mysql_url: str

    # Optuna-specific parameters
    delete_existing_study: bool
    n_trials: int
    study_name: str
    sampler: str
    pruner: str

    # Control checkpointing
    enable_checkpointing: bool = False
    delete_existing_study_folder: bool = True

    # Direction for Optuna study
    direction: str = "minimize"  # 'minimize' or 'maximize'

    @staticmethod
    def from_yaml(yaml_file: str) -> "Args":
        """
        Load configuration from a YAML file and return an Args instance.

        Args:
            yaml_file (str): Path to the YAML configuration file.

        Returns:
            Args: Configuration object populated from YAML.
        """
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return from_dict(data_class=Args, data=config_dict)
