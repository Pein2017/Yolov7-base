import os
import time
from typing import Any, Dict

from .config import Args


def generate_exp_setting(args: Args, search_space: Dict[str, Any]) -> str:
    """
    Generate a shorthand string representing the hyperparameter settings.

    Args:
        args (Args): Configuration object.
        search_space (Dict[str, Any]): Dictionary of hyperparameters and their suggested values.

    Returns:
        str: Shorthand string.
    """
    exp_settings = []
    # List of hyperparameters to include in the exp_setting with their shorthand names
    keys = [
        ("lr", "lr"),
        ("hidden_dim", "hid_dim"),
        ("attn_heads", "attn_h"),
        ("dropout", "drop"),
        ("fc_hidden_dims_1", "fc_hid1"),
        ("fc_hidden_dims_2", "fc_hid2"),
        ("embedding_dim", "emb_dim"),
        ("num_layers", "num_L"),
    ]
    for long_name, short_name in keys:
        value = search_space.get(long_name, getattr(args, long_name, None))
        # if isinstance(value, float):
        #     if long_name == "lr":
        #         value = f"{value:.4f}"
        #     elif long_name == "dropout":
        #         value = f"{value:.2f}"
        exp_settings.append(f"{short_name}_{value}")
    return "-".join(exp_settings)


def remove_directory_contents(directory):
    for _ in range(10):  # Try up to 5 times
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.unlink(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            return True  # Successfully removed all contents
        except Exception as e:
            print(f"Error while removing directory contents: {e}")
            time.sleep(1.5)  # Wait for 1 second before retrying
    return False  # Failed to remove after 5 attempts
