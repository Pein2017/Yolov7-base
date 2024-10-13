from typing import Any, Dict

from .config import Args

# ================================================================
# Experimental Setting Generation
# ================================================================


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
    ]
    for long_name, short_name in keys:
        value = search_space.get(long_name, getattr(args, long_name, None))
        if isinstance(value, float):
            if long_name == "lr":
                value = f"{value:.4f}"
            elif long_name == "dropout":
                value = f"{value:.2f}"
        exp_settings.append(f"{short_name}_{value}")
    return "-".join(exp_settings)
