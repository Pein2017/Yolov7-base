# optim/search_space.py

from typing import Any, Dict

import optuna


def define_search_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.

    Returns:
        Dict[str, Any]: Dictionary of suggested hyperparameters.
    """
    search_space = {
        "optimizer": "Adam",  # Fixed to Adam as per user's requirement
        "lr": trial.suggest_float("lr", 5e-4, 2e-2, log=True),
        "embedding_dim": trial.suggest_int("embedding_dim", 16, high=128, step=16),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 4, 8]),
        "attn_heads": trial.suggest_int("attn_heads", 1, 64),
        "dropout": 0.1,  # Fixed dropout as per user's setting
        "fc_hidden_dims_2": trial.suggest_categorical(
            "fc_hidden_dims_2", [16, 32, 64, 128]
        ),
    }

    # Set fc_hidden_dims_1 to be twice fc_hidden_dims_2
    search_space["fc_hidden_dims_1"] = search_space["fc_hidden_dims_2"] * 2

    # Combine fc_hidden_dims into a list
    search_space["fc_hidden_dims"] = [
        search_space["fc_hidden_dims_1"],
        search_space["fc_hidden_dims_2"],
    ]

    return search_space