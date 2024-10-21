# optim/search_space.py

from typing import Any, Dict

import optuna


def define_search_space(trial: optuna.trial.Trial, direction: str) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.
        direction (str): Direction of the study, either 'minimize' or 'maximize'.

    Returns:
        Dict[str, Any]: Dictionary of suggested hyperparameters.
    """

    search_space = {
        "optimizer": "AdamW",
        "lr": round(trial.suggest_float("lr", 5e-3, 1e-2, step=0.0001, log=False), 5),
        "embedding_dim": trial.suggest_int("embedding_dim", 8, high=32, step=4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 8, 16]),
        "attn_heads": trial.suggest_int("attn_heads", 4, 16, 4),
        "fc_hidden_dims_2": trial.suggest_categorical("fc_hidden_dims_2", [16, 32, 64]),
        "dropout": round(trial.suggest_float("dropout", 0.1, 0.3, step=0.001), 3),
    }

    # Set fc_hidden_dims_1 to be twice fc_hidden_dims_2
    search_space["fc_hidden_dims_1"] = search_space["fc_hidden_dims_2"] * 2

    # Combine fc_hidden_dims into a list
    search_space["fc_hidden_dims"] = [
        search_space["fc_hidden_dims_1"],
        search_space["fc_hidden_dims_2"],
    ]

    return search_space
