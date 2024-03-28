from functools import partial

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from utils import Model


class Optimiser:
    def __init__(self, model: Model, n_trials: int = 5000, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.model = model
        self.study = None
        self.best_params = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()

        self.study = optuna.create_study(
            sampler=sampler, pruner=pruner, study_name="OptunaCatBoost", direction="maximize"
        )
        objective_partial = partial(self._objective, X=X, y=y)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.best_params = self.study.best_params

    def _objective(self, trial: optuna.trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 4000, 5000),
            # "depth": trial.suggest_int("depth", 1, 12),
            # "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            # "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
            # "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            # "border_count": trial.suggest_int("border_count", 1, 255),
            # "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 30.0),
            "random_state": self.random_state,
        }

        model = self.model(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        return scores.mean()
