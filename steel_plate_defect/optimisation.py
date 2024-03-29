from typing import Callable
from functools import partial
from logging import getLogger

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from utils import Parameter, ParameterType


logger = getLogger(__name__)


class Optimiser:
    def __init__(self, model: Callable, parameters: list[Parameter], n_trials: int = 5000, random_state: int = 42):
        self.model = model
        self.parameters = parameters
        self.n_trials = n_trials
        self.random_state = random_state
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

    @staticmethod
    def _suggest_value(param: Parameter, trial: optuna.trial) -> int | float | str:

        if len(param.space) == 1:
            return param.space[0]

        if param.type == ParameterType.INTEGER:
            return trial.suggest_int(param.name, *param.space)
        elif param.type == ParameterType.FLOAT:
            return trial.suggest_float(param.name, *param.space)
        elif param.type == ParameterType.CATEGORICAL:
            return trial.suggest_categorical(param.name, param.space)

        message = f"Unknown parameter type: {param.type}"
        logger.error(message)
        raise ValueError(message)

    def _objective(self, trial: optuna.trial, X: pd.DataFrame, y: pd.Series) -> float:

        params = {param.name: self._suggest_value(param, trial) for param in self.parameters}

        model = self.model(**params)
        scores = cross_val_score(model, X, y, cv=4, scoring="roc_auc")
        return scores.mean()
