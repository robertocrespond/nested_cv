import random
import time
from copy import deepcopy
from datetime import timedelta
import json
from typing import Any
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from nitro.backend_adapter import BackendAdapter
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


class NestedCV(BackendAdapter):
    def __init__(
        self,
        model_class,
        scoring: str,
        outer_cv,
        inner_cv,
        param_grid,
        refit=True,
        random_state=1,
        random_iterations=10,
        n_jobs=-1,
    ):
        """Fit a classifier under a Nested Cross Validation training scheme

        Args:
            model_class (sklearn.BaseEstimator): Sklearn model already initialized. Ideally this is a
                sklearn.pipeline.Pipeline so that per-sample training/test isolation is performed correctly.
            scoring (str): sklearn metric. Run sklearn.metrics.SCORERS.keys() to get a list
            outer_cv (cross-validation generator): Used for outer loop validation (Model Evaluation). E.g. KFold, StratisfiedKFold, GroupKFold
            inner_cv (cross-validation generator): Used for inner loop validation (Hyperparameters Evaluation). E.g. KFold, StratisfiedKFold, GroupKFold
            param_grid (dict): hyperparameter grid search space
            refit (bool, optional): Whether to retrain model with best selected parameters with all available data. Defaults to True.
            random_state (int, optional): Random number for repodrucibility. Defaults to 1.
            random_iterations (int, optional): Random search iteration count. If set to None, exhaustive grid search is performed. Defaults to 10.
            n_jobs (int, optional): The maximum number of concurrently running jobs. Defaults to -1.
        """
        self.model_class = model_class
        self.scoring = scoring
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.param_grid = param_grid
        self.random_iterations = random_iterations
        self.random_state = random_state
        self.refit = refit
        self.backend_key = type(self).__name__ + "_" + type(model_class).__name__ + "_" + str(random_state)
        self.n_jobs = n_jobs

    def _fit_generator(self, X, y, grid):
        """Dataset & Hyperparameter generator for the inner loop"""
        for params in grid:
            for j, (inner_train_idx, inner_val_idx) in enumerate(self.inner_cv.split(X)):
                X_inner_train, y_inner_train = X.loc[inner_train_idx, :], y[inner_train_idx]
                X_inner_val, y_inner_val = X.loc[inner_val_idx, :], y[inner_val_idx]
                yield params, X_inner_train, y_inner_train, X_inner_val, y_inner_val

    def _inner_fit(self, params, X_inner_train, y_inner_train, X_inner_val, y_inner_val):
        """Fit a Dataset & Hyperparameter combo. Used in the inner loop"""
        inner_train_start = time.time()
        param_key = json.dumps(params)

        scorer = get_scorer(self.scoring)
        mod = deepcopy(self.model_class)  # copy model

        if isinstance(mod, Pipeline):
            inner_model = mod.set_params(**params)
        else:
            inner_model = mod.set_params(**params, random_state=self.random_state)
        inner_model.fit(X_inner_train, y_inner_train)

        # Compute evaluation metric on both sets
        inner_train_score = scorer(inner_model, X_inner_train, y_inner_train)
        inner_val_score = scorer(inner_model, X_inner_val, y_inner_val)

        return param_key, inner_train_score, inner_val_score, time.time() - inner_train_start

    def _log(self, *args, render: bool):
        """Conditional renderer used for verbose level"""
        # TODO: change print to a logger
        if render:
            print(*args)

    def _process_parallel_results(self, parallel_results: dict):
        """Consolidate data from all different processes into a unified data structure"""
        results_by_params = {}
        inner_models_times = []
        param_list = []
        param_train_scores_mean = []
        param_train_scores_std = []
        param_val_scores_mean = []
        param_val_scores_std = []
        for param_key, inner_train_score, inner_val_score, time_spent in parallel_results:
            if param_key not in results_by_params:
                results_by_params[param_key] = {}
                results_by_params[param_key]["train_scores"] = []
                results_by_params[param_key]["val_scores"] = []

            results_by_params[param_key]["train_scores"].append(inner_train_score)
            results_by_params[param_key]["val_scores"].append(inner_val_score)
            inner_models_times.append(time_spent)

        # Restructure results per param_key
        for params_json, v in results_by_params.items():
            # Store evaluation metrics of parameter combination
            param_list.append(json.loads(params_json))
            param_train_scores_mean.append(np.mean(v["train_scores"]))
            param_train_scores_std.append(np.std(v["train_scores"]))
            param_val_scores_mean.append(np.mean(v["val_scores"]))
            param_val_scores_std.append(np.std(v["val_scores"]))

        inner_training_results = {}
        inner_training_results["params"] = param_list
        inner_training_results["train_scores_mean"] = np.array(param_train_scores_mean)
        inner_training_results["train_scores_std"] = np.array(param_train_scores_std)
        inner_training_results["val_scores_mean"] = np.array(param_val_scores_mean)
        inner_training_results["val_scores_std"] = np.array(param_val_scores_std)
        inner_training_results["execution_time"] = np.array(inner_models_times)
        return inner_training_results

    def _reproducibility(self, obj):
        """Set all random states for reproducibility"""
        if isinstance(obj, Pipeline):
            for step in obj.steps:
                obj = step[1]
                try:
                    obj.set_params(random_state=self.random_state)
                except ValueError:
                    pass
        else:
            try:
                obj.set_params(random_state=self.random_state)
            except ValueError:
                pass

    def fit(self, X, y, verbose=1) -> None:
        if self._load():
            return

        self._reproducibility(self.model_class)

        self.outer_scores = []
        self.inner_scores = []
        self.trained_models = []

        scorer = get_scorer(self.scoring)

        # [Outer Loop]
        outer_train_start = time.time()
        for i, (train_idx, val_idx) in enumerate(self.outer_cv.split(X)):
            self._log("[Outer Loop {}] Started *******************************************************".format(i + 1), render=verbose >= 1)

            # [Split Sets]
            # Training and validation sets. Validation set must not be oversampled
            # or tampered with as it it used for generalization error estimation.
            # Only modify if you want it to be an OOT sample.
            X_train, y_train = X.loc[train_idx, :], y[train_idx]
            X_val, y_val = X.loc[val_idx, :], y[val_idx]
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            # [Define Hyperparameter grid space]
            # If random search, shuffle list and select only specified count
            grid = ParameterGrid(self.param_grid)
            if self.random_iterations > 0:
                possible_params = [g for g in grid]
                random.Random(self.random_state).shuffle(possible_params)
                if self.random_iterations > len(possible_params):
                    raise Exception("More iterations that possible parameter combinations")
                grid = iter(possible_params[: self.random_iterations])

            # [Inner Loop] (Executed in parallel)
            parallel = Parallel(n_jobs=self.n_jobs, verbose=verbose)
            inner_parallel_training_results = parallel(
                delayed(self._inner_fit)(params, X_inner_train, y_inner_train, X_inner_val, y_inner_val)
                for params, X_inner_train, y_inner_train, X_inner_val, y_inner_val in self._fit_generator(X_train, y_train, grid)
            )

            # Consolidate data from all different processes into a unified data structure
            inner_training_results = self._process_parallel_results(inner_parallel_training_results)

            # [Hyperparameter Selection]
            # Select best parameter combination using (mean - std) to account for  metric distribution.
            normalized_params_val_scores = inner_training_results["val_scores_mean"] - inner_training_results["val_scores_std"]
            best_eval_idx = normalized_params_val_scores.argmax()
            best_params = inner_training_results["params"][best_eval_idx]

            # [Model Refit]
            # Use training set of outer loop
            mod = deepcopy(self.model_class)
            best_model = mod.set_params(**best_params)
            best_model.fit(X_train, y_train)
            self.trained_models.append(best_model)

            # [Model Evaluation]
            inner_score = scorer(best_model, X_train, y_train)
            self.inner_scores.append(inner_score)
            outer_score = scorer(best_model, X_val, y_val)
            self.outer_scores.append(outer_score)

            # [Save Model to Backend]
            self.commit(key=self.backend_key + "_it_{}".format(i), value=best_model)
            self.commit(key=self.backend_key + "_it_{}_inner_score".format(i), value=inner_score)
            self.commit(key=self.backend_key + "_it_{}_outer_score".format(i), value=outer_score)

            iteration_info = {
                "Inner Models Trained": len(inner_training_results["execution_time"]),
                "Avg. Inner Model Training Time": str(timedelta(seconds=(np.mean(inner_training_results["execution_time"])))),
                "Total Training Time": str(timedelta(seconds=(time.time() - outer_train_start))),
                "Train Score": self.inner_scores[i],
                "Validation Score": self.outer_scores[i],
                "Generalization Error": self.inner_scores[i] - self.outer_scores[i],
            }
            self._log(pd.DataFrame({" ": [k for k in iteration_info.keys()], "  ": [v for v in iteration_info.values()]}), "\n", render=verbose >= 1)

        # [Best Model Selection]
        gen_errors = np.array(self.inner_scores) - np.array(self.outer_scores)
        best_eval_idx = gen_errors.argmin()
        self.best_model = self.trained_models[best_eval_idx]

        # [Save Best Model to Backend]
        self.commit(key=self.backend_key, value=self.best_model)

    def _load(self):
        self.best_model = self.load(self.backend_key)
        if self.best_model:
            self.trained_models = [self.load(key=self.backend_key + "_it_{}".format(i)) for i in range(self.outer_cv.get_n_splits())]
            self.inner_scores = [self.load(key=self.backend_key + "_it_{}_inner_score".format(i)) for i in range(self.outer_cv.get_n_splits())]
            self.outer_scores = [self.load(key=self.backend_key + "_it_{}_outer_score".format(i)) for i in range(self.outer_cv.get_n_splits())]
            self._log("Loaded [{}] from Backend".format(self.backend_key), render=True)
            return True