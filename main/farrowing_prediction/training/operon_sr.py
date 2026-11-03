# Copyright (C) 2026 Elisabeth Mayrhuber
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Symbolic regression via Operon, through pyoperon's scikit-learn-compatible
``SymbolicRegressor``. Operon fits a genetic-programming expression tree
plus an outer affine transform (scale * tree(x) + intercept) by default.
"""
import json

import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


_FORMULA_NAMESPACE = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
}


def evaluate_formula(formula: str, X) -> np.ndarray:
    """Evaluate a formula string from ``OperonSymbolicRegressorWrapper.formula()``
    against a feature table, vectorized over its columns — this is the
    *actual* fitted model's function (verified to agree with
    ``.predict()`` to ~1e-6 at the default precision), not an
    approximation, this is a way to run "the selected
    model" without needing the original (unpicklable — see
    ``dump_model``) pyoperon tree object.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"evaluate_formula needs a DataFrame to know each column's name, got {type(X)}")
    namespace = dict(_FORMULA_NAMESPACE)
    namespace.update({col: X[col].to_numpy() for col in X.columns})
    namespace["__builtins__"] = {}
    return np.asarray(eval(formula.replace("^", "**"), namespace), dtype=np.float64)


class FormulaModel:
    """A fitted Operon model reduced to its formula string: the result
    of ``OperonSymbolicRegressorWrapper.load_model()``. Implements the
    same ``.predict()``/``.formula()`` interface as the original wrapper, but not ``.fit()``:
    this is inference-only, by construction, since it has no training
    state beyond the expression itself.
    """

    def __init__(self, formula: str, feature_names: list, size: int = None, depth: int = None):
        self._formula = formula
        self.feature_names = list(feature_names)
        self.size = size
        self.depth = depth

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return evaluate_formula(self._formula, X[self.feature_names])

    def formula(self, precision: int = 3) -> str:
        return self._formula


class OperonSymbolicRegressorWrapper:
    def __init__(self, population_size: int = 1000, generations: int = 200,
                 allowed_symbols: str = "add,sub,mul,aq,constant,variable",
                 max_length: int = 25, max_depth: int = 7,
                 optimizer_iterations: int = 10,
                 objectives: list = ("r2", "length"),
                 model_selection_criterion: str = "minimum_description_length",
                 n_threads: int = 1, random_state: int = 0, max_fit_retries: int = 3, **kwargs):
        self.n_threads = n_threads
        self.max_fit_retries = max_fit_retries
        self._sr_kwargs = dict(
            population_size=population_size,
            generations=generations,
            allowed_symbols=allowed_symbols,
            max_length=max_length,
            max_depth=max_depth,
            optimizer_iterations=optimizer_iterations,
            objectives=list(objectives),
            model_selection_criterion=model_selection_criterion,
            n_threads=n_threads,
            random_state=random_state,
            **kwargs,
        )
        self.model = SymbolicRegressor(**self._sr_kwargs)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        X_arr = np.ascontiguousarray(X, dtype=np.float64).copy()
        y_arr = np.ascontiguousarray(y, dtype=np.float64).copy()

        attempt = 0
        while True:
            try:
                self.model.fit(X_arr, y_arr)
                return self
            except TypeError as e:
                attempt += 1
                if "FitLeastSquares" not in str(e) or attempt >= self.max_fit_retries:
                    raise
                # transient pyoperon multi-threading race (see module docstring);
                # re-seed so the retry isn't a byte-for-byte repeat of the failure
                retry_kwargs = dict(self._sr_kwargs)
                retry_kwargs["random_state"] = int(self._sr_kwargs.get("random_state") or 0) + attempt
                self.model = SymbolicRegressor(**retry_kwargs)

    def predict(self, X):
        return self.model.predict(np.ascontiguousarray(X, dtype=np.float64).copy())

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mae, r2

    def formula(self, precision: int = 3) -> str:
        """The best model as a readable expression, using real feature
        names if the model was fit on a DataFrame."""
        return self.model.get_model_string(self.model.model_, precision=precision, names=self.feature_names)

    @property
    def size(self) -> int:
        """Number of nodes in the best model's expression tree."""
        return self.model.model_.Length

    @property
    def depth(self) -> int:
        """Maximum depth of the best model's expression tree."""
        return self.model.model_.Depth

    def dump_model(self, filename: str, precision: int = 10) -> None:
        """Persist this model as its formula string (JSON), not a pickle
        of the raw fitted object.
        """
        payload = {
            "formula": self.formula(precision=precision),
            "feature_names": self.feature_names,
            "size": self.size,
            "depth": self.depth,
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_model(cls, filename: str) -> "FormulaModel":
        with open(filename) as f:
            payload = json.load(f)
        return FormulaModel(payload["formula"], payload["feature_names"], payload.get("size"), payload.get("depth"))
