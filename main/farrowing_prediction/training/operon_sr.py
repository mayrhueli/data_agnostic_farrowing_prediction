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

``pyoperon.sklearn.SymbolicRegressor``'s own defaults are misleading for
a small tabular problem like this one — verified by reading its source
(``_resolve_params``), not by rumor:

- ``optimizer_iterations=0``: Operon's actual selling point over other GP
  engines is *memetic* search — Levenberg-Marquardt local refinement of
  each individual's constants every generation. The library default
  disables it entirely, leaving plain unrefined-constant GP.
- ``objectives=['r2']``: single-objective by default, so there is no
  active pressure toward smaller trees during evolution, only the hard
  ``max_length``/``max_depth`` caps (50/10 by default — generous for a
  few dozen rows per sow). ``['r2', 'length']`` turns on NSGA2 for a real
  accuracy/complexity Pareto front, picked from via
  ``model_selection_criterion`` (default ``minimum_description_length``,
  kept here).
- plain ``div`` blows up near a zero denominator — exactly what produces
  wildly inflated error when applying a model fit on one signal's scale
  to another signal's raw values. ``aq`` (analytic quotient,
  x/sqrt(1+y^2)) is Operon's own answer to this and is used here instead.

The defaults below (``optimizer_iterations=10``, ``max_length=25``,
``max_depth=7``, ``allowed_symbols`` with ``aq`` in place of ``div``,
``objectives=['r2', 'length']``) reflect that; every one is still a
plain constructor argument you can override.

Traced a failure mode — ``TypeError: FitLeastSquares(): incompatible
function arguments ... Invoked with types: ndarray, ndarray``, deep
inside ``.fit()`` — to a buffer-aliasing issue in pyoperon's C++
binding: converting X/y to a fresh, C-contiguous, float64 array via
``np.ascontiguousarray`` is not enough on its own to make it reliable
once that conversion happens inside a function/method call (as opposed
to bare script top-level) — byte-identical arrays (verified by hashing
them right before the failing call) fail when built that way but not
when built at top level. An explicit extra ``.copy()`` on top of
``ascontiguousarray`` reliably avoids it in every trial run here; kept
even though it looks redundant, because dropping it reintroduces the
failure. ``fit`` also retries on that specific error a few times with a
perturbed seed as a second line of defense, since a residual flake
under heavier multi-threading can't be fully ruled out.
"""
import json

import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# numpy ufuncs available to a persisted formula's eval() — covers every
# function pyoperon's own allowed_symbols vocabulary can produce (aq
# itself is already expanded to sqrt/division in get_model_string's
# output, so sqrt covers it; the others are here for allowed_symbols
# values beyond this module's own default).
_FORMULA_NAMESPACE = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
}


def evaluate_formula(formula: str, X) -> np.ndarray:
    """Evaluate a formula string from ``OperonSymbolicRegressorWrapper.formula()``
    against a feature table, vectorized over its columns — this is the
    *actual* fitted model's function (verified to agree with
    ``.predict()`` to ~1e-6 at the default precision), not an
    approximation, so this is a legitimate way to run a trained model
    without needing the original (unpicklable — see ``dump_model``)
    pyoperon tree object.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"evaluate_formula needs a DataFrame to know each column's name, got {type(X)}")
    namespace = dict(_FORMULA_NAMESPACE)
    namespace.update({col: X[col].to_numpy() for col in X.columns})
    namespace["__builtins__"] = {}
    return np.asarray(eval(formula.replace("^", "**"), namespace), dtype=np.float64)


class FormulaModel:
    """A fitted Operon model reduced to its formula string — the result
    of ``OperonSymbolicRegressorWrapper.load_model()``. Implements the
    same ``.predict()``/``.formula()`` interface as the original wrapper
    (so it's a drop-in for scoring/plotting code), but not ``.fit()``:
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
        # .copy() is load-bearing here — see module docstring
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
        of the raw fitted object: pyoperon's underlying C++ tree
        (``pyoperon.pyoperon.Tree``) fails to pickle in this build —
        its ``__getstate__`` returns a type pybind11 can't convert back
        to Python (confirmed by hitting it directly, not assumed) — so
        joblib/pickle is not an option here. The formula is the exact
        fitted function (``evaluate_formula`` reproduces ``.predict()``
        to ~1e-6 at this precision), so nothing about the model's actual
        behavior is lost; what's gone is the population/search state,
        which inference doesn't need anyway. Load back with
        ``load_model`` — the result is a ``FormulaModel``, not a new
        ``OperonSymbolicRegressorWrapper`` (it can predict, not refit).
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
