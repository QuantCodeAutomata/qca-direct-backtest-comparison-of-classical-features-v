"""Model factory and hyperparameter grids for PQFM RFQ fill-prediction experiment."""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

MODEL_FAMILIES = ['LR', 'XGB', 'RF', 'NN']

# Precomputed C grid for logistic regression: 10^v for v in [-4, -3.2, ..., 4] (step 0.8)
_LR_C_VALUES = [float(10 ** v) for v in [-4, -3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2, 4]]

# Full paper-specified hyperparameter grids
_PARAM_GRIDS: dict[str, dict] = {
    'LR': {
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cholesky', 'newton-cg', 'liblinear'],
        'C': _LR_C_VALUES,
    },
    'XGB': {
        'max_depth': [3, 5, 7, 9, 11],
        'n_estimators': [80, 100, 120, 140, 160, 180],
        'learning_rate': [0.15, 0.1, 0.05, 0.01],
    },
    'RF': {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [80, 100, 120, 140, 160, 180],
    },
    'NN': {
        'hidden_layer_sizes': [
            (3,), (100,), (164,), (164, 65),
            (262, 164, 98), (262, 196, 131, 65),
            (262, 196, 131, 65, 3), (196, 3),
        ],
        'activation': ['relu', 'logistic', 'tanh'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    },
}

# Single fixed parameters for ultrafast CI runs — no grid search, 1 fit per day
_SINGLE_PARAMS: dict[str, dict] = {
    'LR':  {'solver': 'lbfgs',  'C': 1.0},
    'XGB': {'max_depth': 3,     'n_estimators': 80, 'learning_rate': 0.1},
    'RF':  {'criterion': 'gini','n_estimators': 80},
    'NN':  {'hidden_layer_sizes': (100,), 'activation': 'relu', 'learning_rate': 'constant'},
}

# Reduced fast grids: representative subset for speed (CI / small-sample experiments)
_FAST_PARAM_GRIDS: dict[str, dict] = {
    'LR': {
        'solver': ['lbfgs', 'liblinear'],
        'C': [0.01, 1.0, 100.0],
    },
    'XGB': {
        'max_depth': [3, 5],
        'n_estimators': [80, 120],
        'learning_rate': [0.1, 0.05],
    },
    'RF': {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [80, 120],
    },
    'NN': {
        'hidden_layer_sizes': [(100,), (164, 65)],
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive'],
    },
}


def get_model_and_grid(family: str, fast: bool = False) -> tuple[object, dict]:
    """Return a fresh (estimator, param_grid) pair for the requested model family.

    The estimator is configured with fixed non-tuned hyperparameters (penalty, max_iter,
    eval_metric, etc.) while the returned param_grid contains every axis to be searched
    via GridSearchCV.

    Parameters
    ----------
    family:
        One of 'LR', 'XGB', 'RF', 'NN'.
    fast:
        If ``True``, return a reduced parameter grid suitable for small-sample
        experiments; the full paper-specified grid is returned when ``False``.

    Returns
    -------
    tuple[object, dict]
        ``(estimator, param_grid)`` where *estimator* is an unfitted sklearn-compatible
        classifier and *param_grid* maps parameter names to lists of candidate values.

    Raises
    ------
    ValueError
        If *family* is not one of the supported ``MODEL_FAMILIES``.

    Notes
    -----
    Full paper-specified hyperparameter grids (``fast=False``):

    * **LR** – ``solver`` ∈ {lbfgs, sag, saga, newton-cholesky, newton-cg, liblinear},
      ``C`` ∈ 10^{-4, -3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2, 4},
      ``penalty='l2'``, ``max_iter=10000``.
    * **XGB** – ``max_depth`` ∈ {3,5,7,9,11},
      ``n_estimators`` ∈ {80,100,120,140,160,180},
      ``learning_rate`` ∈ {0.15, 0.1, 0.05, 0.01}.
    * **RF** – ``criterion`` ∈ {gini, entropy, log_loss},
      ``n_estimators`` ∈ {80,100,120,140,160,180}.
    * **NN** – ``hidden_layer_sizes`` ∈ 8 architectures,
      ``activation`` ∈ {relu, logistic, tanh},
      ``learning_rate`` ∈ {constant, invscaling, adaptive}, ``max_iter=10000``.
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family '{family}'. Choose from {MODEL_FAMILIES}.")

    if family == 'LR':
        estimator = LogisticRegression(penalty='l2', max_iter=10000)
    elif family == 'XGB':
        estimator = XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
        )
    elif family == 'RF':
        estimator = RandomForestClassifier()
    else:  # 'NN'
        estimator = MLPClassifier(max_iter=10000)

    grid = _FAST_PARAM_GRIDS[family] if fast else _PARAM_GRIDS[family]
    return estimator, dict(grid)


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    family: str,
    cv: int = 3,
    n_jobs: int = 1,
    random_state: int = 42,
    fast: bool = False,
    ultrafast: bool = False,
) -> object:
    """Run ``GridSearchCV`` for *family* and return the refitted best estimator.

    Parameters
    ----------
    X_train:
        Feature matrix, shape ``(n_samples, n_features)``.
    y_train:
        Binary label vector, shape ``(n_samples,)``.
    family:
        Model family string – one of ``MODEL_FAMILIES``.
    cv:
        Number of cross-validation folds (default 3).
    n_jobs:
        Parallelism for ``GridSearchCV`` (default 1 = sequential).
    random_state:
        Seed forwarded to the estimator where supported.
    fast:
        If ``True``, use a reduced hyperparameter grid (fewer combos).
    ultrafast:
        If ``True``, bypass GridSearchCV entirely and fit once with fixed default
        parameters from ``_SINGLE_PARAMS``.  Intended only for CI/smoke runs.

    Returns
    -------
    object or None
        The refitted best estimator on the full training set, or ``None`` when:

        * Only one class is present in *y_train*.
        * The training set is too small for the requested CV (``n_samples < cv * 2``
          or any class has fewer than *cv* examples).
        * An unexpected exception occurs during fitting.

    Notes
    -----
    ``ConvergenceWarning`` messages from iterative solvers (LR, NN) are suppressed
    silently because the wide grid ensures at least one converged configuration is
    typically found during CV.
    """
    classes, counts = np.unique(y_train, return_counts=True)

    # Guard: single-class
    if len(classes) < 2:
        warnings.warn(
            f"train_best_model ({family}): only one class in y_train – skipping.",
            stacklevel=2,
        )
        return None

    # ultrafast: single fit, no grid search — for CI speed only
    if ultrafast:
        estimator, _ = get_model_and_grid(family, fast=True)
        if hasattr(estimator, 'random_state'):
            estimator.set_params(random_state=random_state)
        estimator.set_params(**_SINGLE_PARAMS[family])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                estimator.fit(X_train, y_train)
            return estimator
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"train_best_model ultrafast ({family}): {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            return None

    if len(X_train) < cv * 2 or counts.min() < cv:
        warnings.warn(
            f"train_best_model ({family}): insufficient samples "
            f"(n={len(X_train)}, min_class={counts.min()}) for cv={cv} – skipping.",
            stacklevel=2,
        )
        return None

    estimator, param_grid = get_model_and_grid(family, fast=fast)

    # Inject random_state where the estimator supports it
    if hasattr(estimator, 'random_state'):
        estimator.set_params(random_state=random_state)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        error_score=np.nan,
        refit=True,
        n_jobs=n_jobs,
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', message='.*lbfgs.*')
            search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"train_best_model ({family}): GridSearchCV raised {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return None


def predict_proba(model: object, X_test: np.ndarray) -> np.ndarray:
    """Return the predicted probability for class 1.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible classifier that exposes ``predict_proba``.
    X_test:
        Feature matrix, shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_samples,)`` with P(y=1 | X).

    Raises
    ------
    ValueError
        If *model* is ``None`` or does not implement ``predict_proba``.
    """
    if model is None:
        raise ValueError("predict_proba received None model.")
    if not hasattr(model, 'predict_proba'):
        raise ValueError(
            f"Model {type(model).__name__} does not implement predict_proba."
        )
    return model.predict_proba(X_test)[:, 1]
