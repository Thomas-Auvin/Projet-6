# fonctions_mlflow.py

from typing import Any, Dict

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def run_model(
    name: str,
    estimator,
    X_train,
    y_train,
    X_valid,
    y_valid,
    phase: str = "baseline",
    feature_set: str = "v1_all_tables",
) -> Dict[str, Any]:
    """
    Lance un modèle dans un Pipeline simple (imputer -> scaler -> modèle),
    loggue les métriques dans MLflow, et renvoie un dict de scores + pipeline.
    """

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    with mlflow.start_run(run_name=name):
        # tags de contexte
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("feature_set", feature_set)

        # Entraînement
        pipe.fit(X_train, y_train)

        # Prédictions proba
        y_train_proba = pipe.predict_proba(X_train)[:, 1]
        y_valid_proba = pipe.predict_proba(X_valid)[:, 1]

        # Métriques
        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_valid = roc_auc_score(y_valid, y_valid_proba)
        ap_train = average_precision_score(y_train, y_train_proba)
        ap_valid = average_precision_score(y_valid, y_valid_proba)

        # Log MLflow
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_valid", auc_valid)
        mlflow.log_metric("ap_train", ap_train)
        mlflow.log_metric("ap_valid", ap_valid)

        print(
            f"[{name}] AUC train={auc_train:.3f} | valid={auc_valid:.3f} | "
            f"AP train={ap_train:.3f} | valid={ap_valid:.3f}"
        )

        return {
            "name": name,
            "auc_train": auc_train,
            "auc_valid": auc_valid,
            "ap_train": ap_train,
            "ap_valid": ap_valid,
            "pipeline": pipe,
        }


def run_model_cv(
    name: str,
    estimator,
    X,
    y,
    model_name: str = "unknown",
    n_splits: int = 3,
    random_state: int = 42,
    use_sampling: bool = True,
    smote_ratio: float = 0.5,
    use_under: bool = True,
    under_ratio: float = 1.0,
    phase: str = "cv",
    feature_set: str = "v1_all_tables",
) -> Dict[str, Any]:
    """
    CV stratifiée avec option :
      - SMOTE (over-sampling)
      - RandomUnderSampler (under-sampling)
    Retourne les scores OOF et la moyenne par fold.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    aucs = []
    aps = []
    oof_probas = np.zeros(len(y))  # <--- on prépare le vecteur des proba OOF

    # Construction dynamique du pipeline
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_sampling:
        smote = SMOTE(
            sampling_strategy=smote_ratio,
            random_state=random_state,
        )
        steps.append(("smote", smote))

        if use_under:
            under = RandomUnderSampler(
                sampling_strategy=under_ratio,
                random_state=random_state,
            )
            steps.append(("under", under))

    steps.append(("model", estimator))

    pipe = ImbPipeline(steps=steps)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag(
            "validation",
            "cv_smote_under" if (use_sampling and use_under) else
            "cv_smote" if use_sampling else "cv",
        )
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("smote_ratio", smote_ratio if use_sampling else None)
        mlflow.log_param("under_ratio", under_ratio if (use_sampling and use_under) else None)

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # Fit sur le fold
            pipe.fit(X_train_fold, y_train_fold)

            # Probas sur le fold de validation
            y_valid_proba = pipe.predict_proba(X_valid_fold)[:, 1]
            oof_probas[valid_idx] = y_valid_proba  # <--- on remplit le vecteur global

            # Scores du fold
            auc_fold = roc_auc_score(y_valid_fold, y_valid_proba)
            ap_fold = average_precision_score(y_valid_fold, y_valid_proba)
            aucs.append(auc_fold)
            aps.append(ap_fold)

        # Scores OOF globaux
        auc_oof = roc_auc_score(y, oof_probas)
        ap_oof = average_precision_score(y, oof_probas)

        mlflow.log_metric("auc_oof", auc_oof)
        mlflow.log_metric("ap_oof", ap_oof)
        mlflow.log_metric("auc_mean_cv", float(np.mean(aucs)))
        mlflow.log_metric("ap_mean_cv", float(np.mean(aps)))

        print(
            f"[{name}] AUC OOF={auc_oof:.3f} | AP OOF={ap_oof:.3f} "
            f"(moyenne folds AUC={np.mean(aucs):.3f}, AP={np.mean(aps):.3f})"
        )

        return {
            "name": name,
            "auc_oof": auc_oof,
            "ap_oof": ap_oof,
            "auc_folds": aucs,
            "ap_folds": aps,
            "pipeline": pipe,
            "oof_probas": oof_probas,    
            "y_true": y.values,          
        }
