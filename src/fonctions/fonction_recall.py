from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np


def pick_threshold_for_recall(y_true, y_scores, target_recall=0.80):
    """
    Choisit un seuil tel que le recall soit >= target_recall.
    On prend le SEUIL LE PLUS ÉLEVÉ qui satisfait cette contrainte
    (pour avoir la meilleure précision possible).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # On enlève le premier point (recall[0] = 1.0 sans seuil associé)
    recall_t = recall[1:]          # shape (N-1,)
    thresholds_t = thresholds      # shape (N-1,)

    # Indices où le recall atteint au moins la cible
    idx_candidates = np.where(recall_t >= target_recall)[0]

    if len(idx_candidates) == 0:
        print(f"Aucun seuil ne permet d'atteindre un recall >= {target_recall:.2f}")
        # fallback : on prend le recall le plus proche
        idx = int(np.argmin(np.abs(recall_t - target_recall)))
    else:
        # On prend le DERNIER index qui respecte la contrainte
        # => seuil le plus ÉLEVÉ possible avec recall >= cible
        idx = int(idx_candidates[-1])

    thr = float(thresholds_t[idx])

    y_pred = (y_scores >= thr).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": thr,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "cm": cm,
    }
