import pandas as pd

# === Fonctions utilitaires pour le feature engineering ===


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    """
    One-hot encode toutes les colonnes de type 'object'.
    Retourne :
      - df_enc : le dataframe encodé
      - new_cols : la liste des colonnes créées (les dummies)
    """
    original_cols = list(df.columns)
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    
    df_enc = pd.get_dummies(
        df,
        columns=cat_cols,
        dummy_na=nan_as_category
    )
    
    new_cols = [c for c in df_enc.columns if c not in original_cols]
    return df_enc, new_cols


def agg_numeric_and_cat(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """
    Agrège un dataframe au niveau de `key` (ex: SK_ID_CURR) :
      - pour les colonnes numériques : min, max, mean, sum, var
      - pour les colonnes catégorielles (dummies) : mean (proportion)
    Ajoute un préfixe à toutes les colonnes de sortie.
    """
    # One-hot encoding des colonnes object
    df_enc, cat_cols = one_hot_encoder(df, nan_as_category=True)
    
    # Colonnes numériques (hors clé et dummies)
    num_cols = df_enc.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [key]]
    
    agg_dict = {}
    for col in num_cols:
        agg_dict[col] = ["min", "max", "mean", "sum", "var"]
    for col in cat_cols:
        # Pour les dummies, la moyenne = proportion
        agg_dict[col] = ["mean"]
    
    agg = df_enc.groupby(key).agg(agg_dict)
    
    # Renommer les colonnes multi-index
    agg.columns = [
        f"{prefix}{col}_{stat.upper()}" for col, stat in agg.columns
    ]
    
    agg = agg.reset_index()
    return agg
