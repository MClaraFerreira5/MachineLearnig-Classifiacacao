"""
Funções utilitárias para limpeza, exploração e pré-processamento
do dataset.
Estas funções são usadas no Notebook 01 (EDA e Pré-processamento)
"""

import pandas as pd


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto."""
    return pd.read_csv(path)


def remove_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove colunas desnecessárias."""
    return df.drop(columns=cols, errors="ignore")


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa x (dados preditivos) e y (dados alvo).
    """

    y = df[target]
    x = df.drop(columns=[target])

    return x, y
