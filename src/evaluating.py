"""
Funções utilitárias para carregamento, avaliação e comparação
dos resultados dos modelos de Machine Learning.

Estas funções são usadas no Notebook 03 (Avaliação e Comparação Final).
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# ------------------------ LOADING CSVs ------------------------

def load_results(csv_dir: str, file_map: dict) -> pd.DataFrame:
    """
    Lê múltiplos CSVs de métricas (um de cada modelo),
    seleciona a melhor configuração (maior Mean F1) de cada modelo
    e retorna um DataFrame consolidado.

    csv_dir : str
        Caminho da pasta que contém os arquivos CSV.

    file_map : dict
        Mapeamento { NomeDoModelo : nome_do_csv }.
    """

    melhores = []

    print("\n🔄 PROCESSANDO MODELOS\n")

    for model_name, csv_name in file_map.items():
        full_path = os.path.join(csv_dir, csv_name)

        if not os.path.exists(full_path):
            print(f"[ERRO] Arquivo não encontrado: {csv_name}")
            continue

        df = pd.read_csv(full_path)

        if "Mean F1" not in df.columns:
            print(f"[PULAR] {model_name}: CSV sem coluna 'Mean F1'.")
            continue

        df_sorted = df.sort_values(by="Mean F1", ascending=False)
        best_row = df_sorted.iloc[0].to_dict()
        best_row["Modelo"] = model_name

        melhores.append(best_row)
        print(f"[OK] {model_name} carregado.")

    if not melhores:
        print("\n❌ Nenhum resultado válido foi encontrado.")
        return pd.DataFrame()

    df_best = pd.DataFrame(melhores)

    cols = ["Modelo"] + [c for c in df_best.columns if c != "Modelo"]
    return df_best[cols]



# -------------------- TABLE VISUALIZATION ----------------------

def show_metrics_table(df: pd.DataFrame):
    """
    Exibe as principais métricas de cada modelo, caso exista suporte
    ao display (ex.: Jupyter Notebook). Caso contrário, imprime no terminal.
    """

    if df.empty:
        print("Nenhum dado disponível para exibir tabela.")
        return

    cols = ["Modelo", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1"]

    try:
        from IPython.display import display
        display(df[cols])
    except:
        print(df[cols].to_string(index=False))



# --------------------- BARPLOT COMPARISON ----------------------

def plot_f1_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Gera um gráfico de barras comparando o F1-score médio dos modelos.
    Inclui barra de erro (Std F1) quando disponível.
    Agora inclui legenda explicando as barras e o traço de desvio padrão.

    save_path : str ou None
        Caminho completo onde o gráfico será salvo. Se None, apenas exibe.
    """

    if df.empty:
        print("Nenhum dado disponível para gerar gráfico.")
        return

    plt.figure(figsize=(10, 6))

    # Gráfico de barras
    sns.barplot(
        data=df,
        x="Modelo",
        y="Mean F1",
        hue="Modelo",
        palette="viridis",
        dodge=False
    )

    legend_labels = ["F1 médio"]

    # Adiciona barra de erro se existir Std F1
    if "Std F1" in df.columns:
        plt.errorbar(
            x=range(len(df)),
            y=df["Mean F1"],
            yerr=df["Std F1"],
            fmt="none",
            c="black",
            capsize=5,
            label="Desvio padrão do F1"
        )
        legend_labels.append("Desvio padrão do F1")

    # Evita duplicar entrada de legendas por causa do hue
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(df)] + handles[-1:], ["Modelos"] + legend_labels[1:])

    plt.title("Comparação do F1-Score (Média ± Desvio)")
    plt.ylabel("F1-Score")
    plt.ylim(0.8, 1.0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
