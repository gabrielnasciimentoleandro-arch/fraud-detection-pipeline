"""
Nucleo de experimentos do pipeline de deteccao de fraudes.

Reprodutiveis por linha de comando (mesma saida do notebook, sem graficos):
    python3 rodar.py            # dataset completo
    python3 rodar.py sub        # 80k linhas, para smoke test

Estrutura:
  carregar()          -> baixa/le o creditcard.csv
  baseline_ingenuo()  -> RegLog crua, split aleatorio (regua de comparacao)
  melhorado()         -> split temporal + RobustScaler + class_weight
  tecnica_avancada()  -> HistGradientBoosting cru
  gbdt_undersampling()-> HistGradientBoosting + undersampling 1:10 (a tecnica final)
  relatorio()         -> tabela de metricas comparando as quatro

Sem bibliotecas exoticas: so pandas, numpy e scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
CSV_LOCAL = "creditcard.csv"


# ----------------------------------------------------------------------- dados
def carregar(caminho=CSV_LOCAL, url=URL):
    """Le de um CSV local se existir; senao baixa da URL do enunciado."""
    import os

    if os.path.exists(caminho):
        df = pd.read_csv(caminho)
    else:
        df = pd.read_csv(url)
    return df


def split_temporal(df):
    """Dia 0 treina, dia 1 testa. Simula o uso real: o modelo ve o passado e
    julga o futuro. Evita o vazamento do split aleatorio."""
    treino = df[df.Time < 86_400]
    teste = df[df.Time >= 86_400]
    X_col = [c for c in df.columns if c not in ("Class",)]
    return (
        treino[X_col].to_numpy(np.float32),
        teste[X_col].to_numpy(np.float32),
        treino["Class"].to_numpy(np.int8),
        teste["Class"].to_numpy(np.int8),
    )


def split_aleatorio_estratificado(df, frac=0.3, seed=0):
    """O split 'padrao de aluno': embaralha tudo e corta 30% p/ teste."""
    X = df.drop(columns="Class").to_numpy(np.float32)
    y = df["Class"].to_numpy(np.int8)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    i_tr, i_te = next(iter(sss.split(X, y)))
    return X[i_tr], X[i_te], y[i_tr], y[i_te]


def _indice_amount(colunas):
    return [i for i, c in enumerate(colunas) if c == "Amount"]


# ------------------------------------------------------------------ experimentos
def baseline_ingenuo(df):
    """RegLog crua, sem escala, sem tratar desbalanceio, split aleatorio.
    Acuracia alta e inutil: e a regua que mostra por que metrica errada enganou."""
    Xtr, Xte, ytr, yte = split_aleatorio_estratificado(df)
    modelo = LogisticRegression(max_iter=100)  # de proposito: sem escalar Amount, nao converge
    modelo.fit(Xtr, ytr)
    proba = modelo.predict_proba(Xte)[:, 1]
    return yte, proba, modelo


def melhorado(df):
    """Regressao logistica + RobustScaler no Amount + class_weight='balanced'
    + split temporal. Basema honesto da disciplina."""
    colunas = [c for c in df.columns if c != "Class"]
    Xtr, Xte, ytr, yte = split_temporal(df)
    prep = ColumnTransformer(
        [("amount", RobustScaler(), _indice_amount(colunas))],
        remainder="passthrough",
    )
    modelo = Pipeline(
        [
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    modelo.fit(Xtr, ytr)
    return yte, modelo.predict_proba(Xte)[:, 1], modelo


def tecnica_avancada(df):
    """HistGradientBoosting: arvores, nao precisa de escala, lida com
    desbalanceio via +muitas arvores rasas. Costuma ser o melhor custo/beneficio
    neste dataset (fica perto do top do leaderboard sabendo so V1..V28)."""
    Xtr, Xte, ytr, yte = split_temporal(df)
    modelo = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=0,
    )
    modelo.fit(Xtr, ytr)
    return yte, modelo.predict_proba(Xte)[:, 1], modelo


def gbdt_undersampling(df):
    """Tecnica final: o mesmo GBDT, treinado sobre todas as fraudes do dia 1 e
    apenas 10x de transacoes legitimas. E o que da o salto real de PR-AUC/F1
    (mesma amostragem e hiperparametros da secao 9 do notebook)."""
    Xtr, Xte, ytr, yte = split_temporal(df)
    rng = np.random.default_rng(0)
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    amostra = np.sort(np.concatenate([pos, rng.choice(neg, len(pos) * 10, replace=False)]))
    modelo = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=0,
    )
    modelo.fit(Xtr[amostra], ytr[amostra])
    return yte, modelo.predict_proba(Xte)[:, 1], modelo


# --------------------------------------------------------------------- metricas
def melhor_f1_em_0a1(y, proba):
    limiares = np.linspace(0.0, 1.0, 1001)
    melhor = (0.0, 0.0, 0.0, 0.0)
    for t in limiares:
        pred = (proba >= t).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > melhor[2]:
            melhor = (float(t), prec, f1, int(tp) + int(fn))
    return melhor  # limiar, precisao, f1, total de fraudes


def relatorio(resultado, rotulo):
    y, proba, modelo = resultado
    tn, fp, fn, tp = confusion_matrix(y, proba >= 0.5, labels=[0, 1]).ravel()
    limiar, prec_t, f1_t, n_fraudes = melhor_f1_em_0a1(y, proba)
    return {
        "modelo": rotulo,
        "acuracia@0.5": (tp + tn) / len(y),
        "precision@0.5": precision_score(y, proba >= 0.5, zero_division=0),
        "recall@0.5": recall_score(y, proba >= 0.5),
        "f1@0.5": f1_score(y, proba >= 0.5),
        "falsos positivos": int(fp),
        "fraudes pegas": f"{int(tp)}/{int(n_fraudes)}",
        "PR-AUC": average_precision_score(y, proba),
        "ROC-AUC": roc_auc_score(y, proba),
        "limiar otimo": limiar,
        "f1 no limiar otimo": f1_t,
        "precision no limiar otimo": prec_t,
    }


def imprimir_tabela(linhas):
    df = pd.DataFrame(linhas).set_index("modelo")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        for c in ("acuracia@0.5", "precision@0.5", "recall@0.5", "f1@0.5",
                  "PR-AUC", "ROC-AUC", "limiar otimo", "f1 no limiar otimo",
                  "precision no limiar otimo"):
            df[c] = df[c].map(lambda v: f"{v:.4f}")
        print(df.to_string())
    return df


def graficos(pares, caminho=None):
    """Curva precisao-recuperacao comparando os modelos (a que importa aqui).
    pares: lista de (rotulo, y_teste, proba_teste)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 4.2))
    y0 = pares[0][1]
    ax.axhline(y0.mean(), ls="--", c="grey", lw=1,
               label=f"chute aleatorio (P={y0.mean():.4f})")
    for nome, y, proba in pares:
        p, r, _ = precision_recall_curve(y, proba)
        ax.plot(r, p, label=nome)
    ax.set(xlabel="recall", ylabel="precision",
           title="Precision-Recall (quanto mais arriba/direita, melhor)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=130)
    return fig
