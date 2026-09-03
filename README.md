# Detecção de fraudes em transações

Pipeline de triagem de fraude em `creditcard.csv` (284.807 transações), comparando
**quatro técnicas** sob avaliação temporal honesta — não sob a métrica que o enunciado sugere.

> Estudo independente sobre dado público e anonimizado — sem PII, com métricas,
> limitações e decisões de projeto documentadas.

## 1. O problema real

Rotular fraude é o clássico **problema de classe rara**: 492 casos em 284.807 (**0,17%**).

| Armadilha | Consequência |
|---|---|
| Usar acurácia | um classificador que chuta "normal" sempre marca **99,83%** e não pega nada |
| Split aleatório | `Time` é cronológico; embaralhar faz o modelo "ver" transações vizinhas e **infla a nota** |

Custo assimétrico: **falso negativo** = fraude aprovada (prejuízo); **falso positivo** = cliente bloqueado (atrito).

## 2. Metodologia

- **Split temporal**: dia 1 treina (144.786 linhas, 281 fraudes), dia 2 testa (140.021 linhas, 211 fraudes).
- **RobustScaler** só em `Amount` (0 → 25.691, escala que trava modelos lineares).
- **Balanceamento**: `class_weight="balanced"` vs **undersampling 1:10** das transações legítimas.
- **Limiar de decisão** calibrado por maximização de F1 — 0,5 é arbitrário.
- **Métrica principal**: PR-AUC. ROC-AUC, precision/recall/F1 e matriz de confusão como apoio.

## 3. Resultados

Dataset completo, avaliação no **dia 2** (211 fraudes), `scikit-learn 1.6.1`:

| # | Técnica | Acurácia | Precision | Recall | F1 | FP | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline ingênuo (split aleatório) | 0,9989 | 0,7165 | 0,6149 | 0,6618 | 36 | 0,5431 | 0,9065 |
| 2 | LogReg + RobustScaler + `class_weight` | 0,9070 | 0,0148 | 0,9242 | 0,0291 | 13.000 | 0,7657 | 0,9735 |
| 3 | HistGradientBoosting | 0,9992 | 0,7104 | 0,7441 | 0,7269 | 64 | 0,7320 | 0,9822 |
| 4 | **HistGradientBoosting + undersampling 1:10** *(limiar 0,914)* | — | **0,879** | **0,758** | **0,814** | ~25 | **0,7952** | **0,9831** |

**Leituras que valem o projeto:**

1. A técnica 2 tem o **melhor recall** (92%) e o **pior F1** (0,03): 13.000 alertas falsos para pegar 195 fraudes. Inviável em operação.
2. Trocar de modelo (LogReg → GBDT) rende **+1pp de F1**. Tratar o desbalanceio **e** o limiar rende **+7pp**. O ganho está no pré-processamento, não no algoritmo.
3. A acurácia da técnica 1 (99,89%) não a torna melhor que a 4: ela só está mais perto do "chute tudo normal" (99,83%) do que parece.

## 4. Como rodar

Abra `deteccao_fraudes.ipynb` no Google Colab e execute de cima até embaixo (~3 min).
Ele baixa o `creditcard.csv` (144 MB) e cacheia — o arquivo está no `.gitignore` por exceder o limite do GitHub.

Por linha de comando:

```bash
pip install "pandas>=2.2" "scikit-learn>=1.4" matplotlib
python3 rodar.py        # dataset completo
python3 rodar.py sub    # smoke test com 80k linhas
