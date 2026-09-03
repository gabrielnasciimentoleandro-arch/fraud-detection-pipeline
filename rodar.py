import sys, time
sys.path.insert(0, ".")
import pandas as pd
import fraude

SUB = len(sys.argv) > 1 and sys.argv[1] == "sub"
df = fraude.carregar()
if SUB:
    d0 = df[df.Time < 86400].head(40000)
    d1 = df[df.Time >= 86400].head(40000)
    df = pd.concat([d0, d1], ignore_index=True)
    print("** MODO SUBSET:", df.shape, "fraudes:", int(df.Class.sum()), "**")

linhas, pares = [], []
for rotulo, fn in [("baseline_ingenuo (aleatorio)", fraude.baseline_ingenuo),
                   ("logreg+balanced (temporal)", fraude.melhorado),
                   ("HistGBDT (temporal)", fraude.tecnica_avancada),
                   ("HistGBDT+undersample 1:10", fraude.gbdt_undersampling)]:
    t = time.time()
    res = fn(df)
    linhas.append(fraude.relatorio(res, rotulo))
    pares.append((rotulo, res[0], res[1]))
    print(f"ok {rotulo} em {time.time()-t:.1f}s")

suf = "_sub" if SUB else ""
tabela = fraude.imprimir_tabela(linhas)
tabela.to_csv("metricas%s.csv" % suf)
fraude.graficos(pares, "pr_curve%s.png" % suf)
print("salvou metricas%s.csv e pr_curve%s.png" % (suf, suf))
