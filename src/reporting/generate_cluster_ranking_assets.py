from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling import compute_cluster_score  # noqa: E402
from src.reporting.cluster_ranking_visual import salvar_ranking_html  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    df = pd.read_parquet(root / "data" / "processed" / "base_cliente_clusterizada.parquet")

    cluster_to_name = {
        4: "Champions",
        0: "Potenciais",
        1: "Avulsos Engajados",
        2: "Zumbis",
        3: "Churn Iminente",
        5: "Novos",
    }

    counts = df["cluster_id"].value_counts().sort_index()
    pct = (counts / len(df) * 100).to_dict()

    df2 = df.copy()
    df2["recorrente_bin"] = (df2.get("recorrente", 0).fillna(0) == 1).astype(int)
    pay = (
        df2.pivot_table(index="cluster_id", columns="metodo_pagamento", values="cliente", aggfunc="count", fill_value=0)
        .div(df2.groupby("cluster_id").size(), axis=0)
        .fillna(0)
    )

    prof = (
        df2.groupby("cluster_id")
        .agg(
            clientes=("cliente", "size"),
            pct_recorrentes=("recorrente_bin", "mean"),
            pct_nunca_acessou=("nunca_acessou", "mean"),
            dias_sem_acessar=("dias_sem_acessar", "median"),
        )
        .reset_index()
    )
    prof["pct_base"] = prof["cluster_id"].map(pct)
    prof["pct_cartao"] = prof["cluster_id"].map(lambda c: float(pay.get("cartao", pd.Series()).get(c, 0.0)) * 100)
    prof["pct_pix"] = prof["cluster_id"].map(lambda c: float(pay.get("pix", pd.Series()).get(c, 0.0)) * 100)

    rank = compute_cluster_score(df2, cluster_col="cluster_id").sort_values("ranking")
    rank_map = rank.set_index("cluster_id")["ranking"].to_dict()
    score_map = rank.set_index("cluster_id")["score_medio"].to_dict()

    clusters = []
    for cid in sorted(counts.index):
        clusters.append(
            {
                "rank": int(rank_map.get(int(cid))),
                "nome": cluster_to_name.get(int(cid), f"Cluster {int(cid)}"),
                "cluster_id": int(cid),
                "score": float(score_map.get(int(cid))),
                "clientes": int(counts.loc[cid]),
                "pct_base": float(pct.get(int(cid))),
                "pct_recorrentes": float(prof.loc[prof["cluster_id"] == cid, "pct_recorrentes"].iloc[0]) * 100,
                "pct_cartao": float(prof.loc[prof["cluster_id"] == cid, "pct_cartao"].iloc[0]),
                "pct_pix": float(prof.loc[prof["cluster_id"] == cid, "pct_pix"].iloc[0]),
                "pct_nunca_acessou": float(prof.loc[prof["cluster_id"] == cid, "pct_nunca_acessou"].iloc[0]) * 100,
                "dias_sem_acessar": int(round(float(prof.loc[prof["cluster_id"] == cid, "dias_sem_acessar"].iloc[0]))),
            }
        )

    clusters = sorted(clusters, key=lambda x: x["rank"])

    out_dir = root / "reports" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    salvar_ranking_html(
        clusters,
        caminho=str(out_dir / "cluster_ranking_visual.html"),
        titulo="Ranking por score (valor/risco)",
        incluir_legenda=True,
    )


if __name__ == "__main__":
    main()

