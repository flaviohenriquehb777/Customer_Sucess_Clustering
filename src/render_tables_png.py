from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling import compute_cluster_score


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x*100:.{digits}f}%".replace(".", ",")


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int_dot(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def render_table_png(
    title: str,
    df_display: pd.DataFrame,
    out_path: Path,
    figsize: tuple[float, float],
    font_size: int = 10,
    col_align: str = "center",
) -> None:
    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=df_display.values.tolist(),
        colLabels=df_display.columns.tolist(),
        cellLoc=col_align,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold")

    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data" / "processed" / "base_cliente_clusterizada.parquet")

    cluster_to_group = {
        4: "🟢 Champions",
        0: "🟡 Potenciais",
        1: "🟠 Avulsos Engajados",
        2: "🔴 Zumbis",
        3: "⚫ Churn Iminente",
        5: "🔵 Novos",
    }
    order = [4, 0, 1, 2, 3, 5]

    base = df.copy()
    base["recorrente_bin"] = (base.get("recorrente", 0).fillna(0) == 1).astype(int)

    agg = (
        base.groupby("cluster_id")
        .agg(
            clientes=("cliente", "size"),
            log_acessos_med=("log_acessos", "median"),
            dias_sem_acessar_med=("dias_sem_acessar", "median"),
            pct_nunca_acessou=("nunca_acessou", "mean"),
            pct_recorrente=("recorrente_bin", "mean"),
            pct_parcelado=("parcelado", "mean"),
        )
        .reset_index()
    )
    agg["pct"] = agg["clientes"] / agg["clientes"].sum()

    pay = base.pivot_table(index="cluster_id", columns="metodo_pagamento", values="cliente", aggfunc="count", fill_value=0)
    pay = pay.div(pay.sum(axis=1), axis=0).reset_index()

    out = agg.merge(pay, on="cluster_id", how="left").fillna(0)
    out["grupo"] = out["cluster_id"].astype(int).map(cluster_to_group)
    out = out.set_index("cluster_id").loc[order].reset_index()

    resumo = pd.DataFrame(
        {
            "Grupo": out["grupo"],
            "Cluster": out["cluster_id"].astype(int).astype(str),
            "Clientes": out["clientes"].astype(int).map(fmt_int_dot),
            "%": out["pct"].map(lambda x: fmt_pct(x, digits=2)),
            "log_acessos (med)": out["log_acessos_med"].map(fmt_float),
            "dias_sem_acessar (med)": out["dias_sem_acessar_med"].round().astype(int).astype(str),
            "% nunca_acessou": out["pct_nunca_acessou"].map(fmt_pct),
            "% recorrente": out["pct_recorrente"].map(fmt_pct),
            "% parcelado": out["pct_parcelado"].map(fmt_pct),
            "% cartão": out.get("cartao", 0).map(fmt_pct),
            "% pix": out.get("pix", 0).map(fmt_pct),
            "% boleto": out.get("boleto", 0).map(fmt_pct),
        }
    )

    render_table_png(
        title="Resumo numérico (visão cliente-atual)",
        df_display=resumo,
        out_path=root / "resumo_numerico.png",
        figsize=(18, 3.8),
        font_size=10,
        col_align="center",
    )

    rank = compute_cluster_score(base, cluster_col="cluster_id").sort_values("ranking")
    rank["grupo"] = rank["cluster_id"].astype(int).map(cluster_to_group)
    leitura = {
        4: "Maior valor relativo e melhor equilíbrio de engajamento/risco",
        0: "Bom perfil; elevar valor percebido e reduzir inatividade",
        5: "Precisa provar valor rápido (evitar virar “Zumbi”)",
        1: "Boa base para conversão em recorrência (monetização)",
        3: "Risco alto por inatividade; retenção imediata",
        2: "Pagam e não usam; prioridade máxima de ativação",
    }
    rank["leitura_pratica"] = rank["cluster_id"].astype(int).map(leitura)

    ranking_tbl = pd.DataFrame(
        {
            "Ranking": rank["ranking"].astype(int).astype(str),
            "Grupo": rank["grupo"],
            "Cluster": rank["cluster_id"].astype(int).astype(str),
            "Score médio": rank["score_medio"].map(lambda x: fmt_float(float(x), digits=3)),
            "Leitura prática": rank["leitura_pratica"],
        }
    )

    render_table_png(
        title="Ranking de valor/risco (tomada de decisão)",
        df_display=ranking_tbl,
        out_path=root / "ranking_valor_risco.png",
        figsize=(18, 3.2),
        font_size=10,
        col_align="left",
    )

    print("Gerado:", str((root / "resumo_numerico.png").resolve()))
    print("Gerado:", str((root / "ranking_valor_risco.png").resolve()))


if __name__ == "__main__":
    main()

