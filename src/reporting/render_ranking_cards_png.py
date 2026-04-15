from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling import compute_cluster_score  # noqa: E402


def fmt_int_dot(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%".replace(".", ",")


def fmt_score(x: float) -> str:
    if x >= 0:
        return f"+{x:.3f}".replace(".", ",")
    return f"{x:.3f}".replace(".", ",")


def bar_width(score: float, smin: float, smax: float) -> float:
    span = smax - smin
    if span == 0:
        return 0.6
    pct = (score - smin) / span
    return float(np.clip(0.10 + pct * 0.88, 0.08, 0.98))


def render() -> Path:
    root = Path(__file__).resolve().parents[2]
    assets = root / "reports" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / "data" / "processed" / "base_cliente_clusterizada.parquet")

    cluster_to_name = {
        4: "Champions",
        0: "Potenciais",
        5: "Novos",
        1: "Avulsos Engajados",
        3: "Churn Iminente",
        2: "Zumbis",
    }
    cluster_to_color = {
        4: {"accent": "#6FE28F", "pill_bg": "#EAF3DE", "pill_txt": "#27500A"},
        0: {"accent": "#F2D36B", "pill_bg": "#FAEEDA", "pill_txt": "#633806"},
        5: {"accent": "#7F77DD", "pill_bg": "#EEEDFE", "pill_txt": "#3C3489"},
        1: {"accent": "#FF9B5A", "pill_bg": "#FAECE7", "pill_txt": "#712B13"},
        3: {"accent": "#9A9A9A", "pill_bg": "#E3E3E3", "pill_txt": "#3E3E3E"},
        2: {"accent": "#E24B4A", "pill_bg": "#FCEBEB", "pill_txt": "#791F1F"},
    }

    rank = compute_cluster_score(df, cluster_col="cluster_id").sort_values("ranking")
    scores = rank["score_medio"].astype(float).tolist()
    smin, smax = min(scores), max(scores)

    counts = df["cluster_id"].value_counts()
    pct = (counts / len(df) * 100).to_dict()
    df2 = df.copy()
    df2["recorrente_bin"] = (df2.get("recorrente", 0).fillna(0) == 1).astype(int)
    prof = df2.groupby("cluster_id").agg(
        pct_rec=("recorrente_bin", "mean"),
        dias_sem=("dias_sem_acessar", "median"),
        pct_cartao=("metodo_pagamento", lambda s: (s == "cartao").mean()),
        pct_pix=("metodo_pagamento", lambda s: (s == "pix").mean()),
        pct_nunca=("nunca_acessou", "mean"),
    )

    cards = []
    for r in rank.itertuples(index=False):
        cid = int(r.cluster_id)
        cards.append(
            {
                "ranking": int(r.ranking),
                "cluster_id": cid,
                "nome": cluster_to_name.get(cid, f"Cluster {cid}"),
                "score": float(r.score_medio),
                "clientes": int(counts.get(cid, 0)),
                "pct_base": float(pct.get(cid, 0.0)),
                "pct_rec": float(prof.loc[cid, "pct_rec"]) * 100 if cid in prof.index else 0.0,
                "pct_cartao": float(prof.loc[cid, "pct_cartao"]) * 100 if cid in prof.index else 0.0,
                "pct_pix": float(prof.loc[cid, "pct_pix"]) * 100 if cid in prof.index else 0.0,
                "pct_nunca": float(prof.loc[cid, "pct_nunca"]) * 100 if cid in prof.index else 0.0,
                "dias_sem": float(prof.loc[cid, "dias_sem"]) if cid in prof.index else 0.0,
            }
        )

    card_bg = "#2A2A2A"
    card_border = "#3A3A3A"
    text_main = "white"
    text_sub = "#C7C7C7"
    bar_bg = "#1F1F1F"

    fig = plt.figure(figsize=(12.6, 9.2), facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.965, "RANKING POR SCORE (VALOR/RISCO)", color="#BFBFBF", fontsize=12, weight="bold")
    ax.plot([0.04, 0.96], [0.942, 0.942], color="#2A2A2A", linewidth=1)

    top = 0.90
    card_h = 0.115
    gap = 0.022
    left = 0.05
    right = 0.95
    width = right - left

    for i, c in enumerate(cards):
        y = top - i * (card_h + gap) - card_h
        cid = c["cluster_id"]
        palette = cluster_to_color[cid]
        accent = palette["accent"]

        rbox = FancyBboxPatch(
            (left, y),
            width,
            card_h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1,
            edgecolor=card_border,
            facecolor=card_bg,
        )
        ax.add_patch(rbox)
        ax.plot([left, left], [y, y + card_h], color=accent, linewidth=3, solid_capstyle="round")

        cx = left + 0.035
        cy = y + card_h / 2
        ax.scatter([cx], [cy], s=650, color="#EDE7DD", zorder=5)
        ax.text(cx, cy, str(c["ranking"]), color="#3A3A3A", fontsize=12, weight="bold", ha="center", va="center", zorder=6)

        title_x = left + 0.085
        ax.text(
            title_x,
            y + card_h * 0.72,
            f"{c['nome']}",
            color=text_main,
            fontsize=14,
            weight="bold",
            va="center",
        )
        ax.text(
            title_x + 0.11,
            y + card_h * 0.72,
            f"— Cluster {cid}",
            color="#A9A9A9",
            fontsize=12,
            weight="bold",
            va="center",
        )

        parts = [f"{fmt_int_dot(c['clientes'])} clientes", fmt_pct(c["pct_base"], 1)]
        if c["pct_rec"] > 0:
            parts.append(f"{fmt_pct(c['pct_rec'], 1)} recorrentes")
        if c["pct_cartao"] >= 99:
            parts.append("100% cartão")
        elif c["pct_pix"] >= 50:
            parts.append(f"{fmt_pct(c['pct_pix'], 1)} pix")
        if c["pct_nunca"] >= 99:
            parts.append("100% nunca acessou")
        if c["dias_sem"] >= 150:
            parts.append(f"{int(round(c['dias_sem']))} dias sem acessar")

        ax.text(
            title_x,
            y + card_h * 0.52,
            " · ".join(parts),
            color=text_sub,
            fontsize=12,
            va="center",
        )

        bar_x0 = title_x
        bar_y = y + card_h * 0.23
        bar_w = width * 0.72
        bar_h = 0.010
        ax.add_patch(
            FancyBboxPatch(
                (bar_x0, bar_y),
                bar_w,
                bar_h,
                boxstyle="round,pad=0,rounding_size=0.004",
                linewidth=0,
                facecolor=bar_bg,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (bar_x0, bar_y),
                bar_w * bar_width(c["score"], smin, smax),
                bar_h,
                boxstyle="round,pad=0,rounding_size=0.004",
                linewidth=0,
                facecolor=accent,
            )
        )
        ax.text(bar_x0 + bar_w + 0.02, bar_y + bar_h / 2, fmt_score(c["score"]), color="#AFAFAF", fontsize=11, va="center")

        pill_w = 0.09
        pill_h = 0.038
        pill_x = right - pill_w - 0.02
        pill_y = y + card_h * 0.62
        ax.add_patch(
            FancyBboxPatch(
                (pill_x, pill_y),
                pill_w,
                pill_h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=0,
                facecolor=palette["pill_bg"],
            )
        )
        ax.text(
            pill_x + pill_w / 2,
            pill_y + pill_h / 2,
            fmt_score(c["score"]),
            color=palette["pill_txt"],
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

    legend_y = 0.035
    legend_items = [
        (cluster_to_color[4]["accent"], "Verde — excelência"),
        (cluster_to_color[0]["accent"], "Âmbar — potencial"),
        (cluster_to_color[5]["accent"], "Roxo — neutro/incipiente"),
        (cluster_to_color[1]["accent"], "Coral — atenção moderada"),
        (cluster_to_color[3]["accent"], "Cinza — risco crescente"),
        (cluster_to_color[2]["accent"], "Vermelho — alerta máximo"),
    ]
    x = 0.06
    for color, label in legend_items:
        ax.scatter([x], [legend_y], s=55, color=color)
        ax.text(x + 0.015, legend_y, label, color="#BFBFBF", fontsize=11, va="center")
        x += 0.18 if "Vermelho" not in label else 0.20

    out_path = assets / "cluster_ranking_visual.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    render()

