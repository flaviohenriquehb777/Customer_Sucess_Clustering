from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling import compute_cluster_score  # noqa: E402


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x*100:.{digits}f}%".replace(".", ",")


def fmt_pct1(x: float) -> str:
    return f"{x*100:.1f}%".replace(".", ",")


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int_dot(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def add_dots(ax, table, colors: dict[int, str], col: int = 0, x_pad: float = 0.015) -> None:
    for r, color in colors.items():
        cell = table[(r, col)]
        x = cell.get_x() + x_pad
        y = cell.get_y() + cell.get_height() / 2
        ax.scatter([x], [y], s=110, color=color, transform=ax.transAxes, zorder=5, edgecolors="none")


def build_outputs(df: pd.DataFrame):
    cluster_to_group = {
        4: "Champions",
        0: "Potenciais",
        1: "Avulsos Engajados",
        2: "Zumbis",
        3: "Churn Iminente",
        5: "Novos",
    }
    colors = {
        4: "#6FE28F",
        0: "#F2D36B",
        1: "#FF9B5A",
        2: "#FF5A6A",
        3: "#9A9A9A",
        5: "#7F77DD",
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

    t1 = pd.DataFrame(
        {
            "Grupo": out["grupo"],
            "Cluster": out["cluster_id"].astype(int).astype(str),
            "Clientes": out["clientes"].astype(int).map(fmt_int_dot),
            "%": out["pct"].map(fmt_pct),
            "log_acessos (med)": out["log_acessos_med"].map(fmt_float),
            "dias_sem_acessar (med)": out["dias_sem_acessar_med"].round().astype(int).astype(str),
        }
    )
    t2 = pd.DataFrame(
        {
            "Grupo": out["grupo"],
            "% nunca_acessou": out["pct_nunca_acessou"].map(fmt_pct1),
            "% recorrente": out["pct_recorrente"].map(fmt_pct1),
            "% parcelado": out["pct_parcelado"].map(fmt_pct1),
            "% cartão": out.get("cartao", 0).map(fmt_pct1),
            "% pix": out.get("pix", 0).map(fmt_pct1),
            "% boleto": out.get("boleto", 0).map(fmt_pct1),
        }
    )

    rank = compute_cluster_score(base, cluster_col="cluster_id").sort_values("ranking")
    leitura = {
        4: "Maior valor relativo e melhor equilíbrio de engajamento/risco",
        0: "Bom perfil; elevar valor percebido e reduzir inatividade",
        5: "Precisa provar valor rápido (evitar virar “Zumbi”)",
        1: "Boa base para conversão em recorrência (monetização)",
        3: "Risco alto por inatividade; retenção imediata",
        2: "Pagam e não usam; prioridade máxima de ativação",
    }
    rank["Grupo"] = rank["cluster_id"].astype(int).map(cluster_to_group)
    rank["Cluster"] = rank["cluster_id"].astype(int).astype(str)
    rank["Score médio (valor/risco)"] = rank["score_medio"].map(lambda x: fmt_float(float(x), digits=3))
    rank["Leitura prática"] = rank["cluster_id"].astype(int).map(leitura)
    t3 = rank[["ranking", "Grupo", "Cluster", "Score médio (valor/risco)", "Leitura prática"]].rename(columns={"ranking": "Ranking"})
    t3["Ranking"] = t3["Ranking"].astype(int).astype(str)

    legend_rows = []
    total = int(out["clientes"].sum())
    for cid in order:
        row = out[out["cluster_id"] == cid].iloc[0].to_dict()
        legend_rows.append(
            {
                "cid": cid,
                "text": f"{cluster_to_group[cid]} (Cluster {cid} | {fmt_pct(row['pct'])}) → "
                + {
                    4: "maior engajamento e maior score de valor/risco; combinação mais saudável de uso + retenção/valor.",
                    0: "recorrentes e engajados, mas com espaço para elevar valor percebido e reduzir inatividade.",
                    1: "engajamento médio/alto, porém sem recorrência (oportunidade de conversão para assinatura).",
                    2: "pagam e não usam (nunca acessam); maior risco de cancelamento e frustração do cliente.",
                    3: "forte sinal de abandono por alta inatividade; exige retenção imediata.",
                    5: "perfil mais “recente/instável” com sinais mistos (ex.: alta incidência de Pix/Boleto e menor renovação); precisa de onboarding e prova rápida de valor.",
                }[cid],
            }
        )

    dot_colors_rows_t1 = {i + 1: colors[cid] for i, cid in enumerate(order)}
    dot_colors_rows_t2 = {i + 1: colors[cid] for i, cid in enumerate(order)}
    dot_colors_rows_t3 = {i + 1: colors[int(c)] for i, c in enumerate(rank["cluster_id"].astype(int).tolist())}

    return t1, t2, t3, legend_rows, dot_colors_rows_t1, dot_colors_rows_t2, dot_colors_rows_t3


def render_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data" / "processed" / "base_cliente_clusterizada.parquet")
    assets = root / "reports" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    t1, t2, t3, legend_rows, dots1, dots2, dots3 = build_outputs(df)

    bg = "#1E1E1E"
    fg = "white"

    colors = {4: "#6FE28F", 0: "#F2D36B", 1: "#FF9B5A", 2: "#FF5A6A", 3: "#9A9A9A", 5: "#7F77DD"}

    fig = plt.figure(figsize=(15.5, 4.2), facecolor=bg)
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.set_facecolor(bg)
    ax0.axis("off")
    ax0.text(0.02, 0.92, "Legenda (nomenclatura “intuitiva” dos 6 grupos)", color=fg, fontsize=18, weight="bold")
    y = 0.72
    for item in legend_rows:
        ax0.scatter([0.03], [y], s=130, color=colors[item["cid"]], transform=ax0.transAxes, edgecolors="none")
        ax0.text(0.055, y, item["text"], color=fg, fontsize=12.8, va="center", transform=ax0.transAxes)
        y -= 0.14
    fig.savefig(assets / "case_legenda.png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)

    fig = plt.figure(figsize=(10.5, 3.6), facecolor=bg)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_facecolor(bg)
    ax1.axis("off")
    ax1.text(0.02, 1.08, "Resumo numérico (visão cliente-atual) — Parte 1", color=fg, fontsize=15, weight="bold", transform=ax1.transAxes)
    col_widths_1 = [0.26, 0.09, 0.14, 0.09, 0.18, 0.24]
    table1 = ax1.table(
        cellText=t1.values.tolist(),
        colLabels=t1.columns.tolist(),
        cellLoc="center",
        loc="center",
        colWidths=col_widths_1,
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10.5)
    table1.scale(1, 1.55)
    for (r, c), cell in table1.get_celld().items():
        cell.set_edgecolor("#3A3A3A")
        cell.set_linewidth(0.8)
        cell.set_facecolor(bg)
        cell.get_text().set_color(fg)
        if r == 0:
            cell.set_facecolor("#2A2A2A")
            cell.get_text().set_weight("bold")
        if c == 0:
            cell._loc = "left"
            cell.get_text().set_ha("left")
    fig.canvas.draw()
    add_dots(ax1, table1, dots1, col=0, x_pad=0.02)
    for r in range(1, len(t1) + 1):
        txt = table1[(r, 0)].get_text().get_text()
        table1[(r, 0)].get_text().set_text("    " + txt)
    fig.savefig(assets / "case_resumo_parte1.png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)

    fig = plt.figure(figsize=(10.5, 3.6), facecolor=bg)
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_facecolor(bg)
    ax2.axis("off")
    ax2.text(0.02, 1.08, "Resumo numérico (visão cliente-atual) — Parte 2", color=fg, fontsize=15, weight="bold", transform=ax2.transAxes)
    col_widths_2 = [0.24, 0.13, 0.13, 0.13, 0.12, 0.12, 0.13]
    table2 = ax2.table(
        cellText=t2.values.tolist(),
        colLabels=t2.columns.tolist(),
        cellLoc="center",
        loc="center",
        colWidths=col_widths_2,
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10.5)
    table2.scale(1, 1.55)
    for (r, c), cell in table2.get_celld().items():
        cell.set_edgecolor("#3A3A3A")
        cell.set_linewidth(0.8)
        cell.set_facecolor(bg)
        cell.get_text().set_color(fg)
        if r == 0:
            cell.set_facecolor("#2A2A2A")
            cell.get_text().set_weight("bold")
        if c == 0:
            cell._loc = "left"
            cell.get_text().set_ha("left")
    fig.canvas.draw()
    add_dots(ax2, table2, dots2, col=0, x_pad=0.02)
    for r in range(1, len(t2) + 1):
        txt = table2[(r, 0)].get_text().get_text()
        table2[(r, 0)].get_text().set_text("    " + txt)
    fig.savefig(assets / "case_resumo_parte2.png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)

    fig = plt.figure(figsize=(15.5, 4.2), facecolor=bg)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.02, 1.10, "Ranking de valor/risco (tomada de decisão)", color=fg, fontsize=15, weight="bold", transform=ax.transAxes)
    col_widths = [0.08, 0.18, 0.08, 0.18, 0.48]
    table3 = ax.table(
        cellText=t3.values.tolist(),
        colLabels=t3.columns.tolist(),
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1, 1.7)
    for (r, c), cell in table3.get_celld().items():
        cell.set_edgecolor("#3A3A3A")
        cell.set_linewidth(0.8)
        cell.set_facecolor(bg)
        cell.get_text().set_color(fg)
        if r == 0:
            cell.set_facecolor("#2A2A2A")
            cell.get_text().set_weight("bold")
        if c in {0, 2, 3}:
            cell._loc = "center"
            cell.get_text().set_ha("center")
    fig.canvas.draw()
    add_dots(ax, table3, dots3, col=1, x_pad=0.02)
    for r in range(1, len(t3) + 1):
        txt = table3[(r, 1)].get_text().get_text()
        table3[(r, 1)].get_text().set_text("   " + txt)

    fig.savefig(assets / "case_ranking_valor_risco.png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    render_assets()

