"""
Gerador de Ranking Visual de Clusters (Segmentação de Clientes)
Compatível com agentes de IA — retorna HTML pronto para exibição.

Uso básico:
    from cluster_ranking_visual import gerar_ranking_html, CLUSTERS_DEFAULT
    html = gerar_ranking_html(CLUSTERS_DEFAULT)

Uso com dados dinâmicos do seu modelo:
    clusters = [
        {
            "rank": 1,
            "nome": "Champions",
            "cluster_id": 4,
            "score": 0.902,
            "clientes": 4175,
            "pct_base": 9.27,
            "pct_recorrentes": 37.6,
            "pct_cartao": 97.8,
            "pct_pix": 0.0,
            "pct_nunca_acessou": 2.0,
            "dias_sem_acessar": 130,
        },
        ...
    ]
    html = gerar_ranking_html(clusters)
"""

from __future__ import annotations
from typing import Optional


CLUSTERS_DEFAULT = [
    {
        "rank": 1,
        "nome": "Champions",
        "cluster_id": 4,
        "score": 0.902,
        "clientes": 4175,
        "pct_base": 9.27,
        "pct_recorrentes": 37.6,
        "pct_cartao": 97.8,
        "pct_pix": 0.0,
        "pct_nunca_acessou": 2.0,
        "dias_sem_acessar": 130,
    },
    {
        "rank": 2,
        "nome": "Potenciais",
        "cluster_id": 0,
        "score": 0.366,
        "clientes": 9709,
        "pct_base": 21.56,
        "pct_recorrentes": 100.0,
        "pct_cartao": 100.0,
        "pct_pix": 0.0,
        "pct_nunca_acessou": 7.6,
        "dias_sem_acessar": 88,
    },
    {
        "rank": 3,
        "nome": "Novos",
        "cluster_id": 5,
        "score": -0.011,
        "clientes": 4041,
        "pct_base": 8.97,
        "pct_recorrentes": 35.4,
        "pct_cartao": 0.0,
        "pct_pix": 90.5,
        "pct_nunca_acessou": 10.7,
        "dias_sem_acessar": 94,
    },
    {
        "rank": 4,
        "nome": "Avulsos Engajados",
        "cluster_id": 1,
        "score": -0.138,
        "clientes": 16126,
        "pct_base": 35.81,
        "pct_recorrentes": 0.0,
        "pct_cartao": 100.0,
        "pct_pix": 0.0,
        "pct_nunca_acessou": 5.8,
        "dias_sem_acessar": 90,
    },
    {
        "rank": 5,
        "nome": "Churn Iminente",
        "cluster_id": 3,
        "score": -0.165,
        "clientes": 5475,
        "pct_base": 12.16,
        "pct_recorrentes": 84.2,
        "pct_cartao": 99.5,
        "pct_pix": 0.0,
        "pct_nunca_acessou": 4.1,
        "dias_sem_acessar": 201,
    },
    {
        "rank": 6,
        "nome": "Zumbis",
        "cluster_id": 2,
        "score": -0.754,
        "clientes": 5501,
        "pct_base": 12.22,
        "pct_recorrentes": 52.2,
        "pct_cartao": 94.2,
        "pct_pix": 4.2,
        "pct_nunca_acessou": 100.0,
        "dias_sem_acessar": 0,
    },
]


PALETA_POR_RANK = {
    1: {
        "borda": "#3B6D11",
        "badge_bg": "#EAF3DE",
        "badge_txt": "#27500A",
        "barra": "#639922",
        "pill_bg": "#EAF3DE",
        "pill_txt": "#27500A",
        "legenda": "Verde — excelência",
    },
    2: {
        "borda": "#BA7517",
        "badge_bg": "#FAEEDA",
        "badge_txt": "#633806",
        "barra": "#EF9F27",
        "pill_bg": "#FAEEDA",
        "pill_txt": "#633806",
        "legenda": "Âmbar — potencial",
    },
    3: {
        "borda": "#534AB7",
        "badge_bg": "#EEEDFE",
        "badge_txt": "#3C3489",
        "barra": "#7F77DD",
        "pill_bg": "#EEEDFE",
        "pill_txt": "#3C3489",
        "legenda": "Roxo — neutro/incipiente",
    },
    4: {
        "borda": "#D85A30",
        "badge_bg": "#FAECE7",
        "badge_txt": "#712B13",
        "barra": "#D85A30",
        "pill_bg": "#FAECE7",
        "pill_txt": "#712B13",
        "legenda": "Coral — atenção moderada",
    },
    5: {
        "borda": "#5F5E5A",
        "badge_bg": "#D3D1C7",
        "badge_txt": "#444441",
        "barra": "#888780",
        "pill_bg": "#D3D1C7",
        "pill_txt": "#444441",
        "legenda": "Cinza — risco crescente",
    },
    6: {
        "borda": "#A32D2D",
        "badge_bg": "#FCEBEB",
        "badge_txt": "#791F1F",
        "barra": "#E24B4A",
        "pill_bg": "#FCEBEB",
        "pill_txt": "#791F1F",
        "legenda": "Vermelho — alerta máximo",
    },
}


def _subtitulo(c: dict) -> str:
    partes = [
        f"{c['clientes']:,} clientes".replace(",", "."),
        f"{c['pct_base']:.1f}%",
    ]
    if c.get("pct_recorrentes", 0) > 0:
        partes.append(f"{c['pct_recorrentes']:.1f}% recorrentes")
    if c.get("pct_nunca_acessou", 0) >= 99:
        partes.append("100% nunca acessou")
    if c.get("pct_cartao", 0) >= 99:
        partes.append("100% cartão")
    elif c.get("pct_pix", 0) > 50:
        partes.append(f"{c['pct_pix']:.1f}% pix")
    if c.get("dias_sem_acessar", 0) > 150:
        partes.append(f"{c['dias_sem_acessar']} dias sem acessar")
    return " · ".join(partes)


def _largura_barra(score: float, score_min: float, score_max: float) -> int:
    span = score_max - score_min
    if span == 0:
        return 50
    pct = int(((score - score_min) / span) * 88 + 10)
    return max(8, min(98, pct))


def _formatar_score(score: float) -> str:
    return f"+{score:.3f}" if score >= 0 else f"{score:.3f}"


def _card_html(c: dict, largura: int) -> str:
    rank = c["rank"]
    cor = PALETA_POR_RANK.get(rank, PALETA_POR_RANK[6])
    subtitulo = _subtitulo(c)
    score_fmt = _formatar_score(c["score"])

    return f"""
  <div style="background:#ffffff;border:0.5px solid #e2e0d8;border-radius:12px;
              padding:14px 16px;margin-bottom:10px;display:grid;
              grid-template-columns:36px 1fr auto;align-items:center;gap:12px;
              border-left:3px solid {cor['borda']};">
    <div style="width:30px;height:30px;border-radius:50%;display:flex;
                align-items:center;justify-content:center;font-weight:500;
                font-size:13px;flex-shrink:0;
                background:{cor['badge_bg']};color:{cor['badge_txt']};">{rank}</div>
    <div>
      <div style="font-weight:500;font-size:14px;color:#1a1a18;">
        {c['nome']}
        <span style="color:#73726c;font-weight:400;font-size:12px;">— Cluster {c['cluster_id']}</span>
      </div>
      <div style="font-size:12px;color:#73726c;margin-top:2px;">{subtitulo}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:7px;">
        <div style="flex:1;height:5px;background:#f1efe8;border-radius:3px;overflow:hidden;">
          <div style="width:{largura}%;height:100%;border-radius:3px;background:{cor['barra']};"></div>
        </div>
        <span style="font-size:11px;color:#73726c;min-width:42px;text-align:right;">{score_fmt}</span>
      </div>
    </div>
    <div style="font-size:12px;font-weight:500;padding:4px 10px;border-radius:20px;
                white-space:nowrap;background:{cor['pill_bg']};color:{cor['pill_txt']};">{score_fmt}</div>
  </div>"""


def _legenda_html() -> str:
    itens = [
        ("#639922", "Verde — excelência"),
        ("#EF9F27", "Âmbar — potencial"),
        ("#7F77DD", "Roxo — neutro/incipiente"),
        ("#D85A30", "Coral — atenção moderada"),
        ("#888780", "Cinza — risco crescente"),
        ("#E24B4A", "Vermelho — alerta máximo"),
    ]
    spans = "".join(
        f'<span style="font-size:11px;display:flex;align-items:center;gap:5px;color:#73726c;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{cor};display:inline-block;"></span>'
        f'{label}</span>'
        for cor, label in itens
    )
    return (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;'
        f'padding-top:12px;border-top:0.5px solid #e2e0d8;">{spans}</div>'
    )


def gerar_ranking_html(
    clusters: list[dict],
    titulo: str = "Ranking por score (valor/risco)",
    incluir_legenda: bool = True,
) -> str:
    """
    Gera o HTML completo do ranking de clusters.

    Parâmetros
    ----------
    clusters : list[dict]
        Lista de clusters ordenados por rank. Cada dict deve conter:
        rank, nome, cluster_id, score, clientes, pct_base,
        pct_recorrentes, pct_cartao, pct_pix, pct_nunca_acessou, dias_sem_acessar
    titulo : str
        Título exibido no topo do ranking.
    incluir_legenda : bool
        Se True, adiciona legenda de cores ao final.

    Retorna
    -------
    str
        String HTML pronta para ser inserida em um arquivo, email, dashboard ou
        retornada como resposta de um agente de IA.
    """
    clusters_ord = sorted(clusters, key=lambda x: x["rank"])
    scores = [c["score"] for c in clusters_ord]
    score_min, score_max = min(scores), max(scores)

    cards = "".join(
        _card_html(c, _largura_barra(c["score"], score_min, score_max))
        for c in clusters_ord
    )

    legenda = _legenda_html() if incluir_legenda else ""

    total_clientes = sum(c["clientes"] for c in clusters_ord)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ranking de Clusters</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #f8f7f2; padding: 20px; }}
  .wrapper {{ max-width: 720px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: baseline;
             margin-bottom: 16px; padding-bottom: 10px; border-bottom: 0.5px solid #e2e0d8; }}
  .header-title {{ font-size: 13px; font-weight: 500; color: #73726c;
                   letter-spacing: 0.05em; text-transform: uppercase; }}
  .header-total {{ font-size: 12px; color: #888780; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <span class="header-title">{titulo}</span>
    <span class="header-total">{total_clientes:,} clientes totais</span>
  </div>
  {cards}
  {legenda}
</div>
</body>
</html>"""


def salvar_ranking_html(
    clusters: list[dict],
    caminho: str = "ranking_clusters.html",
    **kwargs,
) -> str:
    """Gera e salva o HTML em disco. Retorna o caminho do arquivo."""
    html = gerar_ranking_html(clusters, **kwargs)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html)
    return caminho


if __name__ == "__main__":
    caminho = salvar_ranking_html(CLUSTERS_DEFAULT)
    print(f"Arquivo gerado: {caminho}")
