from __future__ import annotations

import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from xml.sax.saxutils import escape

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling import compute_cluster_score


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _w(tag: str) -> str:
    return f"w:{tag}"


def _cell(text: str, bold: bool = False) -> str:
    t = escape(str(text))
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:tc>"
        "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
        "<w:p><w:r>"
        f"{rpr}"
        f"<w:t xml:space=\"preserve\">{t}</w:t>"
        "</w:r></w:p>"
        "</w:tc>"
    )


def _row(values: list[str], header: bool = False) -> str:
    cells = "".join(_cell(v, bold=header) for v in values)
    return f"<w:tr>{cells}</w:tr>"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    borders = (
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9D9D9\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9D9D9\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9D9D9\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9D9D9\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E6E6E6\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"E6E6E6\"/>"
        "</w:tblBorders>"
    )
    tbl_pr = f"<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>{borders}</w:tblPr>"
    tbl_grid = "<w:tblGrid>" + "".join("<w:gridCol w:w=\"2400\"/>" for _ in headers) + "</w:tblGrid>"
    head = _row(headers, header=True)
    body = "".join(_row(r, header=False) for r in rows)
    return f"<w:tbl>{tbl_pr}{tbl_grid}{head}{body}</w:tbl>"


def _document_xml(table_xml: str) -> str:
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        f"<w:document xmlns:w=\"{W_NS}\" xmlns:r=\"{R_NS}\">"
        "<w:body>"
        f"{table_xml}"
        f"{sect}"
        "</w:body>"
        "</w:document>"
    )


def _content_types_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
        "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
        "</Types>"
    )


def _rels_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
        "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
        "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>"
        "</Relationships>"
    )


def _doc_rels_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "</Relationships>"
    )


def _core_xml(title: str) -> str:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<cp:coreProperties "
        "xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        f"<dc:title>{escape(title)}</dc:title>"
        "<dc:creator>Trae</dc:creator>"
        f"<cp:revision>1</cp:revision>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>"
        "</cp:coreProperties>"
    )


def _app_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
        "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
        "<Application>Trae</Application>"
        "</Properties>"
    )


def write_docx_with_table(out_path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    table_xml = _table(headers, rows)
    doc_xml = _document_xml(table_xml)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".{uuid4().hex}.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _content_types_xml())
        z.writestr("_rels/.rels", _rels_xml())
        z.writestr("word/document.xml", doc_xml)
        z.writestr("word/_rels/document.xml.rels", _doc_rels_xml())
        z.writestr("docProps/core.xml", _core_xml(title))
        z.writestr("docProps/app.xml", _app_xml())
    tmp.replace(out_path)


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x*100:.{digits}f}%".replace(".", ",")


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int_dot(x: int) -> str:
    return f"{x:,}".replace(",", ".")


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

    resumo_headers = [
        "Grupo",
        "Cluster",
        "Clientes",
        "%",
        "log_acessos (med)",
        "dias_sem_acessar (med)",
        "% nunca_acessou",
        "% recorrente",
        "% parcelado",
        "% cartão",
        "% pix",
        "% boleto",
    ]
    resumo_rows: list[list[str]] = []
    for r in out.itertuples(index=False):
        resumo_rows.append(
            [
                str(r.grupo),
                str(int(r.cluster_id)),
                fmt_int_dot(int(r.clientes)),
                fmt_pct(float(r.pct), digits=2),
                fmt_float(float(r.log_acessos_med), digits=3),
                str(int(round(float(r.dias_sem_acessar_med)))),
                fmt_pct(float(r.pct_nunca_acessou), digits=1),
                fmt_pct(float(r.pct_recorrente), digits=1),
                fmt_pct(float(r.pct_parcelado), digits=1),
                fmt_pct(float(getattr(r, "cartao", 0.0)), digits=1),
                fmt_pct(float(getattr(r, "pix", 0.0)), digits=1),
                fmt_pct(float(getattr(r, "boleto", 0.0)), digits=1),
            ]
        )

    write_docx_with_table(
        out_path=root / "tabela_resumo_numerico.docx",
        title="Resumo numérico (visão cliente-atual)",
        headers=resumo_headers,
        rows=resumo_rows,
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
    rank["grupo"] = rank["cluster_id"].astype(int).map(cluster_to_group)
    rank["leitura_pratica"] = rank["cluster_id"].astype(int).map(leitura)

    rank_headers = ["Ranking", "Grupo", "Cluster", "Score médio (valor/risco)", "Leitura prática"]
    rank_rows: list[list[str]] = []
    for r in rank.itertuples(index=False):
        rank_rows.append(
            [
                str(int(r.ranking)),
                str(r.grupo),
                str(int(r.cluster_id)),
                fmt_float(float(r.score_medio), digits=3),
                str(r.leitura_pratica),
            ]
        )

    write_docx_with_table(
        out_path=root / "tabela_ranking_valor_risco.docx",
        title="Ranking de valor/risco (tomada de decisão)",
        headers=rank_headers,
        rows=rank_rows,
    )


if __name__ == "__main__":
    main()

