from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def load_xlsx_base_clientes(path: str, preferred_sheet: str = "BaseClientes") -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = preferred_sheet if preferred_sheet in xl.sheet_names else xl.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def to_snake_case(name: str) -> str:
    name = str(name)
    name = _strip_accents(name)
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [to_snake_case(c) for c in out.columns]
    return out


def normalize_yes_no(value: Any) -> Any:
    if pd.isna(value):
        return value
    s = str(value).strip().lower()
    s = _strip_accents(s)
    if s in {"sim", "s", "yes", "y", "true", "1"}:
        return "sim"
    if s in {"nao", "não", "n", "no", "false", "0"}:
        return "nao"
    if s in {"nao identificado", "não identificado", "indefinido", "desconhecido", "unknown"}:
        return "nao_identificado"
    return s


def coerce_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def data_quality_report(df: pd.DataFrame, sample_n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        examples = s.dropna().astype(str).head(sample_n).tolist()
        rows.append(
            {
                "coluna": col,
                "dtype": str(s.dtype),
                "n_linhas": n,
                "n_missing": missing,
                "pct_missing": missing / n if n else np.nan,
                "n_unique": nunique,
                "exemplos": examples,
            }
        )
    rep = pd.DataFrame(rows).sort_values(["pct_missing", "n_unique"], ascending=[False, False])
    return rep


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def winsorize_df(df: pd.DataFrame, cols: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = winsorize_series(out[col], lower_q=lower_q, upper_q=upper_q)
    return out


def make_inactivity_bins(dias_sem_acessar: pd.Series) -> pd.Categorical:
    bins = [-np.inf, 7, 30, 90, 180, np.inf]
    labels = ["0_7", "8_30", "31_90", "91_180", "181_plus"]
    return pd.cut(dias_sem_acessar, bins=bins, labels=labels)


@dataclass(frozen=True)
class CleaningMetadata:
    dropped_columns: list[str]
    created_flags: list[str]


def clean_base(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningMetadata]:
    df = standardize_columns(df_raw)

    yes_no_cols = [c for c in ["ativo", "finalizou_curso", "renovacao"] if c in df.columns]
    for col in yes_no_cols:
        df[col] = df[col].map(normalize_yes_no)

    date_cols = [
        "data_ordem",
        "data_aprovacao",
        "ultimo_acesso",
        "criado_na_wati",
        "boas_vindas",
        "acomp_mensal",
        "epoca_nps",
        "epoca_fup",
        "um_ano_apos_data_compra",
    ]
    df = coerce_datetime(df, date_cols)

    dropped: list[str] = []
    flags: list[str] = []

    if "atual_de_conclusao" in df.columns:
        if (df["atual_de_conclusao"].fillna(0) == 0).all():
            dropped.append("atual_de_conclusao")
    if "tem_comunidade" in df.columns:
        if df["tem_comunidade"].isna().all():
            dropped.append("tem_comunidade")

    for c in df.columns:
        if c in dropped:
            continue
        if df[c].isna().mean() >= 0.95:
            flag = f"teve_{c}"
            df[flag] = (~df[c].isna()).astype(int)
            flags.append(flag)
            dropped.append(c)

    constant_cols = []
    for c in df.columns:
        if c in dropped:
            continue
        if df[c].nunique(dropna=False) <= 1:
            constant_cols.append(c)
    dropped.extend(constant_cols)

    dropped = sorted(set(dropped))
    df = df.drop(columns=[c for c in dropped if c in df.columns], errors="ignore")

    if "ultimo_acesso" in df.columns:
        df["nunca_acessou"] = df["ultimo_acesso"].isna().astype(int)
        flags.append("nunca_acessou")

    if "total_parcelas" in df.columns:
        df["parcelado"] = (pd.to_numeric(df["total_parcelas"], errors="coerce").fillna(0) > 1).astype(int)
        flags.append("parcelado")

    if "n_acessos" in df.columns:
        df["log_acessos"] = np.log1p(pd.to_numeric(df["n_acessos"], errors="coerce"))
        flags.append("log_acessos")

    if "dias_sem_acessar" in df.columns:
        df["faixa_inatividade"] = make_inactivity_bins(pd.to_numeric(df["dias_sem_acessar"], errors="coerce"))
        flags.append("faixa_inatividade")

    if "tipo_pagamento" in df.columns:
        df["tipo_pagamento"] = pd.to_numeric(df["tipo_pagamento"], errors="coerce")

    if "pagamento_tratado" in df.columns:
        df["pagamento_tratado"] = df["pagamento_tratado"].astype(str).map(lambda x: _strip_accents(str(x)).strip().lower())

    if "pagamento" in df.columns:
        df["pagamento"] = df["pagamento"].astype(str).map(lambda x: _strip_accents(str(x)).strip().lower())

    if "pagamento" in df.columns or "pagamento_tratado" in df.columns:
        base = (
            df.get("pagamento", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
            + " "
            + df.get("pagamento_tratado", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
        ).str.lower()
        base = base.map(_strip_accents)
        metodo = pd.Series("outro", index=df.index, dtype="object")
        metodo.loc[base.str.contains("pix", na=False)] = "pix"
        metodo.loc[base.str.contains("boleto", na=False)] = "boleto"
        metodo.loc[base.str.contains("credit", na=False) | base.str.contains("card", na=False) | base.str.contains("cart", na=False)] = "cartao"
        metodo.loc[base.str.contains("deb", na=False)] = "cartao"
        metodo.loc[base.str.strip() == ""] = np.nan
        df["metodo_pagamento"] = metodo
        flags.append("metodo_pagamento")

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper()

    if "recorrente" in df.columns:
        df["recorrente"] = pd.to_numeric(df["recorrente"], errors="coerce")

    if "total_parcelas" in df.columns:
        df["total_parcelas"] = pd.to_numeric(df["total_parcelas"], errors="coerce")
        if "recorrente" in df.columns:
            non_rec = df["recorrente"].fillna(0) != 1
        else:
            non_rec = pd.Series(True, index=df.index)
        parcelas_nr = df["total_parcelas"].where(non_rec)
        bins = [-np.inf, 1, 3, 6, np.inf]
        labels = ["1", "2_3", "4_6", "7_plus"]
        df["parcelas_nao_recorrente_bin"] = pd.cut(parcelas_nr, bins=bins, labels=labels)
        flags.append("parcelas_nao_recorrente_bin")

    if "n_acessos" in df.columns:
        df["n_acessos"] = pd.to_numeric(df["n_acessos"], errors="coerce")

    if "dias_sem_acessar" in df.columns:
        df["dias_sem_acessar"] = pd.to_numeric(df["dias_sem_acessar"], errors="coerce")

    if "recorrencia" in df.columns:
        df["recorrencia"] = pd.to_numeric(df["recorrencia"], errors="coerce")

    return df, CleaningMetadata(dropped_columns=dropped, created_flags=sorted(set(flags)))


def diagnose_repetition_cause(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"cliente", "transacao", "data_ordem", "total_parcelas", "recorrente", "status"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para diagnóstico: {sorted(missing)}")

    g = df.groupby("cliente", dropna=False)
    rep = (
        g.agg(
            n_linhas=("cliente", "size"),
            n_transacoes=("transacao", pd.Series.nunique),
            min_data_ordem=("data_ordem", "min"),
            max_data_ordem=("data_ordem", "max"),
            n_status=("status", pd.Series.nunique),
            n_recorrente=("recorrente", pd.Series.nunique),
            n_total_parcelas=("total_parcelas", pd.Series.nunique),
        )
        .reset_index()
        .sort_values(["n_linhas", "n_transacoes"], ascending=False)
    )
    rep["tem_multiplas_transacoes"] = (rep["n_transacoes"] > 1).astype(int)
    return rep


def build_client_aggregates(df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> pd.DataFrame:
    needed = {"cliente", "transacao", "data_ordem"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para agregação: {sorted(missing)}")

    if ref_date is None:
        ref_date = pd.to_datetime(df["data_ordem"], errors="coerce").max()

    g = df.groupby("cliente", dropna=False)
    agg = g.agg(
        n_transacoes_cliente=("transacao", pd.Series.nunique),
        primeira_compra=("data_ordem", "min"),
        ultima_compra=("data_ordem", "max"),
    ).reset_index()

    window_days = (agg["ultima_compra"] - agg["primeira_compra"]).dt.days.fillna(0).astype(int) + 1
    agg["janela_dias"] = window_days
    agg["recencia_compra_dias"] = (ref_date - agg["ultima_compra"]).dt.days
    agg["freq_compra_mensal"] = agg["n_transacoes_cliente"] / agg["janela_dias"] * 30.0

    if "status" in df.columns:
        status = df[["cliente", "status"]].copy()
        status["status"] = status["status"].astype(str).str.upper()
        agg["qtd_status_refunded"] = status["status"].str.contains("REFUND", na=False).groupby(status["cliente"]).sum().values
        agg["qtd_chargeback"] = status["status"].str.contains("CHARGEBACK", na=False).groupby(status["cliente"]).sum().values

    return agg


def build_cliente_atual(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"cliente", "data_ordem"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para visão cliente-atual: {sorted(missing)}")

    df2 = df.sort_values(["cliente", "data_ordem"]).copy()
    latest = df2.groupby("cliente", dropna=False).tail(1)
    agg = build_client_aggregates(df2)
    out = latest.merge(agg, on="cliente", how="left", validate="one_to_one")
    return out

