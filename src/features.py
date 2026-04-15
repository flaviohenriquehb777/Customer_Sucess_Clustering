from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ID_COLS = ["cliente", "transacao"]

DATE_COLS = ["data_ordem", "data_aprovacao", "ultimo_acesso"]

DEFAULT_NUMERIC_FEATURES = [
    "log_acessos",
    "dias_sem_acessar",
    "recorrencia",
    "recorrente",
    "parcelado",
    "tipo_pagamento",
    "nunca_acessou",
    "n_transacoes_cliente",
    "recencia_compra_dias",
    "freq_compra_mensal",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "ativo",
    "renovacao",
    "faixa_inatividade",
    "metodo_pagamento",
    "parcelas_nao_recorrente_bin",
    "status",
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def infer_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    numeric = [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df.columns]
    return FeatureSpec(numeric=numeric, categorical=categorical)


def build_modeling_dataframe(df_cliente_atual: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    spec = infer_feature_spec(df_cliente_atual)
    cols = [c for c in (spec.numeric + spec.categorical) if c in df_cliente_atual.columns]
    X = df_cliente_atual[cols].copy()
    return X, spec

