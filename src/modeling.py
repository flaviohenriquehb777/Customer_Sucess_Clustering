from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureSpec


def build_preprocess_transformer(spec: FeatureSpec) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, spec.numeric),
            ("cat", cat_pipe, spec.categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    return preprocess


def build_kmeans_pipeline(spec: FeatureSpec, k: int, random_state: int = 42) -> Pipeline:
    preprocess = build_preprocess_transformer(spec)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("scaler", StandardScaler(with_mean=False)),
            ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init="auto")),
        ]
    )
    return pipe


@dataclass(frozen=True)
class KSelectionResult:
    k_values: list[int]
    inertia: list[float]
    silhouette: list[float]


def evaluate_k_range(
    X: pd.DataFrame,
    spec: FeatureSpec,
    k_min: int = 2,
    k_max: int = 12,
    silhouette_sample: int = 5000,
    random_state: int = 42,
) -> KSelectionResult:
    rng = np.random.default_rng(random_state)
    k_values = list(range(k_min, k_max + 1))
    inertia: list[float] = []
    sil: list[float] = []

    if len(X) > silhouette_sample:
        idx = rng.choice(len(X), size=silhouette_sample, replace=False)
        X_sil = X.iloc[idx].copy()
    else:
        X_sil = X.copy()

    for k in k_values:
        pipe = build_kmeans_pipeline(spec, k=k, random_state=random_state)
        pipe.fit(X)
        inertia.append(float(pipe.named_steps["kmeans"].inertia_))

        Xt = pipe.named_steps["preprocess"].transform(X_sil)
        Xt = pipe.named_steps["scaler"].transform(Xt)
        labels = pipe.named_steps["kmeans"].predict(pipe.named_steps["scaler"].transform(pipe.named_steps["preprocess"].transform(X_sil)))
        sil.append(float(silhouette_score(Xt, labels)))

    return KSelectionResult(k_values=k_values, inertia=inertia, silhouette=sil)


def choose_k(result: KSelectionResult) -> int:
    df = pd.DataFrame({"k": result.k_values, "inertia": result.inertia, "silhouette": result.silhouette})
    df["inertia_drop"] = df["inertia"].shift(1) - df["inertia"]
    df["inertia_drop_pct"] = df["inertia_drop"] / df["inertia"].shift(1)
    df2 = df.dropna().copy()
    if df2.empty:
        return int(df["k"].iloc[0])

    elbow_candidates = df2.sort_values("inertia_drop_pct", ascending=False).head(3)["k"].tolist()
    best = (
        df[df["k"].isin(elbow_candidates)]
        .sort_values(["silhouette", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )
    return int(best)


def fit_final_kmeans(
    X: pd.DataFrame,
    spec: FeatureSpec,
    k: int,
    random_state: int = 42,
) -> Pipeline:
    pipe = build_kmeans_pipeline(spec, k=k, random_state=random_state)
    pipe.fit(X)
    return pipe


def save_pipeline(pipe: Pipeline, path: str) -> None:
    dump(pipe, path)


def load_pipeline(path: str) -> Pipeline:
    p = Path(path)
    if p.exists():
        return load(p)

    if p.name == "kmeans_pipeline.joblib":
        alt = p.parent / "models" / p.name
        if alt.exists():
            return load(alt)

    raise FileNotFoundError(f"Pipeline não encontrado em: {p}")


def get_feature_names(pipe: Pipeline) -> np.ndarray:
    pre = pipe.named_steps["preprocess"]
    return pre.get_feature_names_out()


def transform_for_model(pipe: Pipeline, X: pd.DataFrame):
    Xt = pipe.named_steps["preprocess"].transform(X)
    Xt = pipe.named_steps["scaler"].transform(Xt)
    return Xt


def cluster_profiles(
    df_cliente_atual: pd.DataFrame,
    cluster_col: str = "cluster_id",
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if numeric_cols is None:
        numeric_cols = [c for c in df_cliente_atual.select_dtypes(include=[np.number]).columns if c != cluster_col]
    if categorical_cols is None:
        categorical_cols = [c for c in df_cliente_atual.columns if df_cliente_atual[c].dtype == "object" and c != cluster_col]

    base = df_cliente_atual.copy()
    total = len(base)
    sizes = base.groupby(cluster_col, dropna=False).size().rename("tamanho").to_frame()
    sizes["pct_total"] = sizes["tamanho"] / total

    num = base.groupby(cluster_col)[numeric_cols].agg(["mean", "median"])
    num.columns = [f"{c}_{stat}" for c, stat in num.columns]

    cat_tables = {}
    for col in categorical_cols:
        tab = (
            base.pivot_table(index=cluster_col, columns=col, values="cliente", aggfunc="count", fill_value=0)
            .div(sizes["tamanho"], axis=0)
        )
        tab.columns = [f"{col}__{c}" for c in tab.columns]
        cat_tables[col] = tab

    cat = pd.concat(cat_tables.values(), axis=1) if cat_tables else pd.DataFrame(index=sizes.index)
    summary = sizes.join(num, how="left").join(cat, how="left").reset_index().sort_values("tamanho", ascending=False)
    return {"sizes": sizes.reset_index(), "numeric": num.reset_index(), "categorical": cat.reset_index(), "summary": summary}


def compute_cluster_score(df_cliente_atual: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    df = df_cliente_atual.copy()
    needed = [c for c in ["log_acessos", "dias_sem_acessar", "recorrente", "renovacao", "nunca_acessou"] if c in df.columns]
    if not needed:
        raise ValueError("Não há colunas suficientes para calcular score.")

    tmp = df[needed].copy()
    if "renovacao" in tmp.columns:
        tmp["renovacao"] = tmp["renovacao"].map({"sim": 1, "nao": 0}).fillna(0)
    if "recorrente" in tmp.columns:
        tmp["recorrente"] = pd.to_numeric(tmp["recorrente"], errors="coerce").fillna(0)

    z = (tmp - tmp.mean()) / tmp.std(ddof=0).replace(0, 1)

    score = 0.0
    if "log_acessos" in z.columns:
        score += 0.35 * z["log_acessos"]
    if "recorrente" in z.columns:
        score += 0.25 * z["recorrente"]
    if "renovacao" in z.columns:
        score += 0.25 * z["renovacao"]
    if "dias_sem_acessar" in z.columns:
        score -= 0.35 * z["dias_sem_acessar"]
    if "nunca_acessou" in z.columns:
        score -= 0.30 * z["nunca_acessou"]

    df["score_valor_risco"] = score
    rank = (
        df.groupby(cluster_col)["score_valor_risco"]
        .mean()
        .sort_values(ascending=False)
        .rename("score_medio")
        .reset_index()
    )
    rank["ranking"] = np.arange(1, len(rank) + 1)
    return rank


def assign_business_names(df_cliente_atual: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    def _pct_sim(s: pd.Series) -> float:
        if s.dtype == "object":
            return float((s == "sim").mean())
        return float(np.nan)

    prof = df_cliente_atual.groupby(cluster_col).agg(
        n=("cliente", "size"),
        dias_sem_acessar_med=("dias_sem_acessar", "median"),
        log_acessos_med=("log_acessos", "median"),
        pct_nunca_acessou=("nunca_acessou", "mean"),
        pct_recorrente=("recorrente", "mean"),
        pct_parcelado=("parcelado", "mean"),
        pct_renovacao=("renovacao", _pct_sim) if "renovacao" in df_cliente_atual.columns else ("cliente", lambda s: np.nan),
        recencia_compra_med=("recencia_compra_dias", "median") if "recencia_compra_dias" in df_cliente_atual.columns else ("cliente", lambda s: np.nan),
    )
    names = {}
    score = 0.0
    score += prof["log_acessos_med"].rank(pct=True)
    score += prof["pct_recorrente"].rank(pct=True)
    score += prof["pct_renovacao"].fillna(0).rank(pct=True)
    score -= prof["dias_sem_acessar_med"].rank(pct=True)
    score -= prof["pct_nunca_acessou"].rank(pct=True)
    champions = int(score.idxmax())
    names[champions] = "Champions"

    zumbis = prof.sort_values(["pct_recorrente", "pct_nunca_acessou"], ascending=False).head(1).index.tolist()
    if zumbis:
        cid = int(zumbis[0])
        if cid not in names and prof.loc[cid, "pct_nunca_acessou"] >= 0.5 and prof.loc[cid, "pct_recorrente"] >= 0.5:
            names[cid] = "Zumbis Pagantes"

    for cid in prof.sort_values(["dias_sem_acessar_med", "log_acessos_med"], ascending=[False, True]).index:
        cid = int(cid)
        if cid not in names:
            names[cid] = "Risco de Churn"
            break

    for cid in prof.sort_values(["recencia_compra_med", "n"], ascending=[True, False]).index:
        cid = int(cid)
        if cid not in names:
            names[cid] = "Novos / Onboarding"
            break

    for cid in prof.index:
        cid = int(cid)
        if cid not in names:
            names[cid] = f"Cluster {cid}"

    out = pd.DataFrame({cluster_col: list(names.keys()), "nome_cluster": list(names.values())})
    return out


def predict_with_explanation(pipe: Pipeline, X_new: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    Xt = transform_for_model(pipe, X_new)
    kmeans = pipe.named_steps["kmeans"]
    labels = kmeans.predict(Xt)
    distances = kmeans.transform(Xt)
    min_dist = distances.min(axis=1)

    feature_names = get_feature_names(pipe)
    centers = kmeans.cluster_centers_

    explanations: list[list[dict[str, Any]]] = []
    for i in range(Xt.shape[0]):
        cid = labels[i]
        row = Xt[i]
        vec = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
        diff = vec - centers[cid]
        idx = np.argsort(np.abs(diff))[::-1][:top_n]
        items = []
        for j in idx:
            items.append({"feature": str(feature_names[j]), "delta_scaled": float(diff[j])})
        explanations.append(items)

    return pd.DataFrame(
        {
            "cluster_id": labels.astype(int),
            "distancia_centroide": min_dist.astype(float),
            "top_features_delta": explanations,
        }
    )

