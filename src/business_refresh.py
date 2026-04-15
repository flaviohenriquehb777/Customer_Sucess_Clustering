from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.features import build_modeling_dataframe
from src.modeling import (
    build_kmeans_pipeline,
    cluster_profiles,
    compute_cluster_score,
    transform_for_model,
)
from src.preprocessing import build_cliente_atual, clean_base, load_xlsx_base_clientes, winsorize_df


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_xlsx: Path
    raw_pdf: Path | None
    processed_dir: Path
    reports_dir: Path
    assets_dir: Path
    metrics_dir: Path
    exports_dir: Path
    models_dir: Path


def _find_inputs(root: Path) -> Paths:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    assets_dir = reports_dir / "assets"
    metrics_dir = reports_dir / "metrics"
    exports_dir = reports_dir / "exports"
    models_dir = reports_dir / "models"

    xlsx_candidates = sorted(raw_dir.glob("*.xlsx"))
    if not xlsx_candidates:
        xlsx_candidates = sorted(root.glob("*.xlsx"))
    if not xlsx_candidates:
        raise FileNotFoundError("Nenhum XLSX encontrado em data/raw ou na raiz.")
    raw_xlsx = xlsx_candidates[0]

    pdf_candidates = sorted(raw_dir.glob("*.pdf"))
    if not pdf_candidates:
        pdf_candidates = sorted(root.glob("*.pdf"))
    raw_pdf = pdf_candidates[0] if pdf_candidates else None

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=root,
        raw_xlsx=raw_xlsx,
        raw_pdf=raw_pdf,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        assets_dir=assets_dir,
        metrics_dir=metrics_dir,
        exports_dir=exports_dir,
        models_dir=models_dir,
    )


def _pct_missing_top(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    out = (
        df.isna()
        .mean()
        .rename("pct_missing")
        .to_frame()
        .assign(coluna=lambda d: d.index)
        .sort_values("pct_missing", ascending=False)
        .head(top_n)[["coluna", "pct_missing"]]
        .reset_index(drop=True)
    )
    return out


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].astype(float)
    return df2.to_markdown(index=False, floatfmt=floatfmt)


def _fmt_int_dot(x: int) -> str:
    return f"{int(x):,}".replace(",", ".")


def _fmt_pct_ptbr(x: float, digits: int = 2) -> str:
    return f"{x*100:.{digits}f}%".replace(".", ",")


def _fmt_float_ptbr(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}".replace(".", ",")


def _build_case_summary_tables(df_out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [4, 0, 1, 2, 3, 5]
    cluster_to_group = {
        4: "Champions",
        0: "Potenciais",
        1: "Avulsos Engajados",
        2: "Zumbis",
        3: "Churn Iminente",
        5: "Novos",
    }
    group_to_emoji = {
        "Champions": "🟢",
        "Potenciais": "🟡",
        "Avulsos Engajados": "🟠",
        "Zumbis": "🔴",
        "Churn Iminente": "⚪",
        "Novos": "🟣",
    }

    base = df_out.copy()
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
    out["grupo"] = out["cluster_id"].astype(int).map(cluster_to_group).fillna(out["cluster_id"].map(lambda x: f"Cluster {int(x)}"))
    out["grupo"] = out["grupo"].map(lambda g: f"{group_to_emoji.get(str(g), '•')} {g}")
    out = out.set_index("cluster_id").reindex(order).reset_index()

    t1 = pd.DataFrame(
        {
            "Grupo": out["grupo"],
            "Cluster": out["cluster_id"].astype(int).astype(str),
            "Clientes": out["clientes"].astype(int).map(_fmt_int_dot),
            "%": out["pct"].map(lambda v: _fmt_pct_ptbr(float(v), digits=2)),
            "log_acessos (med)": out["log_acessos_med"].map(lambda v: _fmt_float_ptbr(float(v), digits=3)),
            "dias_sem_acessar (med)": out["dias_sem_acessar_med"].round().astype(int).astype(str),
        }
    )

    t2 = pd.DataFrame(
        {
            "Grupo": out["grupo"],
            "% nunca_acessou": out["pct_nunca_acessou"].map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
            "% recorrente": out["pct_recorrente"].map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
            "% parcelado": out["pct_parcelado"].map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
            "% cartão": out.get("cartao", 0).map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
            "% pix": out.get("pix", 0).map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
            "% boleto": out.get("boleto", 0).map(lambda v: _fmt_pct_ptbr(float(v), digits=1)),
        }
    )

    return t1, t2


def _html_table(df: pd.DataFrame, dot_colors: dict[str, str] | None = None, dot_col: str = "Grupo") -> str:
    df2 = df.copy()
    if dot_colors is not None and dot_col in df2.columns:
        df2[dot_col] = df2[dot_col].map(
            lambda g: f'<span class="dot" style="background:{dot_colors.get(str(g), "#999")}"></span>{g}'
        )
    return df2.to_html(index=False, escape=False, border=0, classes="case")


def _audit_project(paths: Paths) -> dict[str, object]:
    required_dirs = ["data/raw", "data/processed", "notebooks", "src", "reports"]
    dir_status = []
    for d in required_dirs:
        dir_status.append({"item": d, "status": "OK" if (paths.root / d).exists() else "faltando"})
    dir_status = pd.DataFrame(dir_status)

    expected_notebooks = [
        "01_eda_e_qualidade.ipynb",
        "02_limpeza_e_features.ipynb",
        "03_modelo_kmeans_escolha_k.ipynb",
        "04_perfis_dos_clusters_e_acoes.ipynb",
        "05_storytelling_executivo.ipynb",
        "06_classificacao_novos_clientes_pipeline.ipynb",
        "07_shap_explicabilidade.ipynb",
    ]
    nb_dir = paths.root / "notebooks"
    existing = sorted([p.name for p in nb_dir.glob("*.ipynb")]) if nb_dir.exists() else []
    nb_rows = []
    for name in expected_notebooks:
        status = "OK" if name in existing else "faltando"
        nb_rows.append({"notebook": name, "status": status})
    extras = [n for n in existing if n not in expected_notebooks]
    for n in extras:
        nb_rows.append({"notebook": n, "status": "nome fora do padrão"})
    notebooks_status = pd.DataFrame(nb_rows).sort_values(["status", "notebook"])

    req_path = paths.root / "requirements.txt"
    has_requirements = req_path.exists()

    expected_reports = [
        "exports/dataset_clusterizado.xlsx",
        "exports/resumo_clusters.xlsx",
        "models/kmeans_pipeline.joblib",
    ]
    report_rows = []
    for f in expected_reports:
        report_rows.append({"artefato": f, "status": "OK" if (paths.reports_dir / f).exists() else "faltando"})
    report_rows.append(
        {"artefato": "assets/k_selection.png", "status": "OK" if (paths.assets_dir / "k_selection.png").exists() else "será gerado"}
    )
    report_rows.append(
        {"artefato": "assets/k_stability.png", "status": "OK" if (paths.assets_dir / "k_stability.png").exists() else "será gerado"}
    )
    report_rows.append(
        {"artefato": "assets/pca_clusters.png", "status": "OK" if (paths.assets_dir / "pca_clusters.png").exists() else "será gerado"}
    )
    report_rows.append(
        {"artefato": "assets/cluster_heatmap.png", "status": "OK" if (paths.assets_dir / "cluster_heatmap.png").exists() else "será gerado"}
    )
    report_rows.append(
        {"artefato": "assets/radar_clusters.png", "status": "OK" if (paths.assets_dir / "radar_clusters.png").exists() else "será gerado"}
    )
    deliverables = pd.DataFrame(report_rows)

    return {
        "dir_status": dir_status,
        "notebooks_status": notebooks_status,
        "has_requirements": has_requirements,
        "deliverables": deliverables,
    }


def _redundancy_recorrente_parcelas(df_cliente: pd.DataFrame) -> dict[str, object]:
    if "recorrente" not in df_cliente.columns or "total_parcelas" not in df_cliente.columns:
        return {"available": False}
    tmp = df_cliente[["recorrente", "total_parcelas", "parcelado"]].copy()
    tmp["recorrente"] = pd.to_numeric(tmp["recorrente"], errors="coerce").fillna(0).astype(int)
    tmp["total_parcelas"] = pd.to_numeric(tmp["total_parcelas"], errors="coerce")
    tmp["parcelado"] = pd.to_numeric(tmp.get("parcelado"), errors="coerce").fillna(0).astype(int)

    tab = (
        tmp.assign(total_parcelas_cat=lambda d: d["total_parcelas"].fillna(-1).astype(int))
        .groupby(["recorrente", "total_parcelas_cat"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
        .head(15)
    )
    corr = tmp[["recorrente", "total_parcelas", "parcelado"]].corr(numeric_only=True)

    return {"available": True, "top_combos": tab, "corr": corr.reset_index().rename(columns={"index": "feature"})}


def _unit_of_analysis_stats(df_trans: pd.DataFrame) -> dict[str, object]:
    g = df_trans.groupby("cliente", dropna=False)
    n_lines = g.size()
    pct_cliente_repetido = float((n_lines > 1).mean())

    n_trans = g["transacao"].nunique()
    dist = n_trans.describe(percentiles=[0.5, 0.75, 0.95]).to_frame("valor").reset_index().rename(columns={"index": "estat"})

    return {"pct_cliente_repetido": pct_cliente_repetido, "n_transacoes_dist": dist}


def _fit_metrics_for_k(
    X: pd.DataFrame,
    spec,
    k: int,
    sample_idx: np.ndarray,
    random_state: int = 42,
) -> dict[str, object]:
    pipe = build_kmeans_pipeline(spec, k=k, random_state=random_state)
    pipe.fit(X)

    inertia = float(pipe.named_steps["kmeans"].inertia_)

    Xs = X.iloc[sample_idx].copy()
    Xt = transform_for_model(pipe, Xs)
    labels_s = pipe.predict(Xs)

    if hasattr(Xt, "toarray"):
        Xt_dense = Xt.toarray()
    else:
        Xt_dense = np.asarray(Xt)

    silhouette = float(silhouette_score(Xt_dense, labels_s))
    ch = float(calinski_harabasz_score(Xt_dense, labels_s))
    db = float(davies_bouldin_score(Xt_dense, labels_s))

    labels_full = pipe.predict(X)
    counts = pd.Series(labels_full).value_counts().sort_index()
    min_count = int(counts.min())
    min_pct = float(min_count / len(X))

    df_sizes = counts.rename("n").to_frame().reset_index().rename(columns={"index": "cluster_id"})

    return {
        "pipe": pipe,
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "min_cluster_n": min_count,
        "min_cluster_pct": min_pct,
        "sizes": df_sizes,
    }


def _stability_for_k(
    X: pd.DataFrame,
    spec,
    k: int,
    eval_idx: np.ndarray,
    n_runs: int = 5,
    train_frac: float = 0.7,
    base_seed: int = 42,
) -> pd.DataFrame:
    labels_runs = []
    n = len(X)
    for i in range(n_runs):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(n, size=int(n * train_frac), replace=False)
        pipe = build_kmeans_pipeline(spec, k=k, random_state=42)
        pipe.fit(X.iloc[train_idx])
        labels_runs.append(pipe.predict(X.iloc[eval_idx]))

    rows = []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(labels_runs)), 2):
        rows.append(
            {
                "k": k,
                "pair": f"{i}-{j}",
                "ari": float(adjusted_rand_score(a, b)),
                "nmi": float(normalized_mutual_info_score(a, b)),
            }
        )
    return pd.DataFrame(rows)


def _choose_k(k_table: pd.DataFrame, stability: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    stab_med = stability.groupby("k")[["ari", "nmi"]].median().reset_index().rename(columns={"ari": "ari_med", "nmi": "nmi_med"})
    full = k_table.merge(stab_med, on="k", how="left")
    full["elegivel"] = full["min_cluster_pct"] >= 0.02

    cand = full[full["elegivel"]].copy()
    if cand.empty:
        cand = full.copy()

    score = pd.Series(0.0, index=cand.index)
    score += cand["silhouette"].rank(pct=True)
    score += cand["ari_med"].fillna(cand["ari_med"].median()).rank(pct=True)
    score += cand["calinski_harabasz"].rank(pct=True)
    score -= cand["davies_bouldin"].rank(pct=True)
    score += cand["min_cluster_pct"].rank(pct=True)
    score -= cand["k"].rank(pct=True) * 0.15
    cand["score_selecao"] = score

    score_map = cand.set_index("k")["score_selecao"].to_dict()
    full["score_selecao"] = full["k"].map(score_map)

    chosen = int(cand.sort_values(["score_selecao", "silhouette"], ascending=False).iloc[0]["k"])
    return chosen, full.sort_values("k").reset_index(drop=True)


def _cluster_cards(df: pd.DataFrame, cluster_col: str = "cluster_id") -> dict[int, pd.DataFrame]:
    g = df.groupby(cluster_col)
    out = {}
    for cid, sub in g:
        n = len(sub)
        pct = n / len(df)
        row = {
            "cluster_id": int(cid),
            "%_clientes": pct,
            "n": n,
            "log_acessos_med": float(sub["log_acessos"].median()) if "log_acessos" in sub.columns else np.nan,
            "dias_sem_acessar_med": float(sub["dias_sem_acessar"].median()) if "dias_sem_acessar" in sub.columns else np.nan,
            "%_nunca_acessou": float(sub["nunca_acessou"].mean()) if "nunca_acessou" in sub.columns else np.nan,
            "%_recorrente": float((sub["recorrente"].fillna(0) == 1).mean()) if "recorrente" in sub.columns else np.nan,
            "%_renovacao": float((sub["renovacao"] == "sim").mean()) if "renovacao" in sub.columns else np.nan,
            "%_parcelado": float(sub["parcelado"].mean()) if "parcelado" in sub.columns else np.nan,
        }
        if "metodo_pagamento" in sub.columns:
            dist = sub["metodo_pagamento"].value_counts(normalize=True)
            row["%_pag_cartao"] = float(dist.get("cartao", 0.0))
            row["%_pag_pix"] = float(dist.get("pix", 0.0))
            row["%_pag_boleto"] = float(dist.get("boleto", 0.0))
        out[int(cid)] = pd.DataFrame([row])
    return out


def _save_radar(df_profiles_norm: pd.DataFrame, out_path: Path, cluster_col: str = "cluster_id") -> None:
    cats = [c for c in df_profiles_norm.columns if c != cluster_col]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in df_profiles_norm.iterrows():
        values = row[cats].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=str(int(row[cluster_col])))
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Radar (perfis normalizados)", pad=18)
    ax.legend(title="cluster", bbox_to_anchor=(1.15, 1.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    sns.set_theme(style="whitegrid", palette="Set2")

    root = Path(__file__).resolve().parents[1]
    paths = _find_inputs(root)

    audit = _audit_project(paths)

    df_raw = load_xlsx_base_clientes(str(paths.raw_xlsx))
    df_trans, _ = clean_base(df_raw)
    df_trans = winsorize_df(df_trans, cols=[c for c in ["dias_sem_acessar"] if c in df_trans.columns], lower_q=0.01, upper_q=0.99)
    df_cliente = build_cliente_atual(df_trans)

    df_trans.to_parquet(paths.processed_dir / "base_limpa_transacao.parquet", index=False)
    df_cliente.to_parquet(paths.processed_dir / "base_cliente_atual.parquet", index=False)

    unit_stats = _unit_of_analysis_stats(df_trans)
    redundancy = _redundancy_recorrente_parcelas(df_cliente)
    missing_top = _pct_missing_top(df_raw, top_n=10)

    X, spec = build_modeling_dataframe(df_cliente)
    X.to_parquet(paths.processed_dir / "modeling_X.parquet", index=False)
    feature_spec_rows = (
        pd.DataFrame({"tipo": ["numeric"] * len(spec.numeric) + ["categorical"] * len(spec.categorical), "feature": spec.numeric + spec.categorical})
        .sort_values(["tipo", "feature"])
        .reset_index(drop=True)
    )
    feature_spec_rows.to_json(paths.processed_dir / "feature_spec.json", orient="records", force_ascii=False)

    candidates = [3, 4, 5, 6, 7, 8, 9, 10, 12]
    rng = np.random.default_rng(42)
    sample_n = min(6000, len(X))
    sample_idx = rng.choice(len(X), size=sample_n, replace=False)

    metrics_rows = []
    stability_rows = []

    for k in candidates:
        res = _fit_metrics_for_k(X, spec, k=k, sample_idx=sample_idx, random_state=42)
        sizes_str = ",".join([f"{int(r.cluster_id)}:{int(r.n)}" for r in res["sizes"].itertuples(index=False)])
        metrics_rows.append(
            {
                "k": k,
                "inertia": res["inertia"],
                "silhouette": res["silhouette"],
                "calinski_harabasz": res["calinski_harabasz"],
                "davies_bouldin": res["davies_bouldin"],
                "min_cluster_n": res["min_cluster_n"],
                "min_cluster_pct": res["min_cluster_pct"],
                "cluster_sizes": sizes_str,
            }
        )

        eval_n = min(5000, len(X))
        eval_idx = np.random.default_rng(42).choice(len(X), size=eval_n, replace=False)
        stability_rows.append(_stability_for_k(X, spec, k=k, eval_idx=eval_idx, n_runs=5, train_frac=0.7, base_seed=42))

    k_table = pd.DataFrame(metrics_rows).sort_values("k").reset_index(drop=True)
    stability = pd.concat(stability_rows, ignore_index=True)

    chosen_k, k_table_out = _choose_k(k_table, stability)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.lineplot(data=k_table, x="k", y="inertia", marker="o", ax=ax[0, 0])
    ax[0, 0].set_title("Elbow (Inertia)")
    sns.lineplot(data=k_table, x="k", y="silhouette", marker="o", ax=ax[0, 1])
    ax[0, 1].set_title("Silhouette (amostra)")
    sns.lineplot(data=k_table, x="k", y="calinski_harabasz", marker="o", ax=ax[1, 0])
    ax[1, 0].set_title("Calinski-Harabasz (amostra)")
    sns.lineplot(data=k_table, x="k", y="davies_bouldin", marker="o", ax=ax[1, 1])
    ax[1, 1].set_title("Davies-Bouldin (amostra; menor é melhor)")
    for a in ax.ravel():
        a.axvline(chosen_k, color="black", linestyle="--", linewidth=1)
    fig.suptitle(f"Seleção de K (candidatos) — K escolhido: {chosen_k}", y=1.02)
    fig.tight_layout()
    fig.savefig(paths.assets_dir / "k_selection.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(data=stability, x="k", y="ari", ax=ax1)
    ax1.set_title("Estabilidade por K (ARI) — 5 subamostras (70%)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("ARI")
    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(data=stability, x="k", y="nmi", ax=ax2)
    ax2.set_title("Estabilidade por K (NMI) — 5 subamostras (70%)")
    ax2.set_xlabel("K")
    ax2.set_ylabel("NMI")
    fig.tight_layout()
    fig.savefig(paths.assets_dir / "k_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    final_pipe = build_kmeans_pipeline(spec, k=chosen_k, random_state=42)
    final_pipe.fit(X)
    dump(final_pipe, paths.models_dir / "kmeans_pipeline.joblib")

    df_out = df_cliente.copy()
    df_out["cluster_id"] = final_pipe.predict(X).astype(int)

    cluster_to_name = {
        4: "Champions",
        0: "Potenciais",
        1: "Avulsos Engajados",
        2: "Zumbis",
        3: "Churn Iminente",
        5: "Novos",
    }
    all_cluster_ids = sorted(df_out["cluster_id"].astype(int).unique().tolist())
    names = pd.DataFrame(
        {
            "cluster_id": all_cluster_ids,
            "nome_cluster": [cluster_to_name.get(cid, f"Cluster {cid}") for cid in all_cluster_ids],
        }
    )
    df_out = df_out.merge(names, on="cluster_id", how="left")

    rank = compute_cluster_score(df_out, cluster_col="cluster_id").merge(names, on="cluster_id", how="left")
    profiles = cluster_profiles(
        df_out,
        cluster_col="cluster_id",
        numeric_cols=[c for c in ["log_acessos", "dias_sem_acessar", "recorrente", "parcelado", "nunca_acessou", "n_transacoes_cliente", "recencia_compra_dias"] if c in df_out.columns],
        categorical_cols=[c for c in ["metodo_pagamento", "ativo", "renovacao", "faixa_inatividade"] if c in df_out.columns],
    )

    key = [c for c in ["log_acessos", "dias_sem_acessar", "recorrente", "parcelado", "nunca_acessou", "n_transacoes_cliente", "recencia_compra_dias"] if c in df_out.columns]
    prof_mean = df_out.groupby(["cluster_id", "nome_cluster"])[key].mean().reset_index()
    mat = prof_mean.set_index(["cluster_id", "nome_cluster"])[key]
    mat_norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)

    plt.figure(figsize=(12, 6))
    sns.heatmap(mat_norm, cmap="YlGnBu", linewidths=0.2)
    plt.title("Perfis normalizados por cluster (média)")
    plt.tight_layout()
    plt.savefig(paths.assets_dir / "cluster_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()

    radar_df = mat_norm.reset_index().drop(columns=["nome_cluster"])
    _save_radar(radar_df, paths.assets_dir / "radar_clusters.png", cluster_col="cluster_id")

    pca_n = min(10000, len(X))
    pca_idx = np.random.default_rng(42).choice(len(X), size=pca_n, replace=False)
    Xt_pca = transform_for_model(final_pipe, X.iloc[pca_idx])
    if hasattr(Xt_pca, "toarray"):
        Xt_pca_dense = Xt_pca.toarray()
    else:
        Xt_pca_dense = np.asarray(Xt_pca)
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(Xt_pca_dense)
    dfp = pd.DataFrame({"pc1": comp[:, 0], "pc2": comp[:, 1], "cluster_id": df_out.iloc[pca_idx]["cluster_id"].values})
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=dfp, x="pc1", y="pc2", hue="cluster_id", s=16, alpha=0.7)
    ax.set_title("PCA 2D (amostra) — clusters")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(paths.assets_dir / "pca_clusters.png", dpi=160, bbox_inches="tight")
    plt.close()

    cols_export = [c for c in ["cliente", "cluster_id", "nome_cluster"] + key + ["metodo_pagamento", "ativo", "renovacao", "faixa_inatividade", "tipo_pagamento"] if c in df_out.columns]
    dataset_clusterizado = df_out[cols_export].copy()

    mask_q4 = (df_out.get("recorrente", 0).fillna(0) == 1) & (df_out.get("nunca_acessou", 0).fillna(0) == 1)
    q4 = df_out.loc[mask_q4].copy()
    q4_by_cluster = (
        q4.groupby(["cluster_id", "nome_cluster"])
        .size()
        .rename("n")
        .to_frame()
        .assign(pct_do_q4=lambda d: d["n"] / d["n"].sum(), pct_do_cluster=lambda d: d["n"] / df_out.groupby("cluster_id").size())
        .reset_index()
        .sort_values("n", ascending=False)
    )

    with pd.ExcelWriter(paths.exports_dir / "dataset_clusterizado.xlsx", engine="xlsxwriter") as writer:
        dataset_clusterizado.to_excel(writer, sheet_name="cliente_atual_cluster", index=False)
        q4[cols_export].to_excel(writer, sheet_name="q4_recorrente_sem_acesso", index=False)

    acoes = names.copy()
    acoes["objetivo"] = acoes["nome_cluster"].map(
        {
            "Champions": "Expansão e defesa",
            "Potenciais": "Elevar valor percebido e reduzir inatividade",
            "Avulsos Engajados": "Converter para recorrência",
            "Zumbis": "Ativação imediata (primeiro uso)",
            "Churn Iminente": "Retenção e reativação",
            "Novos": "Onboarding e prova rápida de valor",
        }
    ).fillna("Engajamento e otimização")
    acoes["mensagem"] = acoes["nome_cluster"].map(
        {
            "Champions": "Convite para trilhas avançadas, indicação e upgrades",
            "Potenciais": "Conteúdo e jornada de progresso para aumentar uso e valor percebido",
            "Avulsos Engajados": "Oferta de assinatura/upgrade com proposta clara de benefício contínuo",
            "Zumbis": "Onboarding intensivo (7 dias) + primeiros resultados",
            "Churn Iminente": "Win-back guiado + remover fricções + plano de sucesso",
            "Novos": "Boas-vindas com microvitórias e prova rápida de valor",
        }
    ).fillna("Conteúdo personalizado por perfil")
    acoes["canal"] = "Email + WhatsApp + In-app"
    acoes["trigger"] = acoes["nome_cluster"].map(
        {
            "Champions": "Alta atividade por 14 dias",
            "Potenciais": "7–14 dias sem avanço ou queda de acessos",
            "Avulsos Engajados": "Conclusão de milestones (ex.: aula concluída) + oferta de próximo passo",
            "Zumbis": "Pagamento confirmado e nunca acessou",
            "Churn Iminente": ">=30 dias sem acessar ou queda de acessos",
            "Novos": "D+3 e D+7 sem acesso após cadastro/pagamento",
        }
    ).fillna("Segmentação semanal")
    acoes["kpi_esperado"] = acoes["nome_cluster"].map(
        {
            "Champions": "LTV ↑, upsell ↑, NPS ↑",
            "Potenciais": "Engajamento ↑, dias sem acessar ↓, renovação ↑",
            "Avulsos Engajados": "Conversão p/ recorrência ↑, LTV ↑",
            "Zumbis": "Ativação (primeiro acesso) ↑, churn ↓",
            "Churn Iminente": "Churn ↓, reativação ↑, renovação ↑",
            "Novos": "Ativação em 14 dias ↑, conclusão de onboarding ↑",
        }
    ).fillna("Engajamento ↑")

    with pd.ExcelWriter(paths.exports_dir / "resumo_clusters.xlsx", engine="xlsxwriter") as writer:
        profiles["summary"].to_excel(writer, sheet_name="perfil_resumo", index=False)
        rank.to_excel(writer, sheet_name="ranking", index=False)
        acoes.to_excel(writer, sheet_name="playbooks", index=False)
        q4_by_cluster.to_excel(writer, sheet_name="q4_concentracao", index=False)

    df_out.to_parquet(paths.processed_dir / "base_cliente_clusterizada.parquet", index=False)

    from src.render_case_tables_assets import render_assets as _render_case_assets
    from src.reporting.render_ranking_cards_png import render as _render_ranking_cards

    _render_case_assets()
    _render_ranking_cards()

    cards = _cluster_cards(df_out)

    stab_med = stability.groupby("k")[["ari", "nmi"]].median().reset_index().rename(columns={"ari": "ari_med", "nmi": "nmi_med"})
    k_table_out = k_table_out.copy()

    report_path = paths.reports_dir / "final_report.md"

    q4_size = int(mask_q4.sum())
    q4_pct = float(mask_q4.mean())

    k3_row = k_table_out[k_table_out["k"] == 3].copy()
    chosen_row = k_table_out[k_table_out["k"] == chosen_k].copy()

    topo = rank.sort_values("ranking").iloc[0]
    fundo = rank.sort_values("ranking").iloc[-1]

    novo_cliente_exemplo = {
        "log_acessos": 0.0,
        "dias_sem_acessar": 999.0,
        "recorrencia": 0.0,
        "recorrente": 1.0,
        "parcelado": 1.0,
        "tipo_pagamento": 1.0,
        "nunca_acessou": 1.0,
        "n_transacoes_cliente": 1.0,
        "recencia_compra_dias": 5.0,
        "freq_compra_mensal": 1.0,
        "ativo": "sim",
        "renovacao": "nao",
        "faixa_inatividade": "181_plus",
        "metodo_pagamento": "cartao",
        "parcelas_nao_recorrente_bin": None,
        "status": "COMPLETE",
    }
    novo_df = pd.DataFrame([novo_cliente_exemplo])
    pred_cluster = int(final_pipe.predict(novo_df)[0])
    pred_nome = str(names.set_index("cluster_id").loc[pred_cluster, "nome_cluster"])

    toc = "\n".join(
        [
            "- [1. Resumo executivo](#1-resumo-executivo)",
            "- [2. Auditoria do projeto](#2-auditoria-do-projeto)",
            "- [3. Dados e limpeza](#3-dados-e-limpeza)",
            "- [4. Features escolhidas](#4-features-escolhidas)",
            "- [5. Escolha de K orientada ao negócio](#5-escolha-de-k-orientada-ao-negócio)",
            "- [6. Perfis dos clusters (Cluster Cards)](#6-perfis-dos-clusters-cluster-cards)",
            "- [7. Respostas às 6 perguntas do case](#7-respostas-às-6-perguntas-do-case)",
            "- [8. Plano de ação por cluster](#8-plano-de-ação-por-cluster)",
            "- [9. Classificação automática de novos clientes](#9-classificação-automática-de-novos-clientes)",
            "- [10. Limitações e próximos passos](#10-limitações-e-próximos-passos)",
        ]
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Relatório Final — Segmentação de Clientes (K-Means)\n\n")
        f.write("## Sumário\n")
        f.write(toc + "\n\n")

        f.write("## 1. Resumo executivo\n")
        f.write(f"- K final: **{chosen_k}** (seleção por métricas + estabilidade + acionabilidade)\n")
        f.write("- Critérios de decisão: eliminamos clusters <2% (salvo exceção justificada), balanceamos elbow/silhouette e exigimos estabilidade (ARI/NMI) aceitável\n")
        f.write(f"- Grupo recorrente que não acessa: **{q4_size} ({q4_pct*100:.2f}%)**; concentração por cluster na seção 7\n")
        f.write(f"- Melhor cluster (valor/risco): **{topo['nome_cluster']}** | pior: **{fundo['nome_cluster']}** (score transparente)\n\n")
        f.write("Ações prioritárias (alto impacto):\n")
        f.write("- Ativação imediata de pagantes sem uso (nudges + onboarding 7 dias + contato CS)\n")
        f.write("- Playbook de retenção para risco de churn (gatilhos por inatividade + ofertas e diagnóstico de fricção)\n")
        f.write("- Expansão em Champions (upsell/cross-sell e indicação)\n\n")

        f.write("Legenda (nomenclatura “intuitiva” dos 6 grupos):\n")
        cluster_pct = (df_out["cluster_id"].value_counts(normalize=True) * 100).to_dict()

        def _pct_str(cid: int) -> str:
            return f"{cluster_pct.get(cid, 0.0):.2f}%".replace(".", ",")

        legend_items = [
            (
                4,
                "🟢",
                "maior engajamento e maior score de valor/risco; combinação mais saudável de uso + retenção/valor.",
            ),
            (0, "🟡", "recorrentes e engajados, mas com espaço para elevar valor percebido e reduzir inatividade."),
            (1, "🟠", "engajamento médio/alto, porém sem recorrência (oportunidade de conversão para assinatura)."),
            (2, "🔴", "pagam e não usam (nunca acessam); maior risco de cancelamento e frustração do cliente."),
            (3, "⚪", "forte sinal de abandono por alta inatividade; exige retenção imediata."),
            (
                5,
                "🟣",
                "perfil mais “recente/instável” com sinais mistos (ex.: alta incidência de Pix/Boleto e menor renovação); precisa de onboarding e prova rápida de valor.",
            ),
        ]
        for cid, emoji, desc in legend_items:
            nm = str(names.set_index("cluster_id").loc[cid, "nome_cluster"])
            f.write(f"- {emoji} {nm} (Cluster {cid} | {_pct_str(cid)}) → {desc}\n")
        f.write("\n")

        f.write("Resumo numérico (visão cliente-atual), em duas partes (melhor para relatório em retrato):\n\n")
        t1, t2 = _build_case_summary_tables(df_out)
        f.write("### Resumo numérico — Parte 1\n")
        f.write(_markdown_table(t1) + "\n\n")
        f.write("- Versão em imagem: `assets/case_resumo_parte1.png`\n\n")
        f.write("### Resumo numérico — Parte 2\n")
        f.write(_markdown_table(t2) + "\n\n")
        f.write("- Versão em imagem: `assets/case_resumo_parte2.png`\n\n")

        f.write("![Ranking de valor/risco](assets/cluster_ranking_visual.png)\n\n")

        f.write("## 2. Auditoria do projeto\n")
        f.write("### Checklist de entregáveis\n")
        f.write(_markdown_table(audit["deliverables"]) + "\n\n")
        f.write("### Estrutura de pastas\n")
        f.write(_markdown_table(audit["dir_status"]) + "\n\n")
        f.write("### Notebooks\n")
        f.write(_markdown_table(audit["notebooks_status"]) + "\n\n")
        f.write(f"- requirements.txt: **{'OK' if audit['has_requirements'] else 'faltando'}**\n\n")

        f.write("## 3. Dados e limpeza\n")
        f.write(f"- XLSX detectado: `{paths.raw_xlsx.relative_to(paths.root)}`\n")
        f.write(f"- PDF detectado: `{paths.raw_pdf.relative_to(paths.root) if paths.raw_pdf else 'não encontrado'}`\n\n")
        f.write("### Nulos (top 10 colunas)\n")
        f.write(_markdown_table(missing_top, floatfmt=".2%") + "\n\n")
        f.write("### Unidade de análise\n")
        f.write(f"- % de clientes com >1 linha (granularidade transacional): **{unit_stats['pct_cliente_repetido']*100:.2f}%**\n\n")
        f.write("Distribuição de `n_transacoes_cliente = nunique(TRANSACAO)` por cliente:\n")
        f.write(_markdown_table(unit_stats["n_transacoes_dist"]) + "\n\n")
        f.write("- Conclusão: a clusterização deve ser no nível **cliente** (visão cliente-atual por data_ordem + agregados históricos), pois linhas representam transações.\n\n")
        f.write("### Outliers\n")
        f.write("- `n_acessos`: transformação `log1p` (reduz assimetria e influência de extremos)\n")
        f.write("- `dias_sem_acessar`: winsorização 1%–99% (reduz sensibilidade do K-Means a outliers)\n\n")

        f.write("## 4. Features escolhidas\n")
        feat_tbl = feature_spec_rows.copy()
        feat_tbl["por_que_entra"] = feat_tbl["feature"].map(
            {
                "log_acessos": "Engajamento (volume de uso) com robustez a outliers",
                "dias_sem_acessar": "Inatividade (proxy de risco de churn)",
                "recorrencia": "Recência/uso recorrente (sinal de hábito)",
                "recorrente": "Plano recorrente (sinal de LTV e retenção)",
                "parcelado": "Estrutura de pagamento (barreira financeira / compromisso)",
                "tipo_pagamento": "Meio de pagamento (padrões comportamentais e risco)",
                "nunca_acessou": "Ativação (0/1); identifica pagantes sem uso",
                "n_transacoes_cliente": "Histórico de compras (freq./engajamento financeiro)",
                "recencia_compra_dias": "Recência de compra (ciclo de vida)",
                "freq_compra_mensal": "Intensidade de compras (proxy de valor)",
                "ativo": "Status operacional do cliente (quando disponível)",
                "renovacao": "Retenção (renovou ou não)",
                "faixa_inatividade": "Segmentação interpretável da inatividade",
                "metodo_pagamento": "Distribuição cartão/pix/boleto",
                "parcelas_nao_recorrente_bin": "Detalhe de parcelamento apenas para não-recorrentes (evita redundância)",
                "status": "Status transacional (sinal de sucesso/ruído operacional)",
            }
        )
        feat_tbl["transformacao"] = feat_tbl["feature"].map(
            {
                "log_acessos": "log1p",
                "dias_sem_acessar": "winsor 1–99%",
                "faixa_inatividade": "binning",
                "metodo_pagamento": "mapeamento (cartao/pix/boleto/outro)",
                "parcelas_nao_recorrente_bin": "binning (1, 2–3, 4–6, 7+)",
            }
        ).fillna("imputação + one-hot/scale no pipeline")
        f.write(_markdown_table(feat_tbl[["feature", "tipo", "por_que_entra", "transformacao"]]) + "\n\n")

        f.write("### Redundância: `recorrente` × `total_parcelas`\n")
        f.write("Decisão: **Opção 1 (recomendada)** — manter `recorrente` e evitar duplicar peso de `total_parcelas` no K-Means.\n")
        f.write("Implementação: mantemos `parcelado` e adicionamos `parcelas_nao_recorrente_bin` apenas para não-recorrentes.\n\n")
        if redundancy.get("available"):
            f.write("Top combinações (recorrente × total_parcelas) na visão cliente-atual:\n")
            f.write(_markdown_table(redundancy["top_combos"]) + "\n\n")
            f.write("Correlação (indicativa de redundância):\n")
            f.write(_markdown_table(redundancy["corr"], floatfmt=".2f") + "\n\n")

        f.write("## 5. Escolha de K orientada ao negócio\n")
        f.write("### Métricas quantitativas (candidatos)\n")
        f.write(_markdown_table(k_table_out[["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin", "min_cluster_n", "min_cluster_pct", "cluster_sizes", "ari_med", "nmi_med", "elegivel", "score_selecao"]]) + "\n\n")
        f.write("**Elbow**: redução de variância intracluster (inertia); buscamos o “joelho” onde ganhos adicionais ficam marginais.\n\n")
        f.write("**Silhouette**: quão bem separadas e coesas são as fronteiras entre clusters (maior é melhor).\n\n")
        f.write("**Estabilidade (ARI/NMI)**: consistência dos clusters sob subamostragem (5 treinos com 70% da base). Valores maiores indicam segmentação mais “confiável”.\n\n")
        f.write("![Seleção de K](assets/k_selection.png)\n\n")
        f.write("![Estabilidade por K](assets/k_stability.png)\n\n")

        f.write("### Regra explícita de decisão\n")
        f.write("- Eliminamos K onde algum cluster <2% (salvo exceção por achado crítico)\n")
        f.write("- Preferimos K na região do joelho do elbow **desde que** silhouette e estabilidade não degradem significativamente\n")
        f.write("- Desempate por clareza de perfis e acionabilidade (cluster cards)\n\n")

        f.write("### Por que não K=3? / Por que sim K=3?\n")
        if not k3_row.empty:
            r3 = k3_row.iloc[0].to_dict()
            f.write(f"- K=3: silhouette={r3['silhouette']:.3f}, ARI_med={r3.get('ari_med', np.nan):.3f}, min_cluster={r3['min_cluster_pct']*100:.2f}%\n")
        if not chosen_row.empty:
            rc = chosen_row.iloc[0].to_dict()
            f.write(f"- K={chosen_k}: silhouette={rc['silhouette']:.3f}, ARI_med={rc.get('ari_med', np.nan):.3f}, min_cluster={rc['min_cluster_pct']*100:.2f}%\n")
        f.write(f"- Conclusão: escolhemos **K={chosen_k}** por melhor equilíbrio entre separação, estabilidade e clusters acionáveis.\n\n")

        f.write("## 6. Perfis dos clusters (Cluster Cards)\n")
        f.write("![PCA 2D](assets/pca_clusters.png)\n\n")
        f.write("![Heatmap de perfis](assets/cluster_heatmap.png)\n\n")
        f.write("![Radar de perfis](assets/radar_clusters.png)\n\n")

        for cid in sorted(cards.keys()):
            nm = str(names.set_index("cluster_id").loc[cid, "nome_cluster"])
            f.write(f"### Cluster {cid} — {nm}\n")
            f.write(_markdown_table(cards[cid], floatfmt=".2f") + "\n\n")

        f.write("## 7. Respostas às 6 perguntas do case\n")
        f.write("1) Quantos grupos existem e qual critério para K?\n")
        f.write(f"- **{chosen_k}** grupos; decisão por métricas (elbow/silhouette/CH/DB), estabilidade (ARI/NMI) e acionabilidade.\n\n")

        f.write("2) Qual o perfil de cada grupo?\n")
        f.write("- Perfis detalhados estão nas Cluster Cards (seção 6) com engajamento, inatividade, recorrência, renovação e pagamento.\n\n")

        f.write("3) Existe algum grupo que se destaca positivamente e negativamente?\n")
        f.write(f"- Positivo: **{topo['nome_cluster']}** (maior score médio de valor/risco).\n")
        f.write(f"- Negativo: **{fundo['nome_cluster']}** (menor score médio; tipicamente baixa ativação/uso).\n\n")

        f.write("4) Existem recorrentes que não acessam? Qual o tamanho e onde se concentra?\n")
        f.write("- Definição operacional: `recorrente==1` e `nunca_acessou==1` na visão cliente-atual.\n")
        f.write(f"- Tamanho: **{q4_size}** clientes (**{q4_pct*100:.2f}%** do total).\n")
        f.write("Concentração por cluster:\n")
        f.write(_markdown_table(q4_by_cluster, floatfmt=".2%") + "\n\n")

        f.write("5) Que ações adotar para cada grupo?\n")
        f.write(_markdown_table(acoes[["cluster_id", "nome_cluster", "objetivo", "mensagem", "canal", "trigger", "kpi_esperado"]]) + "\n\n")

        f.write("6) Como classificar um novo cliente amanhã?\n")
        f.write("- Usar o pipeline treinado e salvo em `reports/models/kmeans_pipeline.joblib`.\n")
        f.write("- A entrada precisa conter as mesmas features do modelo (ver seção 9).\n\n")

        f.write("## 8. Plano de ação por cluster\n")
        f.write(_markdown_table(acoes[["cluster_id", "nome_cluster", "objetivo", "mensagem", "canal", "trigger", "kpi_esperado"]]) + "\n\n")

        f.write("## 9. Classificação automática de novos clientes\n")
        f.write("Exemplo de entrada (JSON):\n\n")
        f.write("```json\n")
        f.write(json.dumps(novo_cliente_exemplo, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        f.write("Saída (cluster previsto):\n\n")
        f.write(_markdown_table(pd.DataFrame([{"cluster_id": pred_cluster, "nome_cluster": pred_nome}])) + "\n\n")

        f.write("## 10. Limitações e próximos passos\n")
        f.write("- K-Means assume clusters aproximadamente esféricos e é sensível a escala; por isso usamos padronização e controle de outliers.\n")
        f.write("- Variáveis categóricas via one-hot podem criar dimensões raras; recomenda-se revisão de categorias pouco frequentes.\n")
        f.write("- Próximos passos: validar playbooks via testes A/B (KPIs: ativação, retenção, renovação e NPS) e explorar coortes/temporais.\n")

    k_table_out.to_csv(paths.metrics_dir / "k_metrics_table.csv", index=False)
    stability.to_csv(paths.metrics_dir / "k_stability_pairs.csv", index=False)


if __name__ == "__main__":
    run()

