from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def plot_missingness(report_df: pd.DataFrame, top_n: int = 20):
    d = report_df.sort_values("pct_missing", ascending=False).head(top_n)
    ax = sns.barplot(data=d, y="coluna", x="pct_missing")
    ax.set_title(f"Top {top_n} colunas por % missing")
    ax.set_xlabel("% missing")
    ax.set_ylabel("")
    return ax


def hist_kde(df: pd.DataFrame, col: str):
    ax = sns.histplot(df[col], kde=True, bins=40)
    ax.set_title(f"Distribuição: {col}")
    return ax


def boxplot(df: pd.DataFrame, col: str):
    ax = sns.boxplot(x=df[col])
    ax.set_title(f"Boxplot: {col}")
    return ax


def corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    ax = sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    ax.set_title("Correlação (numéricas)")
    return ax


def elbow_silhouette_plot(k_values: list[int], inertia: list[float], silhouette: list[float]):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(k_values, inertia, marker="o")
    ax[0].set_title("Elbow (Inertia)")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(k_values, silhouette, marker="o")
    ax[1].set_title("Silhouette")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Silhouette score")
    fig.tight_layout()
    return fig


def pca_2d_plot(X_scaled, labels, title: str = "PCA 2D"):
    if hasattr(X_scaled, "toarray"):
        X_dense = X_scaled.toarray()
    else:
        X_dense = np.asarray(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(X_dense)
    dfp = pd.DataFrame({"pc1": comp[:, 0], "pc2": comp[:, 1], "cluster_id": labels})
    ax = sns.scatterplot(data=dfp, x="pc1", y="pc2", hue="cluster_id", s=20, alpha=0.7)
    ax.set_title(title)
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


def radar_chart(cluster_profiles_norm: pd.DataFrame, cluster_col: str = "cluster_id"):
    import plotly.graph_objects as go

    categories = [c for c in cluster_profiles_norm.columns if c != cluster_col]
    fig = go.Figure()
    for _, row in cluster_profiles_norm.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=row[categories].tolist(),
                theta=categories,
                fill="toself",
                name=str(row[cluster_col]),
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig

