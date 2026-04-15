# Customer Success — Segmentação de Clientes (K-Means)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flávio%20Henrique%20Barbosa-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/flávio-henrique-barbosa-38465938)
[![Email](https://img.shields.io/badge/Email-flaviohenriquehb777%40outlook.com-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:flaviohenriquehb777@outlook.com)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit--Learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

## Sumário
- [Visão Geral do Modelo](#visão-geral-do-modelo)
- [Objetivos da Análise](#objetivos-da-análise)
- [Estrutura do Modelo](#estrutura-do-modelo)
- [Base de Dados](#base-de-dados)
- [Metodologia de Análise](#metodologia-de-análise)
- [Resultados Chave e Apresentação](#resultados-chave-e-apresentação)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação e Uso](#instalação-e-uso)
- [Publicação no GitHub Pages](#publicação-no-github-pages)
- [Licença](#licença)
- [Contato](#contato)

## Visão Geral do Modelo

Este projeto segmenta clientes da Insight Academy usando **K-Means** com foco em **ações de negócio** (retenção, engajamento e receita). A modelagem é feita em nível de **cliente** (visão cliente-atual), consolidando a base transacional e preservando histórico via features agregadas.

Entregáveis principais:
- Relatório executivo completo: `reports/final_report.md`
- Artefatos (figuras): `reports/assets/`
- Pipeline de classificação: `reports/models/kmeans_pipeline.joblib`
- Exports (Excel/JSON): `reports/exports/`

## Objetivos da Análise

Responder às perguntas do case:
- Quantos grupos existem e qual critério para escolha de K?
- Qual perfil de cada grupo (uso, inatividade, pagamento, recorrência, parcelamento)?
- Quais grupos se destacam (positivo/negativo) e por quê?
- Qual o tamanho do grupo “recorrente que não acessa”?
- Quais ações de marketing/CS/vendas por grupo?
- Como classificar novos clientes automaticamente?

## Estrutura do Modelo

- `src/preprocessing.py`: limpeza, padronização e visão cliente-atual
- `src/features.py`: especificação e seleção de features de modelagem
- `src/modeling.py`: pipelines (pré-processamento + scaler + KMeans) e métricas
- `src/business_refresh.py`: geração de artefatos (K orientado ao negócio) e relatório final
- `notebooks/`: notebooks sequenciais do projeto (EDA → features → seleção de K → perfis → SHAP)

## Base de Dados

Arquivos brutos (não alterar):
- `data/raw/Base Customer Sucess.xlsx` (aba de clientes)
- `data/raw/Customer Sucess.pdf` (dicionário/narrativa do case)

Bases processadas:
- `data/processed/base_limpa_transacao.parquet`: base limpa no nível transacional
- `data/processed/base_cliente_atual.parquet`: 1 linha por cliente (registro mais recente por `data_ordem` + agregados)

## Metodologia de Análise

- Unidade de análise: **cliente** (linhas originais são transações; clientes podem repetir)
- Pré-processamento: imputação, one-hot (categóricas) e **StandardScaler**
- Escolha de K:
  - Métricas: Inertia (Elbow), Silhouette, Calinski-Harabasz, Davies-Bouldin
  - Critérios de negócio: tamanho mínimo de cluster e acionabilidade
  - Estabilidade: ARI/NMI via subamostragem (execuções reprodutíveis)
- Treino final: `KMeans(random_state=42)` em pipeline salvo via joblib
- Explicabilidade: SHAP via modelo supervisionado que aproxima a separação dos clusters

## Resultados Chave e Apresentação

Os resultados finais, tabelas e gráficos estão consolidados em:
- `reports/final_report.md`

Imagens principais:
- `reports/assets/k_selection.png` (métricas por K)
- `reports/assets/k_stability.png` (estabilidade ARI/NMI por K)
- `reports/assets/pca_clusters.png`, `reports/assets/cluster_heatmap.png`, `reports/assets/radar_clusters.png`

## Tecnologias Utilizadas

- Python 3
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- openpyxl/xlsxwriter (export)
- shap (explicabilidade)

## Instalação e Uso

1) Criar ambiente e instalar dependências:

```bash
pip install -r requirements.txt
```

2) Executar notebooks (opcional):
- Rode os notebooks em `notebooks/` em ordem numérica.

3) Regerar artefatos e relatório final (recomendado):

```bash
python src/business_refresh.py
```

Saídas:
- `reports/final_report.md`
- `reports/assets/*`
- `reports/models/kmeans_pipeline.joblib`
- `reports/exports/*`

## Publicação no GitHub Pages

Uma forma simples de publicar:
- Copie `reports/final_report.md` e a pasta `reports/assets/` para um diretório `docs/`.
- No GitHub: Settings → Pages → Source: `docs/` (branch principal).

## Licença

Veja `LICENSE.md`.

## Contato

- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [linkedin.com/in/flávio-henrique-barbosa-38465938](https://linkedin.com/in/flávio-henrique-barbosa-38465938)
- **Email:** [flaviohenriquehb777@outlook.com](mailto:flaviohenriquehb777@outlook.com)
- Para dúvidas e ajustes do modelo/relatório, use issues ou abra um PR no repositório.
