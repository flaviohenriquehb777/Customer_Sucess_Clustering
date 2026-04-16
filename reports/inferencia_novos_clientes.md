# Inferência em novos clientes (K-Means) — Exportação

Este material complementa a etapa de “Exportação (etapa esperada no case)” com **inferência (valores preditivos)** em uma nova base no mesmo schema do treino.

## Inputs
- Base: `data/new/Customer Sucess_novos_clientes.xlsx`
- Pipeline treinado: `reports/models/kmeans_pipeline.joblib`

## Notebook
- Execute do zero: `notebooks/08_inferencia_novos_clientes.ipynb`

## Outputs
- Export para auditoria: `reports/novos_clientes_clusterizados.xlsx`
  - colunas mínimas: nome do cliente, `cluster_id`, `nome_cluster`

## Mapeamento fixo
- cluster 4 → 🟢 Champions
- cluster 0 → 🟡 Potenciais
- cluster 1 → 🟠 Avulsos Engajados
- cluster 2 → 🔴 Zumbis
- cluster 3 → ⚪ Churn Iminente
- cluster 5 → 🟣 Novos
