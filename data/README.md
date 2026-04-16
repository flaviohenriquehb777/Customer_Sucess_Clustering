# Dados (não versionados)

Por padrão, os arquivos de dados não são versionados no GitHub para evitar:
- arquivos grandes no repositório
- risco de expor dados sensíveis
- ruído no histórico de commits

Estrutura esperada:
- data/raw/: insumos originais do case
- data/processed/: bases processadas/derivadas
- data/new/: bases para inferência (novos clientes)
