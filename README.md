# Projeto de Análise de Dados COVID-19

Este projeto realiza uma análise completa dos dados da COVID-19 no Brasil, incluindo visualizações, indicadores-chave, modelagem preditiva e técnicas de aprendizado de máquina.

## Visão Geral

O projeto analisa dados da COVID-19 por estados e regiões do Brasil, junto com dados de vacinação, para extrair insights relevantes sobre a pandemia. A análise abrange:

- Indicadores-chave (KPIs) como taxa de letalidade e cobertura vacinal
- Visualizações de dados para storytelling
- Transformação de dados e feature engineering
- Modelagem preditiva (regressão e classificação)
- Algoritmos de clusterização
- Avaliação de modelos

## Estrutura do Projeto

- `main.py`: Script principal que executa toda a análise
- `Basededado/`: Diretório contendo os arquivos de dados
- `visualizacoes/`: Diretório onde as visualizações são salvas
- `conclusoes.md`: Documento com as principais conclusões e insights
- `README.md`: Este arquivo

## Requisitos

O projeto requer Python 3.x e as seguintes bibliotecas:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
mlxtend
plotly
openpyxl
```

## Instalação

Para instalar as dependências necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend plotly openpyxl
```

## Uso

Para executar a análise completa:

```bash
python main.py
```

Isso processará os dados, gerará visualizações e exibirá os resultados no console.

## Visualizações Geradas

O script gera diversas visualizações, incluindo:

1. Top 10 estados com mais casos de COVID-19
2. Top 10 estados com mais óbitos por COVID-19
3. Top 10 estados com maior taxa de letalidade
4. Matriz de correlação entre variáveis
5. Distribuição de casos e óbitos por região
6. Relação entre taxa de vacinação e mortalidade
7. Gráficos de modelagem preditiva
8. Visualização de clusters

Todas as visualizações são salvas no diretório `visualizacoes/`.

## Principais Funcionalidades

### 1. Indicadores-Chave (KPIs)

- Taxa de letalidade por estado
- Taxa de cobertura vacinal por estado
- Relação entre vacinação e mortalidade

### 2. Modelagem Preditiva

- **Regressão Linear**: Prevê óbitos com base em casos e outros indicadores
- **Regressão Logística**: Classifica estados por alta/baixa letalidade
- **Árvore de Decisão**: Modelo alternativo para classificação
- **KNN**: Classificação baseada em vizinhos próximos

### 3. Clusterização

- **K-Means**: Agrupa estados com características similares
- **DBSCAN**: Identificação de clusters baseada em densidade

### 4. Avaliação de Modelos

- Validação cruzada
- Matriz de confusão
- Métricas de avaliação (R², acurácia, precisão, recall, F1-score)

## Conclusões

As conclusões detalhadas da análise estão disponíveis no arquivo `conclusoes.md`. Elas incluem insights sobre:

- Padrões regionais de casos e óbitos
- Efetividade da vacinação
- Fatores associados à letalidade
- Recomendações baseadas nos dados

## Melhorias Futuras

Ideias para aprimorar o projeto:

1. Adicionar análise temporal da evolução da pandemia
2. Incorporar dados socioeconômicos para análise de fatores de risco
3. Desenvolver um dashboard interativo para visualização dinâmica
4. Implementar modelos de séries temporais para previsões
5. Análise comparativa com dados internacionais

## Autores

- Desenvolvido como projeto de análise de dados de COVID-19 