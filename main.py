import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)

# 1. Carregar os dados
print("Carregando dados...")

# Dados de COVID por estados
df_estados = pd.read_excel('Basededado/DadosCovid-Estados.xlsx')

# Dados de COVID por regiões
df_regioes = pd.read_excel('Basededado/DadosCovid-Regiões.xlsx')

# Dados de vacinação
df_vacinacao = pd.read_excel('Basededado/Dados-Vacinação.xlsx')
df_vacinacao_2021 = pd.read_excel('Basededado/Dados-Vacinação-2021.xlsx')
df_vacinacao_2022 = pd.read_excel('Basededado/Dados-Vacinação-2022.xlsx')
df_vacinacao_2023 = pd.read_excel('Basededado/Dados-Vacinação-2023.xlsx')

# 2. Exibir informações sobre os dados
print("\n===== INFORMAÇÕES SOBRE OS DADOS =====")

print("\nDados COVID por estados:")
print(df_estados.info())
print("\nPrimeiras 5 linhas:")
print(df_estados.head())

print("\nDados COVID por regiões:")
print(df_regioes.info())
print("\nPrimeiras 5 linhas:")
print(df_regioes.head())

print("\nDados de vacinação:")
print(df_vacinacao.info())
print("\nPrimeiras 5 linhas:")
print(df_vacinacao.head())

# 3. Pré-processamento dos dados
print("\n===== PRÉ-PROCESSAMENTO DOS DADOS =====")

# Verificar valores nulos
print("\nVerificando valores nulos nos dados de estados:")
print(df_estados.isnull().sum())

print("\nVerificando valores nulos nos dados de regiões:")
print(df_regioes.isnull().sum())

print("\nVerificando valores nulos nos dados de vacinação:")
print(df_vacinacao.isnull().sum())

# Renomear colunas para facilitar o uso
df_estados_renamed = df_estados.rename(columns={
    'UF': 'estado',
    'População': 'populacao',
    'Casos Acumulados': 'casos',
    'Óbitos Acumulados': 'obitos',
    'Incidência covid-19 (100 mil hab)': 'incidencia',
    'Taxa mortalidade (100 mil hab)': 'taxa_mortalidade'
})

df_regioes_renamed = df_regioes.rename(columns={
    'Região': 'regiao',
    'População': 'populacao',
    'Casos Acumulados': 'casos',
    'Óbitos Acumulados': 'obitos',
    'Incidência covid-19 (100 mil hab)': 'incidencia',
    'Taxa mortalidade (100 mil hab)': 'taxa_mortalidade',
    'Casos novos notificados na semana epidemiológica': 'casos_novos',
    'Óbitos novos notificados na semana epidemiológica': 'obitos_novos'
})

df_vacinacao_renamed = df_vacinacao.rename(columns={
    'UF': 'estado',
    'Total de Doses Aplicadas': 'doses_aplicadas',
    '1ª Dose': 'primeira_dose',
    '2ª Dose': 'segunda_dose',
    '3ª Dose': 'terceira_dose',
    'Dose Reforço': 'dose_reforco',
    'Dose Única': 'dose_unica'
})

# 4. Definição e Construção de Indicadores (KPIs)
print("\n===== INDICADORES (KPIs) =====")

# KPI 1: Taxa de letalidade por estado
df_estados_renamed['taxa_letalidade'] = (df_estados_renamed['obitos'] / df_estados_renamed['casos']) * 100
print("\nTaxa de letalidade por estado:")
print(df_estados_renamed[['estado', 'taxa_letalidade']].sort_values(by='taxa_letalidade', ascending=False))

# KPI 2: Taxa de vacinação (cobertura) por estado
if 'populacao' not in df_vacinacao_renamed.columns:
    # Obter população dos estados do dataframe de estados
    pop_estados = df_estados_renamed[['estado', 'populacao']]
    df_vacinacao_renamed = df_vacinacao_renamed.merge(pop_estados, on='estado', how='left')

df_vacinacao_renamed['taxa_vacinacao'] = (df_vacinacao_renamed['primeira_dose'] / df_vacinacao_renamed['populacao']) * 100
print("\nTaxa de cobertura vacinal (1ª dose) por estado:")
print(df_vacinacao_renamed[['estado', 'taxa_vacinacao']].sort_values(by='taxa_vacinacao', ascending=False))

# KPI 3: Relação entre vacinação e mortalidade
df_analise = df_estados_renamed.merge(
    df_vacinacao_renamed[['estado', 'doses_aplicadas', 'primeira_dose', 'taxa_vacinacao']], 
    on='estado', 
    how='left'
)

# 5. Visualização de Dados e Dashboards (Storytelling)
print("\n===== VISUALIZAÇÃO DE DADOS =====")

# Criar diretório para salvar os gráficos
import os
if not os.path.exists('visualizacoes'):
    os.makedirs('visualizacoes')

# Gráfico 1: Casos por estado (top 10)
plt.figure(figsize=(12, 6))
top_estados_casos = df_estados_renamed.sort_values(by='casos', ascending=False).head(10)
sns.barplot(x='estado', y='casos', data=top_estados_casos)
plt.title('Top 10 Estados com Mais Casos de COVID-19')
plt.xlabel('Estado')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizacoes/top10_estados_casos.png')

# Gráfico 2: Óbitos por estado (top 10)
plt.figure(figsize=(12, 6))
top_estados_obitos = df_estados_renamed.sort_values(by='obitos', ascending=False).head(10)
sns.barplot(x='estado', y='obitos', data=top_estados_obitos)
plt.title('Top 10 Estados com Mais Óbitos por COVID-19')
plt.xlabel('Estado')
plt.ylabel('Número de Óbitos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizacoes/top10_estados_obitos.png')

# Gráfico 3: Taxa de letalidade por estado (top 10)
plt.figure(figsize=(12, 6))
top_letalidade = df_estados_renamed.sort_values(by='taxa_letalidade', ascending=False).head(10)
sns.barplot(x='estado', y='taxa_letalidade', data=top_letalidade)
plt.title('Top 10 Estados com Maior Taxa de Letalidade')
plt.xlabel('Estado')
plt.ylabel('Taxa de Letalidade (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizacoes/top10_estados_letalidade.png')

# Gráfico 4: Correlação entre variáveis
plt.figure(figsize=(10, 8))
corr_cols = ['casos', 'obitos', 'incidencia', 'taxa_mortalidade', 'taxa_letalidade']
corr_matrix = df_estados_renamed[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlação entre Variáveis COVID-19')
plt.tight_layout()
plt.savefig('visualizacoes/correlacao_variaveis.png')

# Gráfico 5: Distribuição de casos por região
plt.figure(figsize=(10, 6))
sns.barplot(x='regiao', y='casos', data=df_regioes_renamed)
plt.title('Distribuição de Casos por Região')
plt.xlabel('Região')
plt.ylabel('Número de Casos')
plt.tight_layout()
plt.savefig('visualizacoes/casos_por_regiao.png')

# Gráfico 6: Distribuição de óbitos por região
plt.figure(figsize=(10, 6))
sns.barplot(x='regiao', y='obitos', data=df_regioes_renamed)
plt.title('Distribuição de Óbitos por Região')
plt.xlabel('Região')
plt.ylabel('Número de Óbitos')
plt.tight_layout()
plt.savefig('visualizacoes/obitos_por_regiao.png')

# 6. Transformação de Dados e Feature Engineering
print("\n===== TRANSFORMAÇÃO DE DADOS E FEATURE ENGINEERING =====")

# Criando novas características para análise
df_analise = df_estados_renamed.copy()

# Normalização de dados para casos per capita (já temos incidência, mas vamos manter o exemplo)
df_analise['casos_per_capita'] = df_analise['incidencia'] 

# Categorização da taxa de letalidade
df_analise['letalidade_categoria'] = pd.cut(
    df_analise['taxa_letalidade'], 
    bins=[0, 1.5, 2.5, 3.5, 100], 
    labels=['Baixa (0-1.5%)', 'Média (1.5-2.5%)', 'Alta (2.5-3.5%)', 'Muito Alta (3.5%+)']
)
print("\nDistribuição das categorias de letalidade:")
print(df_analise['letalidade_categoria'].value_counts())

# Calcular a efetividade da vacinação (redução de mortalidade por dose aplicada)
if 'taxa_vacinacao' in df_analise.columns:
    # Relação entre taxa de vacinação e mortalidade
    plt.figure(figsize=(10, 6))
    plt.scatter(df_analise['taxa_vacinacao'], df_analise['taxa_mortalidade'])
    plt.title('Relação entre Taxa de Vacinação e Taxa de Mortalidade')
    plt.xlabel('Taxa de Vacinação (%)')
    plt.ylabel('Taxa de Mortalidade (por 100 mil hab)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizacoes/vacinacao_vs_mortalidade.png')

# 7. Modelagem Preditiva
print("\n===== MODELAGEM PREDITIVA =====")

# Preparar dados para modelagem
# a. Regressão Linear: Prever óbitos com base em casos e outros indicadores
features = ['casos', 'incidencia']
if 'taxa_vacinacao' in df_analise.columns:
    features.append('taxa_vacinacao')

X = df_analise[features]
y = df_analise['obitos']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a. Regressão Linear
print("\nModelo de Regressão Linear:")
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)
y_pred_reg = reg_model.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Gráfico de predições vs valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Valores Reais')
plt.ylabel('Predições')
plt.title('Regressão Linear: Valores Reais vs. Predições')
plt.tight_layout()
plt.savefig('visualizacoes/regressao_linear_predicoes.png')

# b. Regressão Logística (para classificação de letalidade alta/baixa)
# Classificar estados com alta/baixa letalidade
if 'letalidade_categoria' in df_analise.columns:
    # Converter para classificação binária
    y_class = (df_analise['letalidade_categoria'] == 'Alta (2.5-3.5%)') | (df_analise['letalidade_categoria'] == 'Muito Alta (3.5%+)')
    
    # Dividir em treino e teste
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42)
    
    # Normalizar os dados
    X_train_class_scaled = scaler.fit_transform(X_train_class)
    X_test_class_scaled = scaler.transform(X_test_class)
    
    # Regressão Logística
    print("\nModelo de Regressão Logística:")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_class_scaled, y_train_class)
    y_pred_log = log_model.predict(X_test_class_scaled)
    
    print(f"Acurácia: {accuracy_score(y_test_class, y_pred_log):.2f}")
    print(f"Precisão: {precision_score(y_test_class, y_pred_log, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test_class, y_pred_log, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_test_class, y_pred_log, zero_division=0):.2f}")
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test_class, y_pred_log)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('visualizacoes/matriz_confusao_logistica.png')
    
    # Árvore de Decisão
    print("\nModelo de Árvore de Decisão:")
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train_class_scaled, y_train_class)
    y_pred_tree = tree_model.predict(X_test_class_scaled)
    
    print(f"Acurácia: {accuracy_score(y_test_class, y_pred_tree):.2f}")
    print(f"Precisão: {precision_score(y_test_class, y_pred_tree, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test_class, y_pred_tree, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_test_class, y_pred_tree, zero_division=0):.2f}")
    
    # KNN
    print("\nModelo KNN:")
    knn_model = KNeighborsClassifier(n_neighbors=3)  # Usando n=3 por termos poucos dados
    knn_model.fit(X_train_class_scaled, y_train_class)
    y_pred_knn = knn_model.predict(X_test_class_scaled)
    
    print(f"Acurácia: {accuracy_score(y_test_class, y_pred_knn):.2f}")
    print(f"Precisão: {precision_score(y_test_class, y_pred_knn, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test_class, y_pred_knn, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_test_class, y_pred_knn, zero_division=0):.2f}")

# 8. Algoritmos de Machine Learning (Clusterização)
print("\n===== CLUSTERIZAÇÃO =====")

# Selecionando features para clusterização
features_cluster = ['casos', 'obitos', 'incidencia', 'taxa_mortalidade']
if 'taxa_letalidade' in df_analise.columns:
    features_cluster.append('taxa_letalidade')

X_cluster = df_analise[features_cluster].dropna()

# Normalizar dados
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# a. K-Means
print("\nClusterização K-Means:")

# Determinando o número ótimo de clusters (método do cotovelo)
inertia = []
k_range = range(1, min(11, len(X_cluster)))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

# Plotar gráfico do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Método do Cotovelo para Determinar Número Ótimo de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizacoes/kmeans_cotovelo.png')

# Aplicar K-Means com o número ótimo de clusters (vamos usar 3 como exemplo)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_analise['cluster_kmeans'] = kmeans.fit_predict(X_cluster_scaled)

print(f"\nDistribuição dos clusters K-Means (k={n_clusters}):")
print(df_analise['cluster_kmeans'].value_counts())

# Visualização dos clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='casos', y='obitos', hue='cluster_kmeans', data=df_analise, palette='viridis')
plt.title(f'Clusterização K-Means (k={n_clusters})')
plt.xlabel('Casos')
plt.ylabel('Óbitos')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('visualizacoes/kmeans_clusters.png')

# Análise dos clusters
plt.figure(figsize=(14, 10))
cluster_data = df_analise.groupby('cluster_kmeans')[features_cluster].mean().reset_index()
cluster_data = pd.melt(cluster_data, id_vars=['cluster_kmeans'], value_vars=features_cluster)
sns.barplot(x='cluster_kmeans', y='value', hue='variable', data=cluster_data)
plt.title('Características Médias de Cada Cluster')
plt.xlabel('Cluster')
plt.ylabel('Valor Médio (Normalizado)')
plt.legend(title='Variável')
plt.tight_layout()
plt.savefig('visualizacoes/kmeans_caracteristicas.png')

# b. DBSCAN
print("\nClusterização DBSCAN:")

dbscan = DBSCAN(eps=1.0, min_samples=2)  # Parâmetros ajustados para conjunto pequeno
df_analise['cluster_dbscan'] = dbscan.fit_predict(X_cluster_scaled)

print("\nDistribuição dos clusters DBSCAN:")
print(df_analise['cluster_dbscan'].value_counts())

# Visualização dos clusters DBSCAN
plt.figure(figsize=(10, 8))
sns.scatterplot(x='casos', y='obitos', hue='cluster_dbscan', data=df_analise, palette='viridis')
plt.title('Clusterização DBSCAN')
plt.xlabel('Casos')
plt.ylabel('Óbitos')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('visualizacoes/dbscan_clusters.png')

# 9. Métricas de Avaliação de Modelos
print("\n===== MÉTRICAS DE AVALIAÇÃO DE MODELOS =====")

# Validação cruzada para modelo de regressão
print("\nValidação Cruzada para Regressão Linear:")
cv_scores = cross_val_score(reg_model, X, y, cv=5, scoring='r2')
print(f"Scores R² para cada fold: {cv_scores}")
print(f"Média R²: {cv_scores.mean():.2f}")
print(f"Desvio Padrão R²: {cv_scores.std():.2f}")

# Validação cruzada para classificação
if 'letalidade_categoria' in df_analise.columns and 'y_class' in locals():
    print("\nValidação Cruzada para Árvore de Decisão:")
    cv_scores_tree = cross_val_score(tree_model, X, y_class, cv=5, scoring='f1')
    print(f"Scores F1 para cada fold: {cv_scores_tree}")
    print(f"Média F1: {cv_scores_tree.mean():.2f}")
    print(f"Desvio Padrão F1: {cv_scores_tree.std():.2f}")

print("\n===== ANÁLISE FINALIZADA =====")
print("Os gráficos foram salvos no diretório 'visualizacoes/'")
print("Obrigado por utilizar este script de análise!")
