import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Diretório para salvar os gráficos
if not os.path.exists('visualizacoes'):
    os.makedirs('visualizacoes')

# Carregar os dados
print("Carregando dados de vacinação para os anos 2021-2024...")

# Dados de vacinação por ano
df_2021 = pd.read_excel('Basededado/Dados-Vacinação-2021.xlsx')
df_2022 = pd.read_excel('Basededado/Dados-Vacinação-2022.xlsx')
df_2023 = pd.read_excel('Basededado/Dados-Vacinação-2023.xlsx')
df_2024 = pd.read_excel('Basededado/Dados-Vacinação-2024.xlsx')

# Normalizar os nomes das colunas para garantir consistência
def normalizar_colunas(df):
    mapping = {
        'UF': 'estado',
        'Total de Doses Aplicadas': 'doses_aplicadas',
        '1ª Dose': 'primeira_dose',
        '2ª Dose': 'segunda_dose',
        '3ª Dose': 'terceira_dose',
        'Dose Reforço': 'dose_reforco',
        'Dose Única': 'dose_unica',
        'População': 'populacao'
    }
    return df.rename(columns={col: mapping.get(col, col) for col in df.columns})

# Normalizar colunas e adicionar coluna de ano
df_2021 = normalizar_colunas(df_2021)
df_2021['ano'] = 2021

df_2022 = normalizar_colunas(df_2022)
df_2022['ano'] = 2022

df_2023 = normalizar_colunas(df_2023)
df_2023['ano'] = 2023

df_2024 = normalizar_colunas(df_2024)
df_2024['ano'] = 2024

# Exibir informações dos dataframes
print("\nInformações dos dados de vacinação por ano:")
for ano, df in [('2021', df_2021), ('2022', df_2022), ('2023', df_2023), ('2024', df_2024)]:
    print(f"\nDados de {ano}:")
    print(df.info())
    print("\nPrimeiras 3 linhas:")
    print(df.head(3))

# Combinar os dataframes para análise
df_combinado = pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)

# Calcular totais por ano
totais_por_ano = df_combinado.groupby('ano').agg({
    'doses_aplicadas': 'sum',
    'primeira_dose': 'sum',
    'segunda_dose': 'sum',
    'terceira_dose': 'sum',
    'dose_reforco': 'sum',
    'dose_unica': 'sum'
}).reset_index()

print("\nTotais de vacinação por ano:")
print(totais_por_ano)

# Visualizações
print("\nCriando visualizações comparativas...")

# 1. Gráfico de barras: Total de doses aplicadas por ano
plt.figure(figsize=(12, 6))
sns.barplot(x='ano', y='doses_aplicadas', data=totais_por_ano)
plt.title('Total de Doses Aplicadas por Ano (2021-2024)')
plt.xlabel('Ano')
plt.ylabel('Número de Doses')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizacoes/doses_aplicadas_por_ano.png')

# 2. Gráfico de linhas: Evolução dos tipos de doses ao longo dos anos
plt.figure(figsize=(14, 8))
tipos_doses = ['primeira_dose', 'segunda_dose', 'terceira_dose', 'dose_reforco', 'dose_unica']
for tipo in tipos_doses:
    if tipo in totais_por_ano.columns:
        plt.plot(totais_por_ano['ano'], totais_por_ano[tipo], marker='o', linewidth=2, label=tipo.replace('_', ' ').title())
plt.title('Evolução dos Tipos de Doses Aplicadas ao Longo dos Anos (2021-2024)')
plt.xlabel('Ano')
plt.ylabel('Número de Doses')
plt.legend()
plt.grid(True)
plt.xticks(totais_por_ano['ano'])
plt.tight_layout()
plt.savefig('visualizacoes/evolucao_tipos_doses.png')

# 3. Gráfico de área empilhada: Composição das doses por ano
plt.figure(figsize=(14, 8))
tipos_doses_presentes = [tipo for tipo in tipos_doses if tipo in totais_por_ano.columns]
plt.stackplot(totais_por_ano['ano'], 
              [totais_por_ano[tipo] for tipo in tipos_doses_presentes],
              labels=[tipo.replace('_', ' ').title() for tipo in tipos_doses_presentes],
              alpha=0.8)
plt.title('Composição das Doses por Ano (2021-2024)')
plt.xlabel('Ano')
plt.ylabel('Número de Doses')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(totais_por_ano['ano'])
plt.tight_layout()
plt.savefig('visualizacoes/composicao_doses_por_ano.png')

# 4. Média de doses por estado em cada ano
media_por_estado_ano = df_combinado.groupby(['ano', 'estado'])['doses_aplicadas'].mean().reset_index()
top_estados = df_combinado.groupby('estado')['doses_aplicadas'].sum().nlargest(5).index

plt.figure(figsize=(14, 8))
for estado in top_estados:
    dados_estado = media_por_estado_ano[media_por_estado_ano['estado'] == estado]
    plt.plot(dados_estado['ano'], dados_estado['doses_aplicadas'], marker='o', linewidth=2, label=estado)
plt.title('Média de Doses Aplicadas nos 5 Estados com Maior Vacinação (2021-2024)')
plt.xlabel('Ano')
plt.ylabel('Média de Doses Aplicadas')
plt.legend()
plt.grid(True)
plt.xticks(df_combinado['ano'].unique())
plt.tight_layout()
plt.savefig('visualizacoes/media_doses_top5_estados.png')

# 5. Gráfico de radar: Comparação dos tipos de dose por ano
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])

anos = totais_por_ano['ano'].tolist()
tipos_doses_presentes = [tipo for tipo in tipos_doses if tipo in totais_por_ano.columns]
categories = [tipo.replace('_', ' ').title() for tipo in tipos_doses_presentes]

for i, ano in enumerate(anos):
    values = totais_por_ano.loc[totais_por_ano['ano'] == ano, tipos_doses_presentes].values.flatten().tolist()
    # Fechar o gráfico de radar conectando o último ponto ao primeiro
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name=f'Ano {ano}'
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(totais_por_ano[tipos_doses_presentes].max()) * 1.1]
        )
    ),
    showlegend=True,
    title='Comparação dos Tipos de Dose por Ano (2021-2024)'
)

fig.write_html('visualizacoes/radar_doses_por_ano.html')

# 6. Comparação percentual de cada tipo de dose em relação ao total por ano
totais_por_ano_perc = totais_por_ano.copy()
for tipo in tipos_doses_presentes:
    totais_por_ano_perc[f'{tipo}_perc'] = (totais_por_ano_perc[tipo] / totais_por_ano_perc['doses_aplicadas']) * 100

# Converter para formato longo para gráfico
perc_data = pd.melt(
    totais_por_ano_perc, 
    id_vars=['ano'], 
    value_vars=[f'{tipo}_perc' for tipo in tipos_doses_presentes],
    var_name='tipo_dose', 
    value_name='percentual'
)
perc_data['tipo_dose'] = perc_data['tipo_dose'].str.replace('_perc', '').str.replace('_', ' ').str.title()

plt.figure(figsize=(14, 8))
sns.barplot(x='ano', y='percentual', hue='tipo_dose', data=perc_data)
plt.title('Distribuição Percentual dos Tipos de Dose por Ano (2021-2024)')
plt.xlabel('Ano')
plt.ylabel('Percentual do Total (%)')
plt.legend(title='Tipo de Dose')
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizacoes/percentual_tipos_dose_por_ano.png')

# 7. Taxa de variação anual do total de doses
totais_por_ano['variacao_percentual'] = totais_por_ano['doses_aplicadas'].pct_change() * 100

plt.figure(figsize=(10, 6))
sns.barplot(x='ano', y='variacao_percentual', data=totais_por_ano[1:])  # Começando do segundo ano para ter a variação
plt.title('Variação Percentual Anual no Total de Doses Aplicadas (2022-2024)')
plt.xlabel('Ano')
plt.ylabel('Variação Percentual (%)')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizacoes/variacao_percentual_anual.png')

print("\nAnálise e visualização comparativa concluídas!")
print("Os gráficos foram salvos no diretório 'visualizacoes/'") 