from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Union, Any

app = FastAPI()

# Configurar CORS de forma mais permissiva
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos
    allow_headers=["*"],  # Permite todos os headers
    expose_headers=["*"]  # Expõe todos os headers
)

def sanitize_data(data: Any) -> Any:
    """Sanitiza os dados para garantir que são serializáveis em JSON."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return sanitize_data(data.to_dict('records') if isinstance(data, pd.DataFrame) else data.to_dict())
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data) if not np.isnan(data) and not np.isinf(data) else 0
    elif pd.isna(data):
        return None
    return data

def carregar_dados_vacinacao():
    try:
        # Carregar os dados de vacinação
        df_2021 = pd.read_excel('Basededado/Dados-Vacinação-2021.xlsx')
        df_2022 = pd.read_excel('Basededado/Dados-Vacinação-2022.xlsx')
        df_2023 = pd.read_excel('Basededado/Dados-Vacinação-2023.xlsx')
        df_2024 = pd.read_excel('Basededado/Dados-Vacinação-2024.xlsx')

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

        df_2021 = normalizar_colunas(df_2021)
        df_2021['ano'] = 2021
        df_2022 = normalizar_colunas(df_2022)
        df_2022['ano'] = 2022
        df_2023 = normalizar_colunas(df_2023)
        df_2023['ano'] = 2023
        df_2024 = normalizar_colunas(df_2024)
        df_2024['ano'] = 2024

        return pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)
    except Exception as e:
        print(f"Erro ao carregar dados de vacinação: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def carregar_dados_covid():
    try:
        # Carregar dados de COVID por estados e regiões
        df_estados = pd.read_excel('Basededado/DadosCovid-Estados.xlsx')
        df_regioes = pd.read_excel('Basededado/DadosCovid-Regiões.xlsx')

        # Renomear colunas
        df_estados = df_estados.rename(columns={
            'UF': 'estado',
            'População': 'populacao',
            'Casos Acumulados': 'casos',
            'Óbitos Acumulados': 'obitos',
            'Incidência covid-19 (100 mil hab)': 'incidencia',
            'Taxa mortalidade (100 mil hab)': 'taxa_mortalidade'
        })

        df_regioes = df_regioes.rename(columns={
            'Região': 'regiao',
            'População': 'populacao',
            'Casos Acumulados': 'casos',
            'Óbitos Acumulados': 'obitos',
            'Incidência covid-19 (100 mil hab)': 'incidencia',
            'Taxa mortalidade (100 mil hab)': 'taxa_mortalidade',
            'Casos novos notificados na semana epidemiológica': 'casos_novos',
            'Óbitos novos notificados na semana epidemiológica': 'obitos_novos'
        })

        return df_estados, df_regioes
    except Exception as e:
        print(f"Erro ao carregar dados COVID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dados-vacinacao")
async def get_dados_vacinacao():
    try:
        df_combinado = carregar_dados_vacinacao()
        
        totais_por_ano = df_combinado.groupby('ano').agg({
            'doses_aplicadas': 'sum',
            'primeira_dose': 'sum',
            'segunda_dose': 'sum',
            'terceira_dose': 'sum',
            'dose_reforco': 'sum',
            'dose_unica': 'sum'
        }).reset_index()
        
        return JSONResponse(content=sanitize_data(totais_por_ano))
    except Exception as e:
        print(f"Erro em get_dados_vacinacao: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dados-por-estado")
async def get_dados_por_estado():
    try:
        df_combinado = carregar_dados_vacinacao()
        media_por_estado_ano = df_combinado.groupby(['ano', 'estado'])['doses_aplicadas'].mean().reset_index()
        return JSONResponse(content=sanitize_data(media_por_estado_ano))
    except Exception as e:
        print(f"Erro em get_dados_por_estado: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/percentual-tipos-dose")
async def get_percentual_tipos_dose():
    try:
        df_combinado = carregar_dados_vacinacao()
        totais_por_ano = df_combinado.groupby('ano').agg({
            'doses_aplicadas': 'sum',
            'primeira_dose': 'sum',
            'segunda_dose': 'sum',
            'terceira_dose': 'sum',
            'dose_reforco': 'sum',
            'dose_unica': 'sum'
        }).reset_index()
        return JSONResponse(content=sanitize_data(totais_por_ano))
    except Exception as e:
        print(f"Erro em get_percentual_tipos_dose: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dados-covid-estados")
async def get_dados_covid_estados():
    try:
        df_estados, _ = carregar_dados_covid()
        df_estados['taxa_letalidade'] = (df_estados['obitos'] / df_estados['casos']) * 100
        return JSONResponse(content=sanitize_data(df_estados))
    except Exception as e:
        print(f"Erro em get_dados_covid_estados: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dados-covid-regioes")
async def get_dados_covid_regioes():
    try:
        _, df_regioes = carregar_dados_covid()
        return JSONResponse(content=sanitize_data(df_regioes))
    except Exception as e:
        print(f"Erro em get_dados_covid_regioes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/correlacao-vacinacao-mortalidade")
async def get_correlacao_vacinacao_mortalidade():
    try:
        df_vacinacao = carregar_dados_vacinacao()
        df_estados, _ = carregar_dados_covid()
        
        media_vacinacao = df_vacinacao.groupby('estado')['doses_aplicadas'].mean().reset_index()
        df_analise = df_estados.merge(media_vacinacao, on='estado', how='left')
        
        return JSONResponse(content=sanitize_data(df_analise))
    except Exception as e:
        print(f"Erro em get_correlacao_vacinacao_mortalidade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ranking-estados")
async def get_ranking_estados():
    try:
        df_estados, _ = carregar_dados_covid()
        df_vacinacao = carregar_dados_vacinacao()
        
        ranking = df_estados.merge(
            df_vacinacao.groupby('estado')['doses_aplicadas'].mean().reset_index(),
            on='estado',
            how='left'
        )
        
        rankings = {
            'casos': ranking.nlargest(10, 'casos')[['estado', 'casos']],
            'obitos': ranking.nlargest(10, 'obitos')[['estado', 'obitos']],
            'vacinacao': ranking.nlargest(10, 'doses_aplicadas')[['estado', 'doses_aplicadas']],
            'incidencia': ranking.nlargest(10, 'incidencia')[['estado', 'incidencia']]
        }
        
        return JSONResponse(content=sanitize_data(rankings))
    except Exception as e:
        print(f"Erro em get_ranking_estados: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 