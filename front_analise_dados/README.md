# Frontend - Análise de Dados de Vacinação COVID-19

Este é o frontend da aplicação de análise de dados de vacinação COVID-19. Ele exibe visualizações interativas dos dados de vacinação processados pelo backend Python.

## Requisitos

- Node.js (versão 14 ou superior)
- npm (gerenciador de pacotes do Node.js)
- Backend Python em execução (porta 8000)

## Instalação

1. Instale as dependências do projeto:
```bash
npm install
```

2. Inicie o servidor de desenvolvimento:
```bash
npm start
```

O aplicativo será aberto automaticamente em seu navegador padrão no endereço `http://localhost:3000`.

## Funcionalidades

O frontend apresenta três visualizações principais:

1. **Total de Doses Aplicadas por Ano**
   - Gráfico de barras mostrando o total de doses aplicadas em cada ano

2. **Evolução dos Tipos de Doses**
   - Gráfico de linhas mostrando a evolução de cada tipo de dose ao longo dos anos
   - Inclui primeira dose, segunda dose, terceira dose, dose de reforço e dose única

3. **Top 5 Estados - Média de Doses Aplicadas**
   - Mapa de calor mostrando a média de doses aplicadas nos 5 estados com maior vacinação
   - Visualização por estado e ano

## Estrutura do Projeto

- `src/App.tsx`: Componente principal com as visualizações
- `src/index.css`: Estilos globais da aplicação

## Executando o Projeto Completo

1. Primeiro, inicie o backend Python:
```bash
cd ..
python -m pip install fastapi uvicorn pandas openpyxl
python -m uvicorn api:app --reload
```

2. Em outro terminal, inicie o frontend:
```bash
cd front_analise_dados
npm start
```

## Tecnologias Utilizadas

- React
- TypeScript
- Plotly.js
- Axios
