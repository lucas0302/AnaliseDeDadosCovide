import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import './App.css';

function App() {
  const [dadosVacinacao, setDadosVacinacao] = useState([]);
  const [dadosEstado, setDadosEstado] = useState([]);
  const [percentuais, setPercentuais] = useState([]);
  const [dadosCovid, setDadosCovid] = useState([]);
  const [dadosRegioes, setDadosRegioes] = useState([]);
  const [correlacaoData, setCorrelacaoData] = useState([]);
  const [rankingEstados, setRankingEstados] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Iniciando busca de dados...');
        
        const [
          resVacinacao, 
          resEstado, 
          resPercentual, 
          resCovid,
          resRegioes,
          resCorrelacao,
          resRanking
        ] = await Promise.all([
          axios.get('http://localhost:8000/api/dados-vacinacao'),
          axios.get('http://localhost:8000/api/dados-por-estado'),
          axios.get('http://localhost:8000/api/percentual-tipos-dose'),
          axios.get('http://localhost:8000/api/dados-covid-estados'),
          axios.get('http://localhost:8000/api/dados-covid-regioes'),
          axios.get('http://localhost:8000/api/correlacao-vacinacao-mortalidade'),
          axios.get('http://localhost:8000/api/ranking-estados')
        ]);

        console.log('Dados de vacinação:', resVacinacao.data);
        console.log('Dados por estado:', resEstado.data);
        console.log('Dados percentuais:', resPercentual.data);
        console.log('Dados COVID estados:', resCovid.data);
        console.log('Dados regiões:', resRegioes.data);
        console.log('Dados correlação:', resCorrelacao.data);
        console.log('Ranking estados:', resRanking.data);

        setDadosVacinacao(resVacinacao.data || []);
        setDadosEstado(resEstado.data || []);
        setPercentuais(resPercentual.data || []);
        setDadosCovid(resCovid.data || []);
        setDadosRegioes(resRegioes.data || []);
        setCorrelacaoData(resCorrelacao.data || []);
        setRankingEstados(resRanking.data || {});
        setLoading(false);
        setError(null);
      } catch (error) {
        console.error('Erro ao carregar dados:', error);
        setError(error.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Carregando dados...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Erro ao carregar dados</h2>
        <h2>Erro ao carregar dados</h2>
        <p>{error}</p>
        <p>Verifique se a API está rodando em http://localhost:8000</p>
      </div>
    );
  }


  // Configurações base para os gráficos
  const plotConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  };

  const baseLayout = {
    autosize: true,
    height: 400,
    margin: { t: 30, b: 40, l: 60, r: 20 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }
  };

  // Calcular estatísticas para os cards
  const totalDoses = dadosVacinacao.reduce((acc, curr) => acc + (curr.doses_aplicadas || 0), 0);
  const totalCasos = dadosCovid.reduce((acc, curr) => acc + (curr.casos || 0), 0);
  const totalObitos = dadosCovid.reduce((acc, curr) => acc + (curr.obitos || 0), 0);
  const mediaLetal = dadosCovid.length > 0 
    ? dadosCovid.reduce((acc, curr) => acc + (curr.taxa_letalidade || 0), 0) / dadosCovid.length
    : 0;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Dashboard de Vacinação COVID-19</h1>
        <p className="subtitle">Análise Completa da Vacinação no Brasil</p>
      </header>

      <div className="cards-container">
        <div className="info-card">
          <h3>Total de Doses</h3>
          <p className="card-value">{totalDoses.toLocaleString('pt-BR')}</p>
          <p className="card-label">doses aplicadas</p>
        </div>
        <div className="info-card">
          <h3>Total de Casos</h3>
          <p className="card-value">{totalCasos.toLocaleString('pt-BR')}</p>
          <p className="card-label">casos confirmados</p>
        </div>
        <div className="info-card">
          <h3>Total de Óbitos</h3>
          <p className="card-value">{totalObitos.toLocaleString('pt-BR')}</p>
          <p className="card-label">óbitos registrados</p>
        </div>
        <div className="info-card">
          <h3>Taxa de Letalidade</h3>
          <p className="card-value">{mediaLetal.toFixed(2)}%</p>
          <p className="card-label">média nacional</p>
        </div>
      </div>

      <div className="charts-grid">
        {/* Gráfico 1: Total de Doses por Ano */}
        <div className="chart-card">
          <h2>Total de Doses Aplicadas por Ano</h2>
          <Plot
            data={[
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.doses_aplicadas),
                type: 'bar',
                marker: {
                  color: '#4CAF50',
                  opacity: 0.8
                },
                name: 'Doses Aplicadas'
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: { 
                title: { text: 'Ano' },
                fixedrange: true
              },
              yaxis: { 
                title: { text: 'Número de Doses' },
                fixedrange: false
              },
              showlegend: false
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 2: Evolução dos Tipos de Doses */}
        <div className="chart-card">
          <h2>Evolução dos Tipos de Doses</h2>
          <Plot
            data={[
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.primeira_dose),
                type: 'scatter',
                mode: 'lines+markers',
                name: '1ª Dose',
                line: { color: '#2196F3', width: 3 },
                marker: { size: 8 }
              },
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.segunda_dose),
                type: 'scatter',
                mode: 'lines+markers',
                name: '2ª Dose',
                line: { color: '#4CAF50', width: 3 },
                marker: { size: 8 }
              },
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.terceira_dose),
                type: 'scatter',
                mode: 'lines+markers',
                name: '3ª Dose',
                line: { color: '#FFC107', width: 3 },
                marker: { size: 8 }
              },
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.dose_reforco),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Reforço',
                line: { color: '#9C27B0', width: 3 },
                marker: { size: 8 }
              },
              {
                x: dadosVacinacao.map(d => d.ano),
                y: dadosVacinacao.map(d => d.dose_unica),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Dose Única',
                line: { color: '#FF5722', width: 3 },
                marker: { size: 8 }
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: { 
                title: { text: 'Ano' },
                fixedrange: true
              },
              yaxis: { 
                title: { text: 'Número de Doses' },
                fixedrange: false
              },
              legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
              }
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 3: Casos por Região */}
        <div className="chart-card">
          <h2>Distribuição de Casos por Região</h2>
          <Plot
            data={[
              {
                type: 'pie',
                labels: dadosRegioes.map(d => d.regiao),
                values: dadosRegioes.map(d => d.casos),
                hole: 0.4,
                marker: {
                  colors: ['#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#FF5722']
                }
              }
            ]}
            layout={{
              ...baseLayout,
              showlegend: true,
              legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
              }
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 4: Correlação Vacinação vs Mortalidade */}
        <div className="chart-card">
          <h2>Correlação: Vacinação vs Taxa de Mortalidade</h2>
          <Plot
            data={[
              {
                x: correlacaoData.map(d => d.doses_aplicadas),
                y: correlacaoData.map(d => d.taxa_mortalidade),
                mode: 'markers',
                type: 'scatter',
                text: correlacaoData.map(d => d.estado),
                marker: {
                  size: 12,
                  color: '#1976D2',
                  opacity: 0.7
                },
                hovertemplate: 
                  '<b>%{text}</b><br>' +
                  'Doses: %{x:,.0f}<br>' +
                  'Taxa Mortalidade: %{y:.2f}<br>' +
                  '<extra></extra>'
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: {
                title: { text: 'Doses Aplicadas' },
                fixedrange: true
              },
              yaxis: {
                title: { text: 'Taxa de Mortalidade' },
                fixedrange: true
              }
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 5: Top 10 Estados (Casos) */}
        <div className="chart-card">
          <h2>Top 10 Estados - Casos COVID-19</h2>
          <Plot
            data={[
              {
                x: rankingEstados.casos?.map(d => d.estado) || [],
                y: rankingEstados.casos?.map(d => d.casos) || [],
                type: 'bar',
                marker: {
                  color: '#2196F3',
                  opacity: 0.8
                }
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: {
                title: { text: 'Estado' },
                fixedrange: true
              },
              yaxis: {
                title: { text: 'Número de Casos' },
                fixedrange: false
              },
              showlegend: false
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 6: Top 10 Estados (Óbitos) */}
        <div className="chart-card">
          <h2>Top 10 Estados - Óbitos COVID-19</h2>
          <Plot
            data={[
              {
                x: rankingEstados.obitos?.map(d => d.estado) || [],
                y: rankingEstados.obitos?.map(d => d.obitos) || [],
                type: 'bar',
                marker: {
                  color: '#FF5722',
                  opacity: 0.8
                }
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: {
                title: { text: 'Estado' },
                fixedrange: true
              },
              yaxis: {
                title: { text: 'Número de Óbitos' },
                fixedrange: false
              },
              showlegend: false
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 7: Incidência por Estado */}
        <div className="chart-card">
          <h2>Top 10 Estados - Incidência COVID-19</h2>
          <Plot
            data={[
              {
                x: rankingEstados.incidencia?.map(d => d.estado) || [],
                y: rankingEstados.incidencia?.map(d => d.incidencia) || [],
                type: 'bar',
                marker: {
                  color: '#4CAF50',
                  opacity: 0.8
                }
              }
            ]}
            layout={{
              ...baseLayout,
              xaxis: {
                title: { text: 'Estado' },
                fixedrange: true
              },
              yaxis: {
                title: { text: 'Incidência (por 100 mil hab.)' },
                fixedrange: false
              },
              showlegend: false
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>

        {/* Gráfico 8: Distribuição de Óbitos por Região */}
        <div className="chart-card">
          <h2>Distribuição de Óbitos por Região</h2>
          <Plot
            data={[
              {
                type: 'pie',
                labels: dadosRegioes.map(d => d.regiao),
                values: dadosRegioes.map(d => d.obitos),
                hole: 0.4,
                marker: {
                  colors: ['#FF5722', '#9C27B0', '#FFC107', '#4CAF50', '#2196F3']
                }
              }
            ]}
            layout={{
              ...baseLayout,
              showlegend: true,
              legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
              }
            }}
            config={plotConfig}
            className="plot-container"
          />
        </div>
      </div>

      <footer className="dashboard-footer">
        <p>Dados atualizados em tempo real</p>
      </footer>
    </div>
  );
}

export default App; 