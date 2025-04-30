import os
import subprocess
import time

def run_script(script_name):
    """Executa um script Python e controla o progresso"""
    print(f"\n{'='*50}")
    print(f"Executando {script_name}...")
    print(f"{'='*50}")
    
    try:
        # Executa o script
        result = subprocess.run(['python', script_name], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} executado com sucesso!")
        else:
            print(f"\n❌ Erro ao executar {script_name}. Código de retorno: {result.returncode}")
    except Exception as e:
        print(f"\n❌ Exceção ao executar {script_name}: {str(e)}")

def main():
    """Executa todos os scripts de análise de dados"""
    start_time = time.time()
    
    print("\n" + "*"*70)
    print("*" + " "*27 + "ANÁLISE COVID-19" + " "*27 + "*")
    print("*" + " "*13 + "ANÁLISE COMPLETA DE DADOS DA COVID-19 NO BRASIL" + " "*13 + "*")
    print("*" + " "*70 + "*")
    print("*" + " "*18 + "Este processo executará duas análises:" + " "*18 + "*")
    print("*" + " "*20 + "1. Análise geral da COVID-19" + " "*21 + "*")
    print("*" + " "*20 + "2. Comparação de vacinação por ano" + " "*16 + "*")
    print("*"*70)
    print("\nIniciando análise completa...\n")
    
    # Executa o script principal de análise
    run_script('main.py')
    
    # Executa o script de comparação de vacinação
    run_script('comparacao_vacinacao_anual.py')
    
    # Exibe resumo das visualizações geradas
    print("\n" + "="*70)
    print("RESUMO DE VISUALIZAÇÕES GERADAS".center(70))
    print("="*70)
    
    if os.path.exists('visualizacoes'):
        files = os.listdir('visualizacoes')
        
        # Agrupa por categorias
        covid_graphs = [f for f in files if any(x in f for x in ['casos', 'obitos', 'letalidade', 'correlacao', 'regiao', 'vacinacao_vs_mortalidade'])]
        predicao_graphs = [f for f in files if any(x in f for x in ['regressao', 'confusao'])]
        cluster_graphs = [f for f in files if any(x in f for x in ['kmeans', 'dbscan'])]
        vacinacao_graphs = [f for f in files if any(x in f for x in ['doses', 'evolucao', 'composicao', 'media', 'radar', 'percentual', 'variacao'])]
        
        print("\n📊 GRÁFICOS DE ANÁLISE COVID-19:")
        for i, graph in enumerate(covid_graphs, 1):
            print(f"   {i}. {graph}")
        
        print("\n📈 GRÁFICOS DE MODELAGEM PREDITIVA:")
        for i, graph in enumerate(predicao_graphs, 1):
            print(f"   {i}. {graph}")
        
        print("\n🔍 GRÁFICOS DE CLUSTERIZAÇÃO:")
        for i, graph in enumerate(cluster_graphs, 1):
            print(f"   {i}. {graph}")
            
        print("\n💉 GRÁFICOS DE COMPARAÇÃO DE VACINAÇÃO:")
        for i, graph in enumerate(vacinacao_graphs, 1):
            print(f"   {i}. {graph}")
            
        print(f"\nTotal de visualizações: {len(files)}")
    else:
        print("❌ Diretório de visualizações não encontrado!")
    
    # Exibe o tempo total de execução
    end_time = time.time()
    exec_time = end_time - start_time
    
    print("\n" + "*"*70)
    print(f"ANÁLISE COMPLETA FINALIZADA EM {exec_time:.2f} SEGUNDOS".center(70))
    print("*"*70)
    print(f"\nTodas as visualizações estão disponíveis no diretório 'visualizacoes/'")
    print("Para visualizar os gráficos interativos HTML, abra-os em um navegador web.")

if __name__ == "__main__":
    main() 