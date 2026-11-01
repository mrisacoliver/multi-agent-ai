import os
import joblib
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# PASSO 1: Treinar e Salvar a Árvore de Decisão (Scikit-Learn)
# =====================================================================
def preparar_modelo_arvore():
    # Dados fictícios de treino: [Idade, Renda Anual (k$), Score de Crédito (300-850)]
    X_train = pd.DataFrame([
        [22, 25.0, 500],
        [45, 80.0, 750],
        [30, 40.0, 620],
        [20, 18.0, 450],
        [50, 110.0, 800],
        [35, 55.0, 580]
    ], columns=["idade", "renda", "score"])
    
    # Risco: 0 = Baixo Risco (Aprovado), 1 = Alto Risco (Recusado)
    y_train = [1, 0, 0, 1, 0, 1]

    # Treinamento da Árvore de Decisão
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Salva o modelo treinado em disco
    joblib.dump(clf, "arvore_credito.pkl")
    print("-> Árvore de Decisão treinada e salva em 'arvore_credito.pkl'.")

preparar_modelo_arvore()

# =====================================================================
# PASSO 2: Criar a Tool do CrewAI para consultar a Árvore de Decisão
# =====================================================================
@tool("Consultar Árvore de Decisão de Crédito")
def avaliar_risco_arvore(idade: int, renda: float, score: int) -> str:
    """
    Recebe a idade, renda anual em milhares de dólares (ex: 45.0) e score de crédito (300-850)
    e executa o modelo determinístico de Árvore de Decisão para classificar o risco do cliente.
    """
    if not os.path.exists("arvore_credito.pkl"):
        return "Erro: Modelo de Árvore de Decisão não encontrado."
        
    modelo = joblib.load("arvore_credito.pkl")
    
    # Formata a entrada para o pandas
    dados_cliente = pd.DataFrame([[idade, renda, score]], columns=["idade", "renda", "score"])
    
    # Previsão
    predicao = modelo.predict(dados_cliente)[0]
    probabilidade = modelo.predict_proba(dados_cliente).max()
    
    status = "Alto Risco (Crédito Recusado)" if predicao == 1 else "Baixo Risco (Crédito Aprovado)"
    
    return f"Resultado da Árvore de Decisão: {status} com {probabilidade:.1%} de confiança."

# =====================================================================
# PASSO 3: Configurar os Agentes e Tarefas no CrewAI
# =====================================================================

# Dado não estruturado de entrada
email_solicitacao = """
Olá equipe financeiro, meu nome é Roberto da Silva. 
Tenho 21 anos e gostaria de solicitar um empréstimo. 
Atualmente ganho cerca de 20 mil dólares por ano e meu score de crédito no Serasa está em 480.
Aguardo o retorno!
"""

# Agente 1: Extrator de Dados Não Estruturados
agente_extrator = Agent(
    role="Extrator de Dados de Solicitações",
    goal="Analisar textos não estruturados e extrair com precisão os dados numéricos do cliente.",
    backstory="Você é um especialista em processamento de texto e conversão de linguagem natural em dados estruturados.",
    verbose=True
)

# Agente 2: Analista de Crédito (Usa a Árvore de Decisão)
agente_analista = Agent(
    role="Analista de Risco de Crédito",
    goal="Avaliar o risco do cliente executando a ferramenta de Árvore de Decisão e emitir o parecer final.",
    backstory="Você combina análise de regras matemáticas através de ferramentas preditivas com justificativa formal de crédito.",
    tools=[avaliar_risco_arvore],
    verbose=True
)

# Tarefa 1: Extração
tarefa_extracao = Task(
    description=f"Leia o seguinte e-mail e extraia a Idade (anos), Renda Anual (em milhares, ex: 20.0) e Score de Crédito:\n{email_solicitacao}",
    expected_output="Um resumo contendo: Idade, Renda (em milhares) e Score de crédito extraídos.",
    agent=agente_extrator
)

# Tarefa 2: Avaliação via Árvore
tarefa_analise = Task(
    description="Com base nos dados extraídos pelo extrator, execute a ferramenta 'Consultar Árvore de Decisão de Crédito' e elabore um parecer final justificando o resultado.",
    expected_output="Relatório final de aprovação/rejeição com base na resposta da Árvore de Decisão.",
    agent=agente_analista
)

# =====================================================================
# PASSO 4: Executar a Equipe (Crew)
# =====================================================================
equipe = Crew(
    agents=[agente_extrator, agente_analista],
    tasks=[tarefa_extracao, tarefa_analise],
    process=Process.sequential
)

if __name__ == "__main__":
    resultado = equipe.kickoff()
    print("\n================ PARECER FINAL ================")
    print(resultado)