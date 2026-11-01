from crewai.tools import tool
import joblib
import pandas as pd

# Carrega a Árvore de Decisão pré-treinada
decision_tree_model = joblib.load("modelo_arvore_decisao.pkl")

@tool("Executar Classificação por Árvore de Decisão")
def executar_arvore_decisao(idade: int, renda_anual: float, historico_credito: str) -> str:
    """Recebe dados do cliente e retorna a classe prevista pela Árvore de Decisão."""
    # Transforma entradas do agente para o formato exigido pelo Scikit-Learn
    dados_entrada = pd.DataFrame([{
        "idade": idade,
        "renda": renda_anual,
        "historico": 1 if historico_credito == "bom" else 0
    }])
    
    # Previsão determinística
    predicao = decision_tree_model.predict(dados_entrada)[0]
    probabilidade = decision_tree_model.predict_proba(dados_entrada).max()
    
    return f"Classe prevista: {predicao} (Confiança do modelo: {probabilidade:.2%})"