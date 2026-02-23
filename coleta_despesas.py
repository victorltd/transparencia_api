import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Carrega chaves
load_dotenv()
ENV_API_KEY = 'PORTAL_API_KEY'
API_KEY = os.environ.get(ENV_API_KEY)

# Lista de Órgãos (Reutilizando sua lista do NE)
LISTA_ORGAOS_NE = [
    "26231", "26232", "26447", "26351", "26450", "26233", "26449", "26442", "26272", "26240",
    "26252", "26242", "26248", "26456", "26230", "26279", "26455", "26243", "26264", "26281",
    "26402", "26427", "26404", "26405", "26408", "26417", "26418", "26430", "26431", "26435", "26423"
]

def obter_despesas_api(ano, codigo_orgao):
    """Busca despesas no endpoint /despesas/por-orgao."""
    url = "https://api.portaldatransparencia.gov.br/api-de-dados/despesas/por-orgao"
    
    if not API_KEY:
        print("Erro: API_KEY não configurada.")
        return []

    # Parâmetros conforme documentação: ano, orgao e pagina
    params = {
        "ano": ano,
        "orgao": codigo_orgao,
        "pagina": 1
    }
    headers = {"accept": "*/*", "chave-api-dados": API_KEY}
    dados_acumulados = []

    while True:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            dados_json = response.json()

            if not dados_json:
                break

            dados_acumulados.extend(dados_json)
            params["pagina"] += 1

        except requests.exceptions.RequestException as e:
            print(f"   Erro nas despesas do órgão {codigo_orgao} em {ano}: {e}")
            break

    return dados_acumulados

def tratar_e_limpar_despesas(df):
    """Converte valores monetários de string para float e limpa duplicatas."""
    if df is None or df.empty:
        return None

    # Função para converter "1.234,56" em 1234.56
    def converter_moeda(valor):
        if isinstance(valor, str):
            # Remove ponto de milhar e troca vírgula por ponto decimal
            return float(valor.replace('.', '').replace(',', '.'))
        return valor

    # Colunas financeiras identificadas no endpoint
    cols_financeiras = ['empenhado', 'liquidado', 'pago']
    
    for col in cols_financeiras:
        if col in df.columns:
            df[col] = df[col].apply(converter_moeda)

    # Remove duplicatas
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df

def main_despesas():
    # Defina os anos que deseja coletar (ex: 2023, 2024 e 2025)
    ANOS_BUSCA = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    NOME_ARQUIVO_DESPESAS = 'df_despesas_nordeste.csv'
    
    print(f"Iniciando coleta de DESPESAS para {len(LISTA_ORGAOS_NE)} órgãos...")
    todos_dados_despesas = []

    for ano in ANOS_BUSCA:
        print(f"\n--- Coletando Ano: {ano} ---")
        for i, orgao in enumerate(LISTA_ORGAOS_NE, 1):
            print(f"[{i}/{len(LISTA_ORGAOS_NE)}] Órgão: {orgao}...", end="\r")
            dados = obter_despesas_api(ano, orgao)
            if dados:
                todos_dados_despesas.extend(dados)

    if not todos_dados_despesas:
        print("\nNenhuma despesa encontrada.")
        return

    df_bruto = pd.DataFrame(todos_dados_despesas)
    df_final = tratar_e_limpar_despesas(df_bruto)

    # Salva o CSV separado para a página de despesas
    df_final.to_csv(NOME_ARQUIVO_DESPESAS, sep=';', index=False, encoding='latin1')
    print(f"\n\nSucesso! Arquivo '{NOME_ARQUIVO_DESPESAS}' gerado com {len(df_final)} registros.")

if __name__ == "__main__":
    main_despesas()