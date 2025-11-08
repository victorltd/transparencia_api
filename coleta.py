import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (se existir)
load_dotenv()

# Nome da variável de ambiente usada para a chave da API
ENV_API_KEY = 'PORTAL_API_KEY'

# Ler a chave da API a partir do ambiente
API_KEY = os.environ.get(ENV_API_KEY)


def obter_dados_api(codigo_orgao, data_inicial):
    """Busca contratos na API do Portal da Transparência.

    Retorna uma lista (possivelmente vazia) de dicionários.
    """
    url = "https://api.portaldatransparencia.gov.br/api-de-dados/contratos"

    if not API_KEY:
        raise SystemExit(
            f"Chave da API não encontrada. Defina a variável de ambiente '{ENV_API_KEY}' ou crie um arquivo .env com {ENV_API_KEY}=sua_chave"
        )

    params = {"codigoOrgao": codigo_orgao, "quantidade": 100, "dataInicial": data_inicial, "pagina": 1}
    headers = {"accept": "*/*", "chave-api-dados": API_KEY}

    dados_paginas = []

    while True:
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            dados_json = response.json()

            if not dados_json:
                break

            dados_paginas.extend(dados_json)
            params["pagina"] += 1

        except requests.exceptions.RequestException as e:
            # Informa o erro específico do órgão, mas continua a execução
            print(f"   Erro ao fazer a requisição para {codigo_orgao} (página {params['pagina']}): {e}")
            break

    return dados_paginas


def criar_dataframe(codigo_orgao="26230", data_inicial="01/01/2020"):
    dados_contratos = obter_dados_api(codigo_orgao, data_inicial)
    if dados_contratos:
        return pd.DataFrame(dados_contratos)
    return None # Retorna None se a lista 'dados_contratos' estiver vazia


def _safe_normalize_column(df, col):
    """Tenta normalizar uma coluna do tipo lista/dicionário. Se não for possível, retorna None."""
    if col not in df.columns:
        return None
    try:
        # Filtra valores que não são dicionários (ex: None, NaN) antes de normalizar
        valid_data = df[col].dropna().loc[df[col].apply(isinstance, args=(dict,))]
        if valid_data.empty:
             return None
        
        normalized = pd.json_normalize(valid_data)
        
        # Reindexa para bater com o índice original do DF e permitir a concatenação correta
        normalized = normalized.set_index(valid_data.index)
        
        # prefix to avoid collisions
        normalized.columns = [f"{col}.{c}" for c in normalized.columns]
        return normalized
    except Exception as e:
        print(f"Não foi possível normalizar coluna '{col}': {e}")
        return None


def processar_e_salvar(df, output_path="df_final_corrigido.csv"):
    if df is None or df.empty:
        print("DataFrame vazio ou None — nada para processar.")
        return False

    # renomeações iniciais seguras
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'id_inicial'})
    if 'objeto' in df.columns:
        df = df.rename(columns={'objeto': 'objeto_contrato'})

    # Normalizar colunas que podem ser dicionários
    partes = [df] # Começa com o DataFrame original
    for col in ['compra', 'fornecedor', 'unidadeGestora']:
        norm = _safe_normalize_column(df, col)
        if norm is not None:
            partes.append(norm)

    # Concat original + normalizados (se houver)
    # axis=1 garante que a junção seja "lado a lado", alinhada pelo índice
    if len(partes) > 1:
        df_concat = pd.concat(partes, axis=1)
    else:
        df_concat = df.copy()

    # Remover colunas originais com objetos se existirem
    cols_to_drop = [c for c in ['compra', 'unidadeGestora', 'fornecedor', 'id', 'objeto',
                               'orgaoVinculado.codigoSIAFI', 'orgaoVinculado.cnpj',
                               'orgaoVinculado.sigla', 'orgaoVinculado.nome', 'orgaoMaximo.codigo',
                               'orgaoMaximo.sigla', 'orgaoMaximo.nome', 'numeroProcesso',
                               'unidadeGestoraCompras', 'descricaoPoder', 'cpfFormatado',
                               'numeroInscricaoSocial', 'razaoSocialReceita', 'nomeFantasiaReceita', 'tipo']
                      if c in df_concat.columns]

    if cols_to_drop:
        df_concat = df_concat.drop(columns=cols_to_drop)

    # Remover colunas duplicadas mantendo a primeira ocorrência
    df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]

    # renomear colunas comuns se existirem
    if 'codigo' in df_concat.columns:
        df_concat = df_concat.rename(columns={'codigo': 'uge'})
    if 'nome' in df_concat.columns:
        df_concat = df_concat.rename(columns={'nome': 'nome_uge'})
    
    # Renomeia colunas específicas do fornecedor que podem ter vindo da normalização
    # (Evita o bug onde 'nome' (do fornecedor) é renomeado para 'nome_uge')
    if 'fornecedor.nome' in df_concat.columns:
         df_concat = df_concat.rename(columns={'fornecedor.nome': 'nome_fornecedor'})
    if 'fornecedor.tipo' in df_concat.columns:
         df_concat = df_concat.rename(columns={'fornecedor.tipo': 'tipo_fornecedor'})

    # Remover duplicatas — se colunas específicas existirem, use-as; senão, remova duplicatas completas
    subset_cols = [c for c in ['valorInicialCompra', 'valorFinalCompra'] if c in df_concat.columns]
    if subset_cols:
        df_final_corrigido = df_concat.drop_duplicates(subset=subset_cols, keep='first')
    else:
        df_final_corrigido = df_concat.drop_duplicates(keep='first')

    # Garantir diretório de saída
    out_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Salvar CSV com separador ; e codificação latin1
    df_final_corrigido.to_csv(output_path, sep=';', index=False, encoding='latin1')
    print(f"DataFrame salvo com sucesso em '{output_path}' com codificação latin1")
    return True


# ==============================================================================
# FUNÇÃO 'MAIN' (MODIFICADA PARA BUSCAR MÚLTIPLOS ÓRGÃOS)
# ==============================================================================
def main():
    
    # 1. LISTA DE ÓRGÃOS (Conforme sua solicitação)
    LISTA_ORGAOS_NE = [
        # Universidades
        "26231", # UNIVERSIDADE FEDERAL DE ALAGOAS
        "26232", # UNIVERSIDADE FEDERAL DA BAHIA
        "26447", # UNIVERSIDADE FEDERAL DO OESTE DA BAHIA
        "26351", # UNIVERSIDADE FEDERAL DO RECONCAVO DA BAHIA
        "26450", # UNIVERSIDADE FEDERAL DO SUL DA BAHIA
        "26233", # UNIVERSIDADE FEDERAL DO CEARA
        "26449", # UNIVERSIDADE FEDERAL DO CARIRI
        "26442", # UNIV DA INTEG. INTERN DA LUSOF AFRO-BRASILEIRA
        "26272", # FUNDACAO UNIVERSIDADE DO MARANHAO
        "26240", # UNIVERSIDADE FEDERAL DA PARAIBA
        "26252", # UNIVERSIDADE FEDERAL DE CAMPINA GRANDE
        "26242", # UNIVERSIDADE FEDERAL DE PERNAMBUCO
        "26248", # UNIVERSIDADE FEDERAL RURAL DE PERNAMBUCO
        "26456", # UNIVERSIDADE FEDERAL DO AGRESTE DE PERNAMBUCO
        "26230", # FUND UNIVERSIDADE FEDERAL VALE SAO FRANCISCO
        "26279", # FUNDACAO UNIVERSIDADE FEDERAL DO PIAUI
        "26455", # UNIVERSIDADE FEDERAL DO DELTA DO PARNAIBA
        "26243", # UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE
        "26264", # UNIVERSIDADE FEDERAL RURAL DO SEMI-ARIDO/RN
        "26281", # FUNDACAO UNIVERSIDADE FEDERAL DE SERGIPE
        # Institutos Federais
        "26402", # INST FED DE EDUC CIENCE TEC.DE ALAGOAS
        "26427", # INST FED DE EDUC CIENCE TEC.DA BAHIA
        "26404", # INST FED. DE EDUC CIENCE TEC BAIANO
        "26405", # INST FED DE EDUC.,CIENCE TEC. DO CEARA
        "26408", # INST FED DE EDUC CIENCE TEC DO MARANHAO
        "26417", # INST FED DE EDUC.,CIENCE TEC. DA PARAIBA
        "26418", # INST FED DE EDUC CIENCE TEC DE PERNAMBUCO
        "26430", # INST FED DE ED CIENCE TEC. DO S.PERNAMBUCANO
        "26431", # INST FED DE EDUC CIENCE TEC DO PIAUI
        "26435", # INST FED.DE EDUC. CIENCE TEC.DO RN
        "26423", # INST FED DE EDUCCIENCE TEC DE SERGIPE
    ]

    # 2. DATA INICIAL DA BUSCA
    DATA_INICIAL_BUSCA = "01/01/2023" # Você pode alterar esta data
    
    # 3. NOME DO ARQUIVO DE SAÍDA
    NOME_ARQUIVO_SAIDA = 'df_final_corrigido.csv'

    print(f"Iniciando coleta para {len(LISTA_ORGAOS_NE)} órgãos (Período: {DATA_INICIAL_BUSCA} em diante)...")

    # Lista para guardar os DataFrames de cada órgão
    todos_dfs = []

    # 4. LOOP DE COLETA
    for i, codigo_orgao in enumerate(LISTA_ORGAOS_NE, 1):
        print(f"Buscando órgão {i}/{len(LISTA_ORGAOS_NE)}: {codigo_orgao}...")
        
        # Chama sua função original para buscar os dados de um órgão
        df_orgao = criar_dataframe(
            codigo_orgao=codigo_orgao, 
            data_inicial=DATA_INICIAL_BUSCA
        )
        
        if df_orgao is not None and not df_orgao.empty:
            print(f"   -> Sucesso! {len(df_orgao)} registros encontrados para {codigo_orgao}.")
            todos_dfs.append(df_orgao)
        else:
            print(f"   -> Nenhum dado retornado para o órgão {codigo_orgao}.")

    # 5. CONSOLIDAÇÃO
    if not todos_dfs:
        print("\nNenhum dado foi coletado de nenhum órgão. Encerrando.")
        return 1 # Retorna código de erro

    print(f"\nConcatenando dados de todos os {len(todos_dfs)} órgãos...")
    df_completo = pd.concat(todos_dfs, ignore_index=True)
    
    print(f"Total de {len(df_completo)} registros brutos coletados. Iniciando processamento...")

    # 6. PROCESSAMENTO E SALVAMENTO
    # Chama sua função original para processar o DataFrame completo
    success = processar_e_salvar(df_completo, output_path=NOME_ARQUIVO_SAIDA)
    
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())