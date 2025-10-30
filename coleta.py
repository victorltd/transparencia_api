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
            print("Erro ao fazer a requisição:", e)
            break

    return dados_paginas


def criar_dataframe(codigo_orgao="26230", data_inicial="01/01/2025"):
    dados_contratos = obter_dados_api(codigo_orgao, data_inicial)
    if dados_contratos:
        return pd.DataFrame(dados_contratos)
    return None


def _safe_normalize_column(df, col):
    """Tenta normalizar uma coluna do tipo lista/dicionário. Se não for possível, retorna None."""
    if col not in df.columns:
        return None
    try:
        normalized = pd.json_normalize(df[col])
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
    partes = []
    for col in ['compra', 'fornecedor', 'unidadeGestora']:
        norm = _safe_normalize_column(df, col)
        if norm is not None:
            partes.append(norm)

    # Concat original + normalizados (se houver)
    if partes:
        df_concat = pd.concat([df] + partes, axis=1)
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


def main():
    df = criar_dataframe()
    success = processar_e_salvar(df, output_path='df_final_corrigido.csv')
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())