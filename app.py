"""
======================================================================
SCRIPT: app.py (O Dashboard Streamlit) - VERSÃO AJUSTADA
======================================================================
OBJETIVO:
Este script lê o SEU arquivo 'dados_contratos.csv' (com separador ';'
e colunas em CamelCase, ex: 'dataAssinatura') e exibe as
visualizações e análises para o usuário.

COMO EXECUTAR:
1. Certifique-se de que seu arquivo 'dados_contratos.csv' está na
   mesma pasta que este script 'app.py'.
2. No seu terminal, execute:
   streamlit run app.py
"""

# --- BLOCO 1: IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Paleta de cores usada em todos os gráficos (consistente)
COLOR_PALETTE = px.colors.qualitative.Vivid

# --- BLOCO 2: CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Contratos - UNIVASF",
    page_icon="📊",
    layout="wide"
)

# --- BLOCO 3: FUNÇÃO DE CARREGAMENTO DE DADOS ---
# AJUSTADA para ler seu CSV específico.

@st.cache_data
def carregar_dados(caminho_arquivo="df_final_corrigido.csv"):
    """
    Lê o arquivo CSV, trata erros, e converte os tipos de dados
    corretamente (datas e valores com vírgula).
    """
    try:
        # Tenta ler com UTF-8 primeiro. Se houver erro de decodificação, tenta latin1
        try:
            df = pd.read_csv(caminho_arquivo, sep=';')
        except UnicodeDecodeError:
            st.warning(f"Falha ao decodificar '{caminho_arquivo}' como UTF-8 — tentando latin1 (ISO-8859-1).")
            df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin1')
        except Exception:
            # re-raise para ser pego pelo outer except
            raise
        
        # AJUSTADO: Nomes das colunas de data conforme sua imagem
        colunas_data = ['dataAssinatura', 'dataInicioVigencia', 'dataFimVigencia']
        for col in colunas_data:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # AJUSTADO: Nomes das colunas de valor conforme sua imagem
        colunas_valor = ['valorInicialCompra', 'valorFinalCompra']
        for col in colunas_valor:
            if col in df.columns:
                # AJUSTE CRÍTICO: Trata o formato "1400,00" (string com vírgula)
                if pd.api.types.is_string_dtype(df[col]):
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace('.', '', regex=False)  # Remove separador de milhar (ex: 1.000)
                        .str.replace(',', '.', regex=False)  # Troca vírgula decimal por ponto
                    )
                
                # Converte para numérico, preenchendo o que der erro (ex: nulo) com 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        print("Dados carregados e processados com sucesso.")
        return df

    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        st.info("Por favor, certifique-se de que o CSV está na pasta correta.")
        return pd.DataFrame() # Retorna um DataFrame vazio
    except Exception as e:
        st.error(f"Erro inesperado ao ler o arquivo: {e}")
        return pd.DataFrame()

# --- BLOCO 4: FUNÇÕES DE APOIO (Helpers) ---

def formatar_valor(valor):
    """
    Formata um número em um texto amigável (ex: 1.5 M, 250 K)
    """
    if valor >= 1_000_000:
        return f"R$ {valor/1_000_000:.1f} M"
    if valor >= 1_000:
        return f"R$ {valor/1_000:.1f} K"
    return f"R$ {valor:.2f}"


def gerar_nuvem_palavras(df):
    """
    Gera e exibe uma nuvem de palavras a partir da coluna 'objeto_contrato'.
    """
    if 'objeto_contrato' not in df.columns:
        st.info("Coluna 'objeto_contrato' ausente — não é possível gerar a nuvem de palavras.")
        return

    text = " ".join(str(objeto) for objeto in df['objeto_contrato'] if pd.notna(objeto))
    if not text.strip():
        st.info("Nenhum texto de objeto de contrato disponível para gerar a nuvem de palavras.")
        return

    # stopwords em português com termos de domínio
    stopwords_pt = set([
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
        "uma", "os", "no", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
        "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
        "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
        "ela", "entre", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas",
        "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem",
        "seus", "minha", "minhas", "às", "qual", "serviços", "contrato", "prestação",
        "empresa", "fornecimento", "objeto", "univasf", "mediante", "conforme", "termo", "condições estabelecidas", "presente instrumento,"
        "contratação", "referência", "aquisição", "constitui", "campi", "incluindo", "edital destinado", "anexo presente","demais caracteristicas",
        "empresa", "fornecimento", "objeto", "univasf", "mediante", "conforme", "edital", "presente", "anexo", "demais", "instrumento","quantitativos",
        "instrumento","especificações", "condições", "destinado", "estabelecidos", "contida", "destinados"
    ])

    wc = WordCloud(
        stopwords=stopwords_pt,
        background_color="white",
        width=900,
        height=400,
        colormap='viridis'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# Helper para detectar coluna de fornecedor (nome preferencial, fallback para CNPJ)
def detect_fornecedor_column(df):
    """
    Tenta localizar uma coluna que represente o fornecedor.
    Preferência por colunas de nome (razao social / nome), mas aceita
    colunas de CNPJ como 'cnpjFormatado' quando nomes não existem.
    Retorna o nome da coluna ou None.
    """
    name_candidates = ['nome_fornecedor', 'nomeFornecedor', 'fornecedor.nome', 'fornecedorNome', 'fornecedor.razaoSocialReceita', 'fornecedor.nomeFantasiaReceita', 'razao_social', 'razaoSocial']
    for c in name_candidates:
        if c in df.columns:
            return c
        # case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col

    # fallback: colunas de CNPJ que provavelmente identificam o fornecedor
    cnpj_candidates = ['cnpjFormatado', 'fornecedor.cnpj', 'cnpj']
    for c in cnpj_candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col

    # última tentativa: alguma coluna contendo 'fornecedor' no nome
    for col in df.columns:
        if 'fornecedor' in col.lower():
            return col

    return None


def find_supplier_name_column(df):
    """
    Tenta identificar uma coluna que contenha o nome legível do fornecedor
    (razão social / nome fantasia) mesmo que a coluna tenha um nome inesperado
    como 'nome_uge'. Retorna o nome da coluna ou None.
    Heurística simples baseada em amostra de valores: prefere colunas com
    maioria de entradas alfabéticas e com palavras-chave típicas de empresas
    (LTDA, S/A, FUNDACAO, EIRELI, SERVICOS, TECNOLOGIA, etc.).
    """
    keywords = ['ltda', 'ltda', 's/a', 'fundacao', 'fundação', 'fundacao', 'servicos', 'serviços', 'tecnologia', 'comercio', 'empresa', 'assessoria', 'assessoria', 'industria', 'indústria', 'instituto']
    best_col = None
    best_score = 0.0
    for col in df.columns:
        # skip obvious numeric/date/value columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        # fraction of sample entries containing letters
        frac_alpha = (sample.str.contains('[A-Za-zÀ-ÿ]', regex=True)).mean()
        # fraction that are purely numeric
        frac_numeric = (sample.str.match(r'^\d+$', na=False)).mean()
        # keyword presence
        kw_hits = sum(sample.str.lower().str.contains(kw, na=False).any() for kw in keywords)

        # score combines alphabetic fraction and keyword hits, penalize numeric heavy columns
        score = frac_alpha - frac_numeric * 0.5 + (0.1 * kw_hits)
        if score > best_score and frac_alpha > 0.5:
            best_score = score
            best_col = col

    # Heuristic: if we found a reasonable candidate, return it
    return best_col

# --- BLOCO 5: LAYOUT DA INTERFACE (Sidebar e Filtros) ---

st.sidebar.title("Portal da Transparência - UNIVASF")

# 1. Navegação entre páginas
pagina_selecionada = st.sidebar.radio(
    "Navegação",
    ["Visão Geral (Contratos)", "Licitações (em breve)", "Despesas (em breve)"],
    key="nav_principal"
)

st.sidebar.markdown("---") # Linha divisória

# Carrega os dados
df_contratos = carregar_dados()

# Só exibe filtros e o app se os dados foram carregados
if not df_contratos.empty:

    # 2. Filtros Globais (só aparecem se os dados existirem)
    st.sidebar.header("Filtros da Página")
    
    # AJUSTADO: Usa 'dataAssinatura'
    data_min_contrato = df_contratos['dataAssinatura'].min().date()
    data_max_contrato = df_contratos['dataAssinatura'].max().date()

    data_inicio = st.sidebar.date_input(
        "Data de Assinatura (Início)",
        value=data_min_contrato,
        min_value=data_min_contrato,
        max_value=data_max_contrato,
        key="filtro_data_inicio"
    )

    data_fim = st.sidebar.date_input(
        "Data de Assinatura (Fim)",
        value=data_max_contrato,
        min_value=data_min_contrato,
        max_value=data_max_contrato,
        key="filtro_data_fim"
    )

    # AJUSTADO: Usa 'nome_uge' (conforme sua imagem)
    # Detecta coluna entre candidatos (helper usado em várias partes do app)
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
            if c.lower() in (col.lower() for col in df.columns):
                # Retorna a coluna com case original
                for col in df.columns:
                    if col.lower() == c.lower():
                        return col
        return None

    # Unidade Gestora: usar explicitamente a coluna 'unidadeGestora.orgaoVinculado.sigla'
    # (com alguns nomes alternativos como fallback). Removemos heurísticas complexas
    # para obedecer ao requisito de usar a coluna específica.
    ug_col = _find_col(df_contratos, ['unidadeGestora.orgaoVinculado.sigla', 'orgaovinculado.sigla', 'orgaoVinculado.sigla', 'nome_uge', 'uge'])
    if ug_col and ug_col in df_contratos.columns:
        # valores únicos (limpos) — se houver mais de um, ofereça filtro na sidebar
        ug_values = (df_contratos[ug_col].astype(str).str.strip().replace({'nan': None}).dropna().unique().tolist())
        if len(ug_values) <= 1:
            st.sidebar.info("Unidade Gestora detectada — apenas 1 UG presente no dataset; filtro desabilitado.")
            ug_selected = None
        else:
            ug_values_sorted = sorted(ug_values)
            ug_options = ['Todos'] + ug_values_sorted
            ug_selected = st.sidebar.selectbox("Unidade Gestora", options=ug_options, index=0, key="filtro_ug")
    else:
        ug_col = None
        ug_selected = None
        st.sidebar.info("Unidade Gestora não detectada — filtros por UG desabilitados.")

    # --- BLOCO 6: LÓGICA DE FILTRAGEM ---
    data_inicio_ts = pd.to_datetime(data_inicio)
    data_fim_ts = pd.to_datetime(data_fim)

    # AJUSTADO: Filtra por 'dataAssinatura' e por Unidade Gestora quando selecionada
    df_filtrado = df_contratos[
        (df_contratos['dataAssinatura'] >= data_inicio_ts) &
        (df_contratos['dataAssinatura'] <= data_fim_ts)
    ]
    if ug_col and ug_selected and ug_selected != 'Todos':
        # aplica filtro por UG (comparação de strings limpadas)
        df_filtrado = df_filtrado[df_filtrado[ug_col].astype(str).str.strip() == str(ug_selected).strip()]

    # Não filtramos por Unidade Gestora neste dataset (apenas uma UG presente)

# --- BLOCO 7: PÁGINA PRINCIPAL (Gráficos e KPIs) ---

if pagina_selecionada == "Visão Geral (Contratos)":

    if 'df_filtrado' not in locals() or df_filtrado.empty:
        st.warning("Nenhum dado encontrado. Verifique o arquivo CSV ou os filtros selecionados.")
    else:
        st.title("📊 Painel de Contratos da UNIVASF")
        st.markdown(f"Dados de {data_inicio.strftime('%d/%m/%Y')} até {data_fim.strftime('%d/%m/%Y')}")

        # --- LINHA 1: KPIs (Grandes Números) ---
        st.markdown("### Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)
        
        # AJUSTADO: Usa 'valorInicialCompra'
        valor_total = df_filtrado['valorInicialCompra'].sum()
        col1.metric("Valor Total Contratado", formatar_valor(valor_total))
        
        num_contratos = len(df_filtrado)
        col2.metric("Total de Contratos Firmados", f"{num_contratos}")
        
        hoje = pd.to_datetime(datetime.now().date())
        # AJUSTADO: Usa 'dataFimVigencia' e 'dataInicioVigencia'
        ativos = df_filtrado[
            (df_filtrado['dataFimVigencia'] >= hoje) & 
            (df_filtrado['dataInicioVigencia'] <= hoje)
        ]
        col3.metric("Contratos Ativos Hoje", f"{len(ativos)}")
        
        # AJUSTADO: Usa 'valorFinalCompra' e 'valorInicialCompra'
        aditivos = (df_filtrado['valorFinalCompra'] - df_filtrado['valorInicialCompra'])
        total_aditivos = aditivos[aditivos > 0].sum()
        col4.metric("Total em Aditivos", formatar_valor(total_aditivos))

        st.markdown("---")

        # --- BLOCO: Gasto por Unidade Gestora e Aditivos por UG (linha inteira abaixo dos KPIs)
        st.markdown("### Análises por Unidade Gestora")
        ug_left, ug_right = st.columns(2)

        # Determina coluna UG preferida
        ug_for_plot = ug_col if (ug_col and ug_col in df_filtrado.columns) else _find_col(df_contratos, ['unidadeGestora.orgaoVinculado.sigla', 'orgaovinculado.sigla', 'orgaoVinculado.sigla', 'nome_uge', 'uge'])

        with ug_left:
            st.markdown("#### Gasto por Unidade Gestora (Top 10)")
            if ug_for_plot and ug_for_plot in df_filtrado.columns and 'valorInicialCompra' in df_filtrado.columns:
                df_ug = df_filtrado.copy()
                df_ug['ug_label'] = df_ug[ug_for_plot].astype(str).str.strip().fillna('--N/A--')
                df_ug_sum = df_ug.groupby('ug_label')['valorInicialCompra'].sum().reset_index()
                if not df_ug_sum.empty:
                    top_ug = df_ug_sum.nlargest(10, 'valorInicialCompra').sort_values('valorInicialCompra')
                    fig_ug = px.bar(
                        top_ug,
                        x='valorInicialCompra',
                        y='ug_label',
                        orientation='h',
                        title='Top 10 Unidade Gestora por Valor Contratado',
                        labels={'valorInicialCompra': 'Valor Total (R$)', 'ug_label': 'Unidade Gestora'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    st.plotly_chart(fig_ug, use_container_width=True, key='ug_gasto_top10_main')
                else:
                    st.info("Não há valores por Unidade Gestora para o período selecionado.")
            else:
                st.info("Unidade Gestora não detectada ou coluna de valor ausente — pulando análise de UG.")

        with ug_right:
            st.markdown("#### Total em Aditivos por Unidade Gestora (Top 10)")
            if 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
                if ug_for_plot and ug_for_plot in df_filtrado.columns:
                    df_ad_ug = df_filtrado.copy()
                    df_ad_ug['aditivo'] = df_ad_ug['valorFinalCompra'] - df_ad_ug['valorInicialCompra']
                    df_ad_pos = df_ad_ug[df_ad_ug['aditivo'] > 0].copy()
                    if df_ad_pos.empty:
                        st.info('Nenhum aditivo positivo encontrado no período selecionado para as Unidades Gestoras.')
                    else:
                        df_ad_pos['ug_label'] = df_ad_pos[ug_for_plot].astype(str).str.strip().fillna('--N/A--')
                        sum_ad_ug = df_ad_pos.groupby('ug_label')['aditivo'].sum().reset_index()
                        top_ad_ug = sum_ad_ug.nlargest(10, 'aditivo').sort_values('aditivo')
                        fig_ad_ug = px.bar(
                            top_ad_ug,
                            x='aditivo',
                            y='ug_label',
                            orientation='h',
                            title='Top 10 Unidades Gestoras por Total em Aditivos (R$)',
                            labels={'aditivo': 'Total Aditivos (R$)', 'ug_label': 'Unidade Gestora'},
                            color_discrete_sequence=COLOR_PALETTE
                        )
                        st.plotly_chart(fig_ad_ug, use_container_width=True, key='ug_aditivos_top10_main')
                else:
                    st.info('Coluna de Unidade Gestora não encontrada para agrupar aditivos.')
            else:
                st.info('Colunas de valor ausentes — não é possível calcular aditivo por UG.')

        # --- BLOCO RÁPIDO: Top-N share, Anomalias em Aditivos (IQR), Resumo por Modalidade ---
        st.markdown("### Análises Rápidas: Concentração, Anomalias e Resumo por Modalidade")
        # configurações rápidas via sidebar
        pareto_top_n = st.sidebar.number_input("Top N (Pareto)", min_value=5, max_value=100, value=10, step=5, key="pareto_top_n")

        # 1) Top-N share (Pareto)
        # Detecta coluna de nome legível do fornecedor (quando disponível) e
        # coluna identificadora (CNPJ) como fallback.
        fornecedor_name_col = find_supplier_name_column(df_contratos)
        fornecedor_id_col = detect_fornecedor_column(df_contratos)

        # preferir nome legível para agrupamento/rotulagem, senão usar id (CNPJ)
        group_col = None
        if fornecedor_name_col and fornecedor_name_col in df_filtrado.columns:
            group_col = fornecedor_name_col
        elif fornecedor_id_col and fornecedor_id_col in df_filtrado.columns:
            group_col = fornecedor_id_col

        if group_col and 'valorInicialCompra' in df_filtrado.columns:
            df_sum = (
                df_filtrado.groupby(group_col)['valorInicialCompra']
                .sum()
                .reset_index(name='valor')
                .sort_values('valor', ascending=False)
            )
            total_val = df_sum['valor'].sum() if not df_sum.empty else 0
            if total_val > 0:
                df_sum['share'] = df_sum['valor'] / total_val
                df_sum['cum_share'] = df_sum['share'].cumsum()

                top_df = df_sum.head(int(pareto_top_n)).copy()
                top_df_display = top_df.copy()
                top_df_display['valor'] = top_df_display['valor'].map(lambda v: f"R$ {v:,.2f}")
                top_df_display['share'] = top_df_display['share'].map(lambda v: f"{v*100:.1f}%")
                st.markdown("#### Participação dos principais fornecedores (Top N)")
                fig_pareto = px.bar(
                    top_df,
                    x=group_col,
                    y='valor',
                    title=f'Top {int(pareto_top_n)} Fornecedores por Valor',
                    labels={group_col: 'Fornecedor', 'valor': 'Valor (R$)'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_pareto, use_container_width=True, key='pareto_chart')

                # Mostra KPI resumidos para Top1/Top5/Top10
                k1 = df_sum.head(1)['share'].sum() if len(df_sum) >= 1 else 0
                k5 = df_sum.head(5)['share'].sum() if len(df_sum) >= 5 else df_sum['share'].sum()
                k10 = df_sum.head(10)['share'].sum() if len(df_sum) >= 10 else df_sum['share'].sum()
                kcol1, kcol2, kcol3 = st.columns(3)
                kcol1.metric("Top-1 share", f"{k1*100:.1f}%")
                kcol2.metric("Top-5 share", f"{k5*100:.1f}%")
                kcol3.metric("Top-10 share", f"{k10*100:.1f}%")
            else:
                st.info("Sem valores suficientes para calcular participação de fornecedores.")
        else:
            st.info("Coluna de fornecedor ou valor ausente — pulando análise Pareto.")

        st.markdown("---")

        # 2) Anomalias em aditivos (IQR simples)
        st.markdown("#### Anomalias em Aditivos (detecção IQR)")
        if 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
            df_anom = df_filtrado.copy()
            df_anom['aditivo'] = df_anom['valorFinalCompra'] - df_anom['valorInicialCompra']
            # evita divisão por zero
            df_anom = df_anom[df_anom['valorInicialCompra'] > 0]
            df_anom['aditivo_pct'] = df_anom['aditivo'] / df_anom['valorInicialCompra']

            # mantém somente valores finitos
            df_pct = df_anom[df_anom['aditivo_pct'].notna() & (~df_anom['aditivo_pct'].isin([pd.NA]))]
            if not df_pct.empty:
                q1 = df_pct['aditivo_pct'].quantile(0.25)
                q3 = df_pct['aditivo_pct'].quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                lower = q1 - 1.5 * iqr

                anomalies = df_pct[(df_pct['aditivo_pct'] > upper) | (df_pct['aditivo_pct'] < lower)].copy()
                st.write(f"Detectadas {len(anomalies)} anomalias (IQR) em aditivos")

                # gráfico de dispersão aditivo_pct ao longo do tempo, destacando anomalias
                if 'dataAssinatura' in df_pct.columns:
                    df_pct['is_anom'] = df_pct.index.isin(anomalies.index)
                    fig_anom = px.scatter(
                        df_pct.sort_values('dataAssinatura'),
                        x='dataAssinatura',
                        y='aditivo_pct',
                        color='is_anom',
                        title='Aditivo (%) ao longo do tempo (anomalias destacadas)',
                        labels={'aditivo_pct': 'Aditivo (%)', 'dataAssinatura': 'Data de Assinatura'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    st.plotly_chart(fig_anom, use_container_width=True, key='anomalies_chart')

                # mostra tabela com top anomalias (ordenar por aditivo_pct desc)
                if not anomalies.empty:
                    disp_cols = []
                    if group_col and group_col in anomalies.columns:
                        disp_cols.append(group_col)
                    if 'numero' in anomalies.columns:
                        disp_cols.append('numero')
                    disp_cols += ['valorInicialCompra', 'valorFinalCompra', 'aditivo', 'aditivo_pct']

                    disp = anomalies.sort_values('aditivo_pct', ascending=False)[disp_cols].head(30).copy()
                    # formatar
                    if 'aditivo' in disp.columns:
                        disp['aditivo'] = disp['aditivo'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                    if 'valorInicialCompra' in disp.columns:
                        disp['valorInicialCompra'] = disp['valorInicialCompra'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                    if 'valorFinalCompra' in disp.columns:
                        disp['valorFinalCompra'] = disp['valorFinalCompra'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                    if 'aditivo_pct' in disp.columns:
                        disp['aditivo_pct'] = disp['aditivo_pct'].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "N/A")

                    with st.expander("Ver top anomalias (IQR)"):
                        disp = disp.rename(columns={group_col: 'fornecedor'}) if group_col and group_col in disp.columns else disp
                        st.dataframe(disp)
                else:
                    st.info('Nenhuma anomalia detectada pelo método IQR para o período selecionado.')
            else:
                st.info('Dados insuficientes para calcular aditivo percentual.')
        else:
            st.info("Colunas de valor ausentes — não é possível detectar anomalias de aditivos.")

        st.markdown("---")

        # 3) Resumo por Modalidade (rápido)
        st.markdown("#### Resumo por Modalidade de Compra")
        if 'modalidadeCompra' in df_filtrado.columns and 'valorInicialCompra' in df_filtrado.columns:
            df_mod = df_filtrado.copy()
            summary_mod = df_mod.groupby('modalidadeCompra').agg(
                contratos=('numero', 'nunique') if 'numero' in df_mod.columns else ('modalidadeCompra', 'count'),
                valor_total=('valorInicialCompra', 'sum'),
                valor_medio=('valorInicialCompra', 'mean')
            ).reset_index()

            # probabilidade de aditivo e duração média quando disponíveis
            if 'valorFinalCompra' in df_mod.columns:
                df_mod['aditivo'] = df_mod['valorFinalCompra'] - df_mod['valorInicialCompra']
                df_mod['teve_aditivo'] = df_mod['aditivo'] > 0
                aditivo_prob = df_mod.groupby('modalidadeCompra')['teve_aditivo'].mean().reset_index(name='prob_aditivo')
                summary_mod = summary_mod.merge(aditivo_prob, on='modalidadeCompra', how='left')
            else:
                summary_mod['prob_aditivo'] = pd.NA

            if 'dataInicioVigencia' in df_mod.columns and 'dataFimVigencia' in df_mod.columns:
                df_mod['duracao_dias'] = (pd.to_datetime(df_mod['dataFimVigencia']) - pd.to_datetime(df_mod['dataInicioVigencia'])).dt.days
                dur = df_mod.groupby('modalidadeCompra')['duracao_dias'].mean().reset_index(name='duracao_media_dias')
                summary_mod = summary_mod.merge(dur, on='modalidadeCompra', how='left')
            else:
                summary_mod['duracao_media_dias'] = pd.NA

            # formatar e exibir
            summary_display = summary_mod.copy()
            summary_display['valor_total'] = summary_display['valor_total'].map(lambda v: f"R$ {v:,.2f}")
            summary_display['valor_medio'] = summary_display['valor_medio'].map(lambda v: f"R$ {v:,.2f}")
            summary_display['prob_aditivo'] = summary_display['prob_aditivo'].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "N/A")
            summary_display['duracao_media_dias'] = summary_display['duracao_media_dias'].map(lambda v: f"{int(v)} d" if pd.notna(v) else "N/A")

            st.dataframe(summary_display)
        else:
            st.info("Colunas para resumo por modalidade ausentes — pulando resumo por modalidade.")

        st.markdown("---")


        # --- LINHA 2: Evolução Temporal ---
        st.markdown("### Evolução dos Gastos ao Longo do Tempo")
        
        # AJUSTADO: Usa 'dataAssinatura' e 'valorInicialCompra'
        df_temporal = df_filtrado.set_index('dataAssinatura')
        df_mensal = df_temporal.resample('ME')['valorInicialCompra'].sum().reset_index()
        
        fig_linha = px.line(
            df_mensal,
            x='dataAssinatura',
            y='valorInicialCompra',
            title='Valor Contratado por Mês',
            labels={'dataAssinatura': 'Mês', 'valorInicialCompra': 'Valor Total (R$)'},
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_linha.update_layout(hovermode="x unified")
        st.plotly_chart(fig_linha, use_container_width=True, key='temporal_line_chart')

       
        

        # --- MÉDIO: Fornecedores recorrentes com normalização por CNPJ ---
        st.markdown('### Fornecedores recorrentes (CNPJ-normalizado quando disponível)')
        # detecta cnpj e nome novamente
        cnpj_col = _find_col(df_contratos, ['fornecedor.cnpjFormatado', 'fornecedor.cnpj', 'fornecedor.cnpjFormatado'])
        forn_name_col = _find_col(df_contratos, ['fornecedor.nome', 'fornecedorNome', 'nome_fornecedor'])

        df_rec = df_filtrado.copy()
        if cnpj_col and cnpj_col in df_rec.columns:
            df_rec['supplier_id'] = df_rec[cnpj_col].astype(str).str.replace(r'\D', '', regex=True).replace({'': pd.NA})
        else:
            df_rec['supplier_id'] = pd.NA

        if df_rec['supplier_id'].isna().all() and forn_name_col and forn_name_col in df_rec.columns:
            df_rec['supplier_id'] = df_rec[forn_name_col].astype(str).str.strip().str.upper()

        if 'supplier_id' in df_rec.columns:
            # contar contratos por fornecedor
            rec_summary = df_rec.groupby('supplier_id').agg(
                contratos=('numero', 'nunique') if 'numero' in df_rec.columns else ('supplier_id', 'count'),
                valor_total=('valorInicialCompra', 'sum')
            ).reset_index()

            # fornecedores com >1 contrato
            rec_summary['recorrente'] = rec_summary['contratos'] > 1
            n_recorrentes = int(rec_summary['recorrente'].sum())
            st.markdown(f"Fornecedores com mais de 1 contrato no período: **{n_recorrentes}**")

            top_rec = rec_summary[rec_summary['recorrente']].nlargest(10, 'contratos')
            if not top_rec.empty:
                # tenta recuperar nome legível do fornecedor quando disponível
                # prefere a coluna de nome detectada (forn_name_col) se existir
                if 'forn_name_col' in locals() and forn_name_col and forn_name_col in df_rec.columns:
                    # cria mapeamento supplier_id -> nome (pega o primeiro nome não-nulo)
                    name_map = (
                        df_rec.dropna(subset=['supplier_id', forn_name_col])
                        .groupby('supplier_id')[forn_name_col]
                        .agg(lambda s: next((x for x in s if pd.notna(x) and str(x).strip()!=''), None))
                        .reset_index()
                        .rename(columns={forn_name_col: 'supplier_name'})
                    )
                    top_rec = top_rec.merge(name_map, on='supplier_id', how='left')
                    # coluna para exibição: prefira nome, senão supplier_id
                    top_rec['supplier_label'] = top_rec['supplier_name'].fillna(top_rec['supplier_id'])
                else:
                    top_rec['supplier_label'] = top_rec['supplier_id']

                fig_recorr = px.bar(
                    top_rec.sort_values('contratos'),
                    x='contratos',
                    y='supplier_label',
                    orientation='h',
                    title='Top Fornecedores recorrentes por # contratos',
                    labels={'supplier_label': 'Fornecedor', 'contratos': 'Número de Contratos'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_recorr, use_container_width=True, key='recorrentes_chart')

                with st.expander('Ver tabela de fornecedores recorrentes (Top 10)'):
                    disp = top_rec.copy()
                    disp['valor_total'] = disp['valor_total'].map(lambda v: f"R$ {v:,.2f}")
                    # mostra coluna legível 'fornecedor' com preferência por nome
                    if 'supplier_name' in disp.columns:
                        disp = disp.rename(columns={'supplier_name': 'fornecedor'})
                    else:
                        disp = disp.rename(columns={'supplier_label': 'fornecedor'})
                    # mantém contrato/valor/contagem de contratos
                    disp = disp[['fornecedor', 'contratos', 'valor_total']]
                    st.dataframe(disp)
            else:
                st.info('Nenhum fornecedor recorrente encontrado no período selecionado.')
        else:
            st.info('Não foi possível normalizar fornecedores — colunas CNPJ/nome ausentes.')

        # # --- LINHA 3: O que está sendo contratado? (Nuvem de Palavras) ---
        # st.markdown("### O que está sendo contratado? (Nuvem de Palavras)")
        # try:
        #     gerar_nuvem_palavras(df_filtrado)
        # except Exception as e:
        #     st.info(f"Erro ao gerar nuvem de palavras: {e}")
        # st.markdown("---")

        # --- BLOCO: Aditivos recorrentes por fornecedor (posicionado abaixo da evolução temporal)
        st.markdown("### Contratos com Aditivos Recorrentes por Fornecedor")
        # Detecta coluna de fornecedor (várias variantes possíveis)
        fornecedor_col = _find_col(df_contratos, ['fornecedor.nome', 'fornecedorNome', 'nome_fornecedor', 'nomeFornecedor', 'fornecedor.razaoSocialReceita', 'fornecedor.nomeFantasiaReceita'])
        if fornecedor_col and 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
            # threshold fixo (20%) conforme pedido
            threshold = 0.20
            top_n = st.sidebar.number_input("Top N fornecedores", min_value=3, max_value=200, value=20, step=1, key="aditivo_top_n")

            df_for_rec = df_filtrado.copy()
            df_for_rec['aditivo'] = df_for_rec['valorFinalCompra'] - df_for_rec['valorInicialCompra']
            df_for_rec['aditivo_pct'] = df_for_rec.apply(
                lambda r: (r['aditivo'] / r['valorInicialCompra']) if r.get('valorInicialCompra', 0) and r['valorInicialCompra'] > 0 else pd.NA,
                axis=1
            )

            # máscara para aditivos acima do threshold
            mask = df_for_rec['aditivo_pct'] > threshold

            total_by_forn = df_for_rec.groupby(fornecedor_col).size().rename('total_contracts')
            large_count = df_for_rec[mask].groupby(fornecedor_col).size().rename('large_aditivos_count')
            large_sum = df_for_rec[mask].groupby(fornecedor_col)['aditivo'].sum().rename('large_aditivos_sum')

            summary = pd.concat([total_by_forn, large_count, large_sum], axis=1).fillna(0)
            summary['pct_contracts_with_large'] = summary['large_aditivos_count'] / summary['total_contracts']
            summary = summary.sort_values(['large_aditivos_count', 'large_aditivos_sum'], ascending=False)

            if summary['large_aditivos_count'].sum() == 0:
                st.info(f"Nenhum contrato com aditivo > {threshold*100:.0f}% encontrado no período selecionado.")
            else:
                st.markdown(f"Fornecedores com contratos que tiveram aditivo > **{threshold*100:.0f}%**")
                top_summary = summary.head(int(top_n)).reset_index()

                # gráfico de barras: número de contratos com grandes aditivos por fornecedor
                fig_rec = px.bar(
                    top_summary.sort_values('large_aditivos_count'),
                    x='large_aditivos_count',
                    y=fornecedor_col,
                    orientation='h',
                    title=f'Top {int(top_n)} Fornecedores por # Contratos com Aditivo > {threshold*100:.0f}%',
                    labels={'large_aditivos_count': 'Contratos com Aditivo (contagem)', fornecedor_col: 'Fornecedor'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_rec, use_container_width=True, key='aditivos_recorrentes_chart')

                # tabela detalhada
                with st.expander(f"Ver tabela detalhada (Top {int(top_n)})"):
                    disp = top_summary.copy()
                    # formatar soma de aditivos
                    if 'large_aditivos_sum' in disp.columns:
                        disp['large_aditivos_sum'] = disp['large_aditivos_sum'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) and v != 0 else "R$ 0.00")
                    disp['pct_contracts_with_large'] = disp['pct_contracts_with_large'].map(lambda v: f"{v*100:.1f}%")
                    st.dataframe(disp.rename(columns={fornecedor_col: 'fornecedor'}))
        else:
            st.info("Coluna de fornecedor ou colunas de valor ausentes — pulando análise de aditivos recorrentes por fornecedor.")

        st.markdown("---")

        # --- LINHA 3: "Como" e "Por Onde" ---
        col_l3_1, col_l3_2 = st.columns(2)
        
        with col_l3_1:
            st.markdown("### Contratos por Modalidade")
            # AJUSTADO: Usa 'modalidadeCompra'
            df_modalidade = df_filtrado['modalidadeCompra'].value_counts().reset_index()
            
            fig_modalidade = px.bar(
                df_modalidade,
                x='count',
                y='modalidadeCompra',
                orientation='h',
                title='Contagem por Modalidade de Compra',
                labels={'count': 'Quantidade', 'modalidadeCompra': 'Modalidade'},
                color_discrete_sequence=COLOR_PALETTE
            )
            st.plotly_chart(fig_modalidade, use_container_width=True, key='modalidade_chart')
        with col_l3_2:
            st.markdown("### Gasto por Unidade Gestora (movido)")
            st.info("Os gráficos de Gasto e Aditivos por Unidade Gestora foram movidos para a seção superior, abaixo dos KPIs.")

            # --- Gráfico: Total em Aditivos por Unidade Gestora (Top 10) ---
            st.markdown("#### Total em Aditivos por Unidade Gestora (Top 10)")
            # precisamos das colunas de valor
            if 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
                # escolhe coluna UG disponível (prefer ug_col definido no sidebar)
                ug_for_ad = ug_col if (ug_col and ug_col in df_filtrado.columns) else _find_col(df_contratos, ['unidadeGestora.orgaoVinculado.sigla', 'orgaovinculado.sigla', 'orgaoVinculado.sigla', 'nome_uge', 'uge'])
                if ug_for_ad and ug_for_ad in df_filtrado.columns:
                    df_ad_ug = df_filtrado.copy()
                    df_ad_ug['aditivo'] = df_ad_ug['valorFinalCompra'] - df_ad_ug['valorInicialCompra']
                    # considera apenas aditivos positivos (incrementos)
                    df_ad_pos = df_ad_ug[df_ad_ug['aditivo'] > 0].copy()
                    if df_ad_pos.empty:
                        st.info('Nenhum aditivo positivo encontrado no período selecionado para as Unidades Gestoras.')
                    else:
                        df_ad_pos['ug_label'] = df_ad_pos[ug_for_ad].astype(str).str.strip().fillna('--N/A--')
                        sum_ad_ug = df_ad_pos.groupby('ug_label')['aditivo'].sum().reset_index()
                        if sum_ad_ug.empty:
                            st.info('Sem valores de aditivo por UG.')
                        else:
                            top_ad_ug = sum_ad_ug.nlargest(10, 'aditivo').sort_values('aditivo')
                            fig_ad_ug = px.bar(
                                top_ad_ug,
                                x='aditivo',
                                y='ug_label',
                                orientation='h',
                                title='Top 10 Unidades Gestoras por Total em Aditivos (R$)',
                                labels={'aditivo': 'Total Aditivos (R$)', 'ug_label': 'Unidade Gestora'},
                                color_discrete_sequence=COLOR_PALETTE
                            )
                            st.plotly_chart(fig_ad_ug, use_container_width=True, key='ug_aditivos_top10_lower')

                            with st.expander('Ver tabela detalhada de aditivos por UG (Top 10)'):
                                disp = top_ad_ug.copy()
                                disp['aditivo'] = disp['aditivo'].map(lambda v: f"R$ {v:,.2f}")
                                disp = disp.rename(columns={'ug_label': 'unidade_gestora', 'aditivo': 'total_aditivo'})
                                st.dataframe(disp)
                else:
                    st.info('Coluna de Unidade Gestora não encontrada para agrupar aditivos.')
            else:
                st.info('Colunas de valor ausentes — não é possível calcular aditivo por UG.')

        st.markdown("---")

        # --- LINHA 4: Análises de Fornecedores (Refatorado para abas) ---
        st.markdown("### Análises de Fornecedores")
        tab1, tab2, tab3 = st.tabs([
            "📈 Top 10 por Valor",
            "👥 Perfil (Tipo)",
            "🚨 Risco (Aditivos)"
        ])

        # Re-detecta colunas quando necessário (usa helper _find_col)
        fornecedor_col_candidates = ['nome_fornecedor', 'nomeFornecedor', 'fornecedor.nome', 'fornecedorNome', 'fornecedor.razaoSocialReceita', 'fornecedor.nomeFantasiaReceita']
        tipo_fornecedor_candidates = ['tipo_fornecedor', 'tipoFornecedor', 'fornecedor.tipo', 'tipo']

        with tab1:
            st.markdown("#### Top 10 Fornecedores por Valor (R$)")
            fornecedor_col = _find_col(df_contratos, fornecedor_col_candidates)
            if fornecedor_col and 'valorInicialCompra' in df_filtrado.columns:
                df_fornecedor = df_filtrado.groupby(fornecedor_col)['valorInicialCompra'].sum()
                df_fornecedor = df_fornecedor.nlargest(10).sort_values(ascending=True).reset_index()

                fig_fornecedor = px.bar(
                    df_fornecedor,
                    x='valorInicialCompra',
                    y=fornecedor_col,
                    orientation='h',
                    title='Top 10 Fornecedores por Valor (R$)',
                    labels={'valorInicialCompra': 'Valor Total (R$)', fornecedor_col: 'Fornecedor'},
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_fornecedor, use_container_width=True, key='fornecedor_top10_chart')
            else:
                st.info("Colunas de fornecedor/valor ausentes — pulando gráfico de fornecedores.")

        with tab2:
            st.markdown("#### Perfil dos Fornecedores por Tipo")
            tipo_fornecedor_col = _find_col(df_contratos, tipo_fornecedor_candidates)
            if tipo_fornecedor_col and 'valorInicialCompra' in df_filtrado.columns:
                df_tipo_forn = df_filtrado.groupby(tipo_fornecedor_col)['valorInicialCompra'].sum().reset_index()

                fig_tipo = px.pie(
                    df_tipo_forn,
                    names=tipo_fornecedor_col,
                    values='valorInicialCompra',
                    title='Distribuição do Valor por Tipo de Fornecedor',
                    hole=0.3,
                    color_discrete_sequence=COLOR_PALETTE
                )
                st.plotly_chart(fig_tipo, use_container_width=True, key='fornecedor_tipo_pie')
            else:
                st.info("Colunas para distribuição por tipo de fornecedor ausentes — pulando gráfico.")

        with tab3:
            st.markdown("#### Fornecedores com Maiores Aditivos (R$)")
            fornecedor_col = _find_col(df_contratos, fornecedor_col_candidates)
            if fornecedor_col and 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
                df_risco = df_filtrado.copy()
                df_risco['valor_aditivo'] = df_risco['valorFinalCompra'] - df_risco['valorInicialCompra']
                df_risco_pos = df_risco[df_risco['valor_aditivo'] > 0]

                if not df_risco_pos.empty:
                    df_risco_forn = df_risco_pos.groupby(fornecedor_col)['valor_aditivo'].sum()
                    df_risco_forn = df_risco_forn.nlargest(10).reset_index()

                    fig_risco = px.bar(
                        df_risco_forn.sort_values('valor_aditivo'),
                        x='valor_aditivo',
                        y=fornecedor_col,
                        orientation='h',
                        title='Top 10 Fornecedores por Valor Total de Aditivos (R$)',
                        labels={'valor_aditivo': 'Total Aditivos (R$)', fornecedor_col: 'Fornecedor'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    st.plotly_chart(fig_risco, use_container_width=True, key='fornecedor_risco_chart')

                    with st.expander("Ver tabela de fornecedores com maiores aditivos"):
                        disp = df_risco_forn.copy()
                        # formatar valores
                        disp['valor_aditivo'] = disp['valor_aditivo'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "R$ 0.00")
                        disp = disp.rename(columns={fornecedor_col: 'fornecedor'})
                        st.dataframe(disp)
                        st.caption("Mostra o valor total de aditivos positivos por fornecedor (Top 10).")
                else:
                    st.info('Nenhum aditivo positivo encontrado no período selecionado.')
            else:
                st.info('Colunas de fornecedor/valor ausentes — pulando análise de risco por aditivos.')

        st.markdown("---")

        # --- LINHA 5: Análises de Risco e Eficiência ---
        col_l5_1, col_l5_2 = st.columns(2)

        with col_l5_1:
            st.markdown("### Top 10 Maiores Aditivos Contratuais (R$)")
            # Análise 5: aditivo = valorFinalCompra - valorInicialCompra
            if 'valorInicialCompra' in df_filtrado.columns and 'valorFinalCompra' in df_filtrado.columns:
                df_aditivos = df_filtrado.copy()
                df_aditivos['aditivo'] = df_aditivos['valorFinalCompra'] - df_aditivos['valorInicialCompra']
                # Considera apenas aditivos positivos (aumento do valor)
                df_aditivos_pos = df_aditivos[df_aditivos['aditivo'] > 0]
                if not df_aditivos_pos.empty:
                    # calcula percentual do aditivo (quando valorInicialCompra > 0)
                    df_aditivos_pos = df_aditivos_pos.copy()
                    df_aditivos_pos['aditivo_pct'] = df_aditivos_pos.apply(
                        lambda r: (r['aditivo'] / r['valorInicialCompra']) if r.get('valorInicialCompra', 0) and r['valorInicialCompra'] > 0 else pd.NA,
                        axis=1
                    )

                    # métricas resumidas para aditivos
                    mean_aditivo = df_aditivos_pos['aditivo'].mean()
                    median_aditivo_pct = df_aditivos_pos['aditivo_pct'].median(skipna=True)
                    max_aditivo_pct = df_aditivos_pos['aditivo_pct'].max(skipna=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Médio Aditivo (R$)", formatar_valor(mean_aditivo if pd.notna(mean_aditivo) else 0))
                    m2.metric("Mediana Aditivo (%)", f"{median_aditivo_pct*100:.1f}%" if pd.notna(median_aditivo_pct) else "N/A")
                    m3.metric("Máx Aditivo (%)", f"{max_aditivo_pct*100:.1f}%" if pd.notna(max_aditivo_pct) else "N/A")

                    # detectar coluna de fornecedor e escolher colunas para exibição na tabela
                    fornecedor_col = _find_col(df_contratos, ['fornecedor.nome', 'fornecedorNome', 'nome_fornecedor', 'nomeFornecedor', 'fornecedor.razaoSocialReceita', 'fornecedor.nomeFantasiaReceita'])

                    cols = []
                    if fornecedor_col and fornecedor_col in df_aditivos_pos.columns:
                        cols.append(fornecedor_col)
                    elif 'numero' in df_aditivos_pos.columns:
                        cols.append('numero')

                    for c in ['valorInicialCompra', 'valorFinalCompra', 'aditivo', 'aditivo_pct']:
                        cols.append(c)

                    df_top = df_aditivos_pos.nlargest(10, 'aditivo')[cols]

                    # escolher eixo Y: preferir fornecedor, senão número do contrato
                    y_axis = fornecedor_col if (fornecedor_col and fornecedor_col in df_top.columns) else ('numero' if 'numero' in df_top.columns else df_top.index.astype(str))

                    fig_aditivos = px.bar(
                        df_top.sort_values('aditivo'),
                        x='aditivo',
                        y=y_axis,
                        orientation='h',
                        title='Top 10 Maiores Aditivos Contratuais (R$)',
                        labels={'aditivo': 'Aditivo (R$)', fornecedor_col: 'Fornecedor', 'numero': 'Número do Contrato'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    fig_aditivos.update_layout(margin=dict(l=80))
                    st.plotly_chart(fig_aditivos, use_container_width=True, key='top_aditivos_chart')

                    # mostra a tabela resumida (formatando valores)
                    with st.expander("Ver tabela dos Top 10 Aditivos"):
                        disp = df_top.copy()
                        if 'aditivo' in disp.columns:
                            disp['aditivo'] = disp['aditivo'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                        if 'valorInicialCompra' in disp.columns:
                            disp['valorInicialCompra'] = disp['valorInicialCompra'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                        if 'valorFinalCompra' in disp.columns:
                            disp['valorFinalCompra'] = disp['valorFinalCompra'].map(lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A")
                        if 'aditivo_pct' in disp.columns:
                            disp['aditivo_pct'] = disp['aditivo_pct'].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "N/A")
                        # renomear coluna de fornecedor para um rótulo consistente no display
                        if fornecedor_col and fornecedor_col in disp.columns:
                            disp = disp.rename(columns={fornecedor_col: 'fornecedor'})
                        st.dataframe(disp)
                else:
                    st.info("Não existem aditivos positivos neste período.")
            else:
                st.info("Colunas 'valorInicialCompra' ou 'valorFinalCompra' ausentes — pulando análise de aditivos.")

        with col_l5_2:
            st.markdown("### Distribuição da Duração dos Contratos (dias)")
            # Análise 6: duração em dias
            if 'dataInicioVigencia' in df_filtrado.columns and 'dataFimVigencia' in df_filtrado.columns:
                df_dur = df_filtrado.copy()
                df_dur['duracao_dias'] = (df_dur['dataFimVigencia'] - df_dur['dataInicioVigencia']).dt.days
                df_dur = df_dur[df_dur['duracao_dias'].notna()]
                if not df_dur.empty:
                    # sumariza métricas de duração
                    mean_days = int(df_dur['duracao_dias'].mean())
                    median_days = int(df_dur['duracao_dias'].median())
                    p90_days = int(df_dur['duracao_dias'].quantile(0.9))
                    count_gt_1y = int((df_dur['duracao_dias'] > 365).sum())
                    count_gt_3y = int((df_dur['duracao_dias'] > 365*3).sum())
                    count_gt_5y = int((df_dur['duracao_dias'] > 365*5).sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Média (dias)", f"{mean_days}")
                    c2.metric("Mediana (dias)", f"{median_days}")
                    c3.metric("90p (dias)", f"{p90_days}")

                    fig_dur = px.histogram(
                        df_dur,
                        x='duracao_dias',
                        nbins=40,
                        title='Distribuição da Duração dos Contratos (dias)',
                        labels={'duracao_dias': 'Duração (dias)'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    st.plotly_chart(fig_dur, use_container_width=True, key='duracao_histogram')

                    # mostra contagens de contratos longos
                    st.markdown(f"Contratos >1 ano: **{count_gt_1y}** • >3 anos: **{count_gt_3y}** • >5 anos: **{count_gt_5y}**")
                else:
                    st.info('Não há dados de início/fim válidos para calcular durações.')
            else:
                st.info("Colunas de datas de vigência ausentes — pulando análise de duração.")

        st.markdown("---")

        # --- LINHA 6: Análises de Auditoria e Status ---
        col_l6_1, col_l6_2 = st.columns(2)

        with col_l6_1:
            st.markdown("### Distribuição de Valor por Modalidade (Boxplot)")
            # Análise 7: boxplot valorInicialCompra por modalidadeCompra
            if 'modalidadeCompra' in df_filtrado.columns and 'valorInicialCompra' in df_filtrado.columns:
                df_box = df_filtrado.copy()
                # remove nulos
                df_box = df_box[df_box['modalidadeCompra'].notna() & df_box['valorInicialCompra'].notna()]
                if not df_box.empty:
                    fig_box = px.box(
                        df_box,
                        x='modalidadeCompra',
                        y='valorInicialCompra',
                        points='outliers',
                        title='Distribuição de Valor Inicial por Modalidade de Compra',
                        labels={'modalidadeCompra': 'Modalidade', 'valorInicialCompra': 'Valor Inicial (R$)'},
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    fig_box.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig_box, use_container_width=True, key='box_modalidade')
                else:
                    st.info('Dados insuficientes para boxplot por modalidade.')
            else:
                st.info("Colunas 'modalidadeCompra' ou 'valorInicialCompra' ausentes — pulando boxplot por modalidade.")

        with col_l6_2:
            st.markdown("### Situação Atual dos Contratos (porcentagem)")
            # Análise 8: situaçãoContrato distribution
            if 'situacaoContrato' in df_filtrado.columns:
                df_status = df_filtrado['situacaoContrato'].value_counts().reset_index()
                df_status.columns = ['situacao', 'count']
                if not df_status.empty:
                    fig_status = px.pie(
                        df_status,
                        names='situacao',
                        values='count',
                        title='Situação Atual dos Contratos',
                        hole=0.4,
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    st.plotly_chart(fig_status, use_container_width=True, key='status_pie')
                else:
                    st.info('Nenhuma informação de situação encontrada no período.')
            else:
                st.info("Coluna 'situacaoContrato' ausente — pulando gráfico de situação.")

        st.markdown("---")

        # --- LINHA 7: Busca Avançada e Tabela de Dados (Expansível) ---
        st.markdown("### Busca avançada / Filtro de Dados")

        # re-detecta colunas relevantes para busca
        fornecedor_name_col = _find_col(df_contratos, ['fornecedor.nome', 'fornecedorNome', 'nome_fornecedor', 'nomeFornecedor', 'fornecedor.razaoSocialReceita', 'fornecedor.nomeFantasiaReceita'])
        cnpj_col_local = _find_col(df_contratos, ['fornecedor.cnpjFormatado', 'fornecedor.cnpj', 'fornecedor.cnpjFormatado'])

        df_search = df_filtrado.copy()
        # cria coluna supplier_id (CNPJ sem pontuação) ou fallback para nome uppercase
        if cnpj_col_local and cnpj_col_local in df_search.columns:
            df_search['supplier_id'] = df_search[cnpj_col_local].astype(str).str.replace(r'\D', '', regex=True).replace({'': pd.NA})
        else:
            if fornecedor_name_col and fornecedor_name_col in df_search.columns:
                df_search['supplier_id'] = df_search[fornecedor_name_col].astype(str).str.strip().str.upper()
            else:
                df_search['supplier_id'] = pd.NA

        # coluna de nome legível para busca/exibição
        if fornecedor_name_col and fornecedor_name_col in df_search.columns:
            df_search['fornecedor_name'] = df_search[fornecedor_name_col].astype(str)
        else:
            df_search['fornecedor_name'] = pd.NA

        # UI: termo de busca e campos a pesquisar
        search_term = st.text_input("Pesquisar (ID / número / fornecedor)", value="", help="Digite parte do CNPJ/ID, número do contrato ou nome do fornecedor.")
        possible_fields = []
        if 'supplier_id' in df_search.columns:
            possible_fields.append('supplier_id')
        if 'numero' in df_search.columns:
            possible_fields.append('numero')
        possible_fields.append('fornecedor_name')

        fields = st.multiselect("Campos a buscar", options=possible_fields, default=possible_fields)

        # aplica filtro
        if search_term and len(search_term.strip()) > 0:
            q = str(search_term).strip()
            mask = pd.Series(False, index=df_search.index)
            if 'supplier_id' in fields and 'supplier_id' in df_search.columns:
                mask = mask | df_search['supplier_id'].astype(str).str.contains(q, case=False, na=False)
            if 'numero' in fields and 'numero' in df_search.columns:
                mask = mask | df_search['numero'].astype(str).str.contains(q, case=False, na=False)
            if 'fornecedor_name' in fields and 'fornecedor_name' in df_search.columns:
                mask = mask | df_search['fornecedor_name'].astype(str).str.contains(q, case=False, na=False)

            df_filtered_search = df_search[mask].copy()
            st.markdown(f"Resultados encontrados: **{len(df_filtered_search)}** contratos")
        else:
            df_filtered_search = df_search.copy()
            st.markdown(f"Mostrando todos os contratos no período selecionado: **{len(df_filtered_search)}**")

        # KPIs rápidos sobre o subconjunto filtrado
        col_a, col_b, col_c, col_d = st.columns(4)
        total_contracts = len(df_filtered_search)
        total_val = df_filtered_search['valorInicialCompra'].sum() if 'valorInicialCompra' in df_filtered_search.columns else 0
        if 'valorFinalCompra' in df_filtered_search.columns and 'valorInicialCompra' in df_filtered_search.columns:
            aditivos_series = (df_filtered_search['valorFinalCompra'] - df_filtered_search['valorInicialCompra']).clip(lower=0)
            total_aditivos_filtrados = aditivos_series.sum()
        else:
            total_aditivos_filtrados = 0

        unique_suppliers = int(df_filtered_search['supplier_id'].nunique()) if 'supplier_id' in df_filtered_search.columns else 0

        col_a.metric("Contratos (filtrados)", f"{total_contracts}")
        col_b.metric("Valor total (R$)", formatar_valor(total_val))
        col_c.metric("Total em Aditivos (R$)", formatar_valor(total_aditivos_filtrados))
        col_d.metric("Fornecedores distintos", f"{unique_suppliers}")

        # Top fornecedores no subconjunto (por valor)
        if 'valorInicialCompra' in df_filtered_search.columns:
            group_for = None
            if 'fornecedor_name' in df_filtered_search.columns and df_filtered_search['fornecedor_name'].notna().any():
                group_for = df_filtered_search.groupby('fornecedor_name')['valorInicialCompra'].sum().reset_index().sort_values('valorInicialCompra', ascending=False).head(5)
                st.markdown("#### Top 5 Fornecedores no filtro (por Valor)")
                fig_topf = px.bar(group_for, x='valorInicialCompra', y='fornecedor_name', orientation='h', labels={'valorInicialCompra':'Valor (R$)', 'fornecedor_name':'Fornecedor'}, color_discrete_sequence=COLOR_PALETTE)
                st.plotly_chart(fig_topf, use_container_width=True, key='topf_chart')

        # mostra tabela detalhada (expansível)
        with st.expander("Clique para visualizar os dados filtrados"):
            # formata algumas colunas para exibição
            disp = df_filtered_search.copy()
            if 'valorInicialCompra' in disp.columns:
                disp['valorInicialCompra'] = disp['valorInicialCompra'].map(lambda v: f"R$ {v:,.2f}")
            if 'valorFinalCompra' in disp.columns:
                disp['valorFinalCompra'] = disp['valorFinalCompra'].map(lambda v: f"R$ {v:,.2f}")
            # mostra colunas úteis (prioriza fornecedor_name e supplier_id)
            cols_show = []
            if 'supplier_id' in disp.columns:
                cols_show.append('supplier_id')
            if 'fornecedor_name' in disp.columns:
                cols_show.append('fornecedor_name')
            if 'numero' in disp.columns:
                cols_show.append('numero')
            # adiciona algumas colunas de valor/datas se existirem
            for c in ['valorInicialCompra', 'valorFinalCompra', 'dataAssinatura']:
                if c in disp.columns:
                    cols_show.append(c)

            # se não houver colunas detectadas, mostra tudo
            if len(cols_show) == 0:
                st.dataframe(disp)
            else:
                st.dataframe(disp[cols_show])

        # --- Detalhes completos de um contrato selecionado ---
        st.markdown("#### Visualizar detalhes de um contrato")
        if not df_filtered_search.empty:
            # identifica cada contrato por 'numero' se existir, senão pelo index
            df_filtered_search = df_filtered_search.copy()
            if 'numero' in df_filtered_search.columns:
                df_filtered_search['contract_key'] = df_filtered_search['numero'].astype(str)
            else:
                df_filtered_search['contract_key'] = df_filtered_search.index.astype(str)

            # rótulo legível: numero | fornecedor | dataAssinatura (quando disponíveis)
            def _make_label(r):
                parts = []
                if pd.notna(r.get('contract_key')):
                    parts.append(str(r.get('contract_key')))
                if 'fornecedor_name' in r and pd.notna(r.get('fornecedor_name')):
                    parts.append(str(r.get('fornecedor_name')))
                if 'dataAssinatura' in r and pd.notna(r.get('dataAssinatura')):
                    try:
                        parts.append(pd.to_datetime(r.get('dataAssinatura')).strftime('%Y-%m-%d'))
                    except Exception:
                        parts.append(str(r.get('dataAssinatura')))
                return ' | '.join(parts)

            df_filtered_search['__label'] = df_filtered_search.apply(_make_label, axis=1)

            labels = df_filtered_search['__label'].fillna('').tolist()
            labels = ['-- nenhum --'] + labels
            sel = st.selectbox('Escolha um contrato para ver detalhes', options=labels)
            if sel and sel != '-- nenhum --':
                row = df_filtered_search[df_filtered_search['__label'] == sel].iloc[0]
                st.subheader('Detalhes do contrato selecionado')

                # mostra campos como JSON (legível) e também em modo chave:valor
                details = row.drop(labels=['__label', 'contract_key'], errors='ignore').to_dict()
                # botão para download CSV/JSON
                single_df = pd.DataFrame([details])
                csv_bytes = single_df.to_csv(sep=';', index=False).encode('utf-8')
                json_text = json.dumps(details, default=str, ensure_ascii=False, indent=2)

                cold1, cold2 = st.columns([1, 1])
                with cold1:
                    st.download_button('Baixar registro (CSV)', data=csv_bytes, file_name=f"contrato_{row.get('contract_key', 'x')}.csv", mime='text/csv')
                with cold2:
                    st.download_button('Baixar registro (JSON)', data=json_text, file_name=f"contrato_{row.get('contract_key', 'x')}.json", mime='application/json')

                # Exibe o JSON e o 'objeto_contrato' de forma legível
                st.markdown('**Visão completa (JSON)**')
                st.json(details)

                if 'objeto_contrato' in row.index and pd.notna(row['objeto_contrato']):
                    st.markdown('**Objeto da Contratação**')
                    st.write(row['objeto_contrato'])
        else:
            st.info('Não há contratos no subconjunto filtrado para exibir detalhes.')

# Se o usuário clicar na outra página
elif pagina_selecionada == "Licitações (em breve)":
    st.title("🚧 Painel de Licitações")
    st.info("Esta seção ainda está em construção.")

# Se os dados nem sequer foram carregados no início
elif 'df_contratos' in locals() and df_contratos.empty:
    st.info("Aguardando geração do arquivo de dados 'dados_contratos.csv'...")