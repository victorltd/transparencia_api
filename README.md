# Transparência API - Dashboard de Contratos e Despesas

Este projeto coleta, processa e visualiza dados de contratos e despesas de órgãos públicos do Nordeste brasileiro, utilizando dados do Portal da Transparência. Ele inclui scripts de coleta, tratamento e um dashboard interativo em Streamlit para análise dos dados.

## Funcionalidades
- **Coleta automática de contratos e despesas** de diversos órgãos federais do Nordeste.
- **Processamento e limpeza dos dados** para facilitar análises.
- **Dashboard interativo** para visualização de contratos e despesas, incluindo gráficos, KPIs e nuvem de palavras.
- **Atualização automática via GitHub Actions** para manter os dados sempre atualizados.

## Estrutura do Projeto
```
├── app.py                   # Dashboard Streamlit
├── coleta.py                # Coleta e processamento de contratos
├── coleta_despesas.py       # Coleta e processamento de despesas
├── df_final_corrigido.csv   # Dados finais de contratos (gerado automaticamente)
├── df_despesas_nordeste.csv # Dados finais de despesas (gerado automaticamente)
├── requirements.txt         # Dependências do projeto
├── .env                     # Chave da API (não versionado)
├── .gitignore               # Arquivos ignorados pelo git
└── .github/workflows/
	 └── atualizar_dados.yml  # Workflow de atualização automática
```

## Como executar localmente
1. **Clone o repositório:**
	```bash
	git clone https://github.com/victorltd/transparencia_api
	cd transparencia_api
	```
2. **Configure a chave da API:**
	- Copie o arquivo `.env` de exemplo e insira sua chave do Portal da Transparência:
	  ```
	  cp .env .env
	  # Edite o arquivo e insira sua chave
	  ```
3. **Instale as dependências:**
	```bash
	pip install -r requirements.txt
	```
4. **Execute os scripts de coleta (opcional):**
	```bash
	python coleta.py
	python coleta_despesas.py
	```
	Os arquivos CSV serão gerados na pasta do projeto.
5. **Inicie o dashboard:**
	```bash
	streamlit run app.py
	```

## Atualização automática dos dados
- O workflow GitHub Actions (`.github/workflows/atualizar_dados.yml`) executa diariamente a coleta e atualização dos arquivos CSV, desde que a chave da API esteja configurada como secret no repositório.

## Principais dependências
- [Streamlit](https://streamlit.io/) - Dashboard interativo
- [Pandas](https://pandas.pydata.org/) - Manipulação de dados
- [Requests](https://requests.readthedocs.io/) - Requisições HTTP
- [WordCloud](https://github.com/amueller/word_cloud) - Nuvem de palavras
- [Plotly](https://plotly.com/python/) - Gráficos interativos
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Gerenciamento de variáveis de ambiente

## Observações
- Os arquivos CSV gerados podem ser grandes, dependendo do período e órgãos consultados.
- A chave da API do Portal da Transparência é obrigatória para coleta dos dados.
- O dashboard foi ajustado para funcionar com os arquivos gerados pelos scripts deste projeto.

## Licença
Este projeto é distribuído sob a licença MIT.

## Autor
Desenvolvido por Victor.