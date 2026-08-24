# Simulador de Gêmeo Digital da Qualidade da Água

Protótipo executável que implementa a metodologia do artigo
**“Gêmeo Digital Orientado por Índices de Qualidade da Água para Sistemas Estuarinos”**.

## O que o sistema faz

- gera uma base sintética espaço-temporal de um estuário;
- recebe um CSV com dados reais no mesmo esquema;
- aplica QA/QC com flags de exclusão e suspeita;
- calcula IQA e seus nove subíndices;
- calcula CCME-WQI em uma janela temporal;
- interpola campos espaciais por Kriging ordinário, mantendo a variância;
- treina Random Forest para prever variáveis ambientais;
- compara o modelo com baseline de persistência;
- recalcula o IQA depois das previsões;
- propaga incerteza do IQA por Monte Carlo;
- executa cenários what-if;
- produz mapa multicritério de prioridade de monitoramento;
- registra execuções de cenário em SQLite.

## Importante sobre o IQA

Os **pesos** do IQA usados no protótipo são os divulgados pela ANA:
OD 0,17; coliformes/E. coli 0,15; pH 0,12; DBO 0,10; variação de temperatura 0,10;
nitrogênio total 0,10; fósforo total 0,10; turbidez 0,08; sólidos/resíduo total 0,08.

As curvas q_i foram discretizadas no código para permitir a simulação e reproduzem a forma geral
das curvas NSF/CETESB. Antes de uso científico final ou regulatório, substitua os pontos de
`src/iqa.py` pelas curvas/tabelas oficiais adotadas no estudo e documente a versão.

Os objetivos do CCME-WQI em `src/ccme.py` também são **demonstrativos**. Para uma aplicação real,
configure os objetivos conforme a classe, uso e enquadramento do corpo hídrico.

## Execução no Windows

1. Instale Python 3.11 ou superior.
2. Dê duplo clique em `run_windows.bat`.

Ou execute:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker build -t gemeo-agua .
docker run --rm -p 8501:8501 gemeo-agua
```

Acesse `http://localhost:8501`.

## Dados reais

O arquivo `data/template_observacoes.csv` contém o cabeçalho necessário.
As coordenadas `x_km` e `y_km` são coordenadas cartesianas do domínio do modelo.
Em produção, pode-se utilizar UTM ou realizar uma etapa de reprojeção antes da modelagem espacial.

Campos mínimos:

- date, station_id
- x_km, y_km
- anthropic_pressure
- rain_mm, flow_m3_s, tide_m, salinity_psu
- water_temp_c, delta_temp_c
- ph, do_mg_l, dbo_mg_l, ecoli_mpn_100ml
- total_n_mg_l, total_p_mg_l, turbidity_ntu, total_solids_mg_l

## Arquitetura de produção recomendada

A versão local usa CSV + SQLite para facilitar testes. Para implantação operacional:

- **PostgreSQL/PostGIS:** observações, estações, geometrias, grades, previsões e cenários;
- **API FastAPI:** ingestão, consulta e acionamento dos modelos;
- **scheduler/worker:** atualização incremental e retreinamento;
- **object storage:** artefatos e versões dos modelos;
- **Streamlit/React/GIS web:** interface;
- **sensores/ANA/laboratórios/satélites:** fontes de ingestão;
- **registro de versões:** dataset, modelo, data de treino e métricas.

## Limitações científicas do protótipo

1. A base padrão é sintética.
2. Os efeitos de cenário incluem relações aprendidas na base sintética e uma camada explícita de perturbação.
3. Não existe modelo hidrodinâmico de advecção-dispersão nesta versão.
4. A distância espacial é euclidiana.
5. O Random Forest é um modelo inicial; a seleção final deve comparar alternativas em validação prospectiva.
6. A importância das variáveis do Random Forest não implica causalidade.
7. O sistema ainda não recebe sensores em tempo real.

Essas limitações são deliberadas: o objetivo desta versão é fornecer um **MVP científico reproduzível**
que possa ser calibrado com os dados reais do estudo.
