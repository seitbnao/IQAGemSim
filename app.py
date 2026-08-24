from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.synthetic import generate_synthetic_estuary
from src.qc import apply_qc
from src.iqa import compute_iqa_dataframe, classify_iqa, WEIGHTS
from src.ccme import calculate_ccme_wqi, classify_ccme, objective_status_table, DEFAULT_OBJECTIVES
from src.spatial import interpolate_surface
from src.temporal import train_models, forecast_station, feature_importance
from src.scenarios import PRESETS
from src.priority import monitoring_priority_surface
from src.utils import validate_columns, add_forecast_iqa, monte_carlo_iqa
from src.db import save_scenario

st.set_page_config(
    page_title="Gêmeo Digital da Qualidade da Água",
    page_icon=None,
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

PARAM_LABELS = {
    "iqa": "IQA",
    "ph": "pH",
    "do_mg_l": "Oxigênio dissolvido (mg/L)",
    "dbo_mg_l": "DBO (mg/L)",
    "ecoli_mpn_100ml": "E. coli (NMP/100 mL)",
    "water_temp_c": "Temperatura da água (°C)",
    "delta_temp_c": "Variação de temperatura (°C)",
    "total_n_mg_l": "Nitrogênio total (mg/L)",
    "total_p_mg_l": "Fósforo total (mg/L)",
    "turbidity_ntu": "Turbidez (NTU)",
    "total_solids_mg_l": "Sólidos totais (mg/L)",
    "salinity_psu": "Salinidade (PSU)",
    "chlorophyll_ug_l": "Clorofila-a (µg/L)",
}

def contour_figure(surface, stations, value_col, title, zlabel):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=surface["x"], y=surface["y"], z=surface["surface"],
        colorbar=dict(title=zlabel), contours=dict(showlabels=True),
        hovertemplate="x=%{x:.1f} km<br>y=%{y:.1f} km<br>valor=%{z:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stations["x_km"], y=stations["y_km"], mode="markers+text",
        text=stations["station_id"], textposition="top center",
        marker=dict(size=9, line=dict(width=1)),
        customdata=stations[[value_col]].to_numpy(),
        hovertemplate="Estação %{text}<br>"+zlabel+"=%{customdata[0]:.2f}<extra></extra>",
        name="Estações",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Eixo longitudinal sintético (km)",
        yaxis_title="Eixo transversal sintético (km)",
        height=540,
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig

def priority_figure(result, stations):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=result["x"], y=result["y"], z=result["priority"],
        colorbar=dict(title="Prioridade"),
        contours=dict(start=0, end=100, size=10, showlabels=True),
        hovertemplate="x=%{x:.1f} km<br>y=%{y:.1f} km<br>prioridade=%{z:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stations["x_km"], y=stations["y_km"], mode="markers+text",
        text=stations["station_id"], textposition="top center",
        marker=dict(size=8, line=dict(width=1)),
        name="Estações existentes",
    ))
    fig.update_layout(
        xaxis_title="Eixo longitudinal sintético (km)",
        yaxis_title="Eixo transversal sintético (km)",
        height=540,
        margin=dict(l=30, r=30, t=50, b=40),
    )
    return fig

@st.cache_data
def synthetic_data():
    return generate_synthetic_estuary(n_stations=14, n_days=420, seed=42)

@st.cache_data
def prepare_data(raw: pd.DataFrame):
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = apply_qc(d)
    # Exclui apenas leituras fisicamente inválidas do cálculo; suspeitas ficam marcadas.
    clean = d[~d["qc_exclude"]].copy()
    clean = compute_iqa_dataframe(clean)
    return d, clean

@st.cache_resource
def cached_train(data: pd.DataFrame):
    return train_models(data)

st.title("Simulador de Gêmeo Digital da Qualidade da Água")
st.caption(
    "Protótipo metodológico espaço-temporal: observação → QA/QC → índices → "
    "interpolação → previsão das variáveis → cenários → incerteza → apoio à decisão."
)

with st.sidebar:
    st.header("Fonte de dados")
    source = st.radio("Base", ["Estuário sintético", "Carregar CSV"], index=0)
    uploaded = None
    if source == "Carregar CSV":
        uploaded = st.file_uploader("CSV no formato do protótipo", type=["csv"])
        st.caption("Use data/template_observacoes.csv como referência de colunas.")

    st.divider()
    st.header("Estado do sistema")
    st.info(
        "Esta versão é um simulador de gêmeo digital. Ela possui sincronização por arquivo, "
        "modelagem, previsão e cenários, mas não está conectada a sensores em tempo real."
    )

if source == "Carregar CSV" and uploaded is not None:
    raw = pd.read_csv(uploaded)
    missing = validate_columns(raw)
    if missing:
        st.error("Colunas ausentes: " + ", ".join(missing))
        st.stop()
elif source == "Carregar CSV" and uploaded is None:
    st.warning("Carregue um CSV ou selecione o estuário sintético.")
    st.stop()
else:
    raw = synthetic_data()

raw_qc, data = prepare_data(raw)
if data.empty:
    st.error("Nenhuma observação válida após QA/QC.")
    st.stop()

tabs = st.tabs([
    "Estado atual",
    "Previsão",
    "Cenários",
    "IQA + CCME-WQI",
    "Prioridade de monitoramento",
    "Dados e QA/QC",
    "Metodologia",
])

with tabs[0]:
    latest_date = data["date"].max()
    dates = sorted(data["date"].dt.date.unique())
    selected_date = st.select_slider("Data do estado digital", options=dates, value=latest_date.date())
    current = data[data["date"].dt.date == selected_date].copy()

    c1, c2, c3, c4 = st.columns(4)
    mean_iqa = current["iqa"].mean()
    c1.metric("IQA médio", f"{mean_iqa:.1f}", classify_iqa(mean_iqa))
    c2.metric("Menor IQA", f"{current['iqa'].min():.1f}")
    c3.metric("Estações", f"{current['station_id'].nunique()}")
    suspect_day = raw_qc[raw_qc["date"].dt.date == selected_date]["qc_suspect"].mean()*100
    c4.metric("Leituras suspeitas", f"{suspect_day:.1f}%")

    param = st.selectbox(
        "Camada espacial",
        options=["iqa", "do_mg_l", "turbidity_ntu", "salinity_psu", "total_n_mg_l", "total_p_mg_l"],
        format_func=lambda x: PARAM_LABELS.get(x, x),
    )
    surf = interpolate_surface(current, param)
    st.plotly_chart(
        contour_figure(
            surf, current, param,
            f"{PARAM_LABELS.get(param,param)} — {selected_date}",
            PARAM_LABELS.get(param,param),
        ),
        use_container_width=True,
    )
    st.caption(f"Método espacial: {surf['method']}. A variância da interpolação é armazenada separadamente.")

    unc = {**surf, "surface": surf["variance"]}
    st.plotly_chart(
        contour_figure(unc, current, param, "Incerteza espacial da interpolação", "Variância"),
        use_container_width=True,
    )

    show_cols = ["station_id", "x_km", "y_km", "iqa", "iqa_class", "do_mg_l", "turbidity_ntu", "salinity_psu"]
    st.dataframe(current[show_cols].sort_values("iqa"), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Previsão das variáveis antes do recálculo do IQA")
    st.caption("O Random Forest é comparado a uma baseline de persistência. O índice é recalculado somente após prever os parâmetros.")

    with st.spinner("Treinando/recuperando modelos..."):
        bundle = cached_train(data)

    c1, c2, c3 = st.columns(3)
    station = c1.selectbox("Estação", sorted(data["station_id"].unique()), key="forecast_station")
    horizon = c2.slider("Horizonte (dias)", 1, 14, 7, key="forecast_horizon")
    explain_target = c3.selectbox(
        "Modelo para explicabilidade",
        list(bundle.models.keys()),
        format_func=lambda x: PARAM_LABELS.get(x, x),
    )

    forecast = add_forecast_iqa(forecast_station(data, bundle, station, horizon=horizon))
    mc = monte_carlo_iqa(forecast)
    hist = data[data["station_id"] == station].sort_values("date").tail(60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["iqa"], name="IQA observado", mode="lines"))
    fig.add_trace(go.Scatter(x=mc["date"], y=mc["iqa_p50"], name="IQA previsto", mode="lines+markers"))
    fig.add_trace(go.Scatter(
        x=pd.concat([mc["date"], mc["date"].iloc[::-1]]),
        y=pd.concat([mc["iqa_p90"], mc["iqa_p10"].iloc[::-1]]),
        fill="toself", line=dict(width=0), name="Intervalo 10–90%",
        hoverinfo="skip",
    ))
    fig.update_layout(height=480, yaxis_title="IQA", xaxis_title="Data")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Validação prospectiva por alvo**")
    metrics = bundle.metrics.copy()
    metrics["variavel"] = metrics["target"].map(lambda x: PARAM_LABELS.get(x, x))
    st.dataframe(
        metrics[["variavel","MAE_RF","RMSE_RF","R2_RF","MAE_persistencia","ganho_MAE_pct"]]
        .sort_values("ganho_MAE_pct", ascending=False),
        use_container_width=True, hide_index=True,
    )

    imp = feature_importance(bundle, explain_target).head(12)
    fig_imp = px.bar(
        imp.sort_values("importance"),
        x="importance", y="feature", orientation="h",
        title=f"Importância global das entradas — {PARAM_LABELS.get(explain_target, explain_target)}",
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "A importância do Random Forest é uma explicação estatística do modelo, não uma afirmação causal. "
        "Em produção, o módulo pode ser substituído/expandido por SHAP."
    )

with tabs[2]:
    st.subheader("Simulação de cenários what-if")
    st.warning("Cenários são experimentos hipotéticos e não devem ser interpretados como previsão probabilística.")

    with st.spinner("Carregando modelos..."):
        bundle = cached_train(data)

    c1, c2, c3 = st.columns(3)
    station_s = c1.selectbox("Estação", sorted(data["station_id"].unique()), key="scenario_station")
    horizon_s = c2.slider("Horizonte (dias)", 1, 14, 7, key="scenario_horizon")
    preset_name = c3.selectbox("Biblioteca de cenários", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    s1, s2, s3 = st.columns(3)
    rain_pct = s1.slider("Precipitação (%)", -80, 150, int(preset["rainfall_pct"]))
    flow_pct = s1.slider("Vazão (%)", -70, 100, int(preset["flow_pct"]))
    tide_pct = s2.slider("Amplitude de maré (%)", -50, 80, int(preset["tide_amplitude_pct"]))
    sal_shift = s2.slider("Deslocamento de salinidade (PSU)", -10.0, 10.0, float(preset["salinity_shift_psu"]), 0.5)
    nutrient = s3.slider("Carga de nutrientes (%)", -50, 250, int(preset["nutrient_load_pct"]))
    turbidity = s3.slider("Carga de turbidez (%)", -50, 250, int(preset["turbidity_load_pct"]))
    point = st.slider("Intensidade de fonte pontual (0–100)", 0, 100, int(preset["point_source_intensity"]))

    scenario = {
        "rainfall_pct": rain_pct,
        "flow_pct": flow_pct,
        "tide_amplitude_pct": tide_pct,
        "salinity_shift_psu": sal_shift,
        "nutrient_load_pct": nutrient,
        "turbidity_load_pct": turbidity,
        "point_source_intensity": point,
    }

    baseline = add_forecast_iqa(forecast_station(data, bundle, station_s, horizon_s, PRESETS["Linha de base"]))
    simulated = add_forecast_iqa(forecast_station(data, bundle, station_s, horizon_s, scenario))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline["date"], y=baseline["iqa"], name="Linha de base", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=simulated["date"], y=simulated["iqa"], name=f"Cenário: {preset_name}", mode="lines+markers"))
    fig.update_layout(height=470, yaxis_title="IQA recalculado", xaxis_title="Data")
    st.plotly_chart(fig, use_container_width=True)

    delta = simulated.iloc[-1]["iqa"] - baseline.iloc[-1]["iqa"]
    c1, c2, c3 = st.columns(3)
    c1.metric("IQA final — base", f"{baseline.iloc[-1]['iqa']:.1f}")
    c2.metric("IQA final — cenário", f"{simulated.iloc[-1]['iqa']:.1f}")
    c3.metric("Δ IQA", f"{delta:+.1f}")

    compare_cols = ["do_mg_l","dbo_mg_l","ecoli_mpn_100ml","total_n_mg_l","total_p_mg_l","turbidity_ntu","total_solids_mg_l","salinity_psu"]
    rows = []
    for col in compare_cols:
        if col in baseline and col in simulated:
            b = float(baseline.iloc[-1][col])
            s = float(simulated.iloc[-1][col])
            rows.append({
                "Variável": PARAM_LABELS.get(col, col),
                "Base": b,
                "Cenário": s,
                "Variação (%)": 100*(s-b)/max(abs(b), 1e-9),
            })
    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    csv = simulated.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar trajetória do cenário (CSV)", csv, "cenario_simulado.csv", "text/csv")

    if st.button("Registrar cenário no banco local"):
        save_scenario(
            station_s, horizon_s, scenario,
            baseline.iloc[-1]["iqa"], simulated.iloc[-1]["iqa"]
        )
        st.success("Execução registrada em data/digital_twin.db.")

with tabs[3]:
    st.subheader("Índices em camadas")
    c1, c2 = st.columns(2)
    station_q = c1.selectbox("Estação", sorted(data["station_id"].unique()), key="quality_station")
    days = c2.selectbox("Janela CCME-WQI", [30, 60, 90, 180], index=2)

    ds = data[data["station_id"] == station_q].sort_values("date").tail(days)
    ccme = calculate_ccme_wqi(ds, DEFAULT_OBJECTIVES)
    latest = ds.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IQA mais recente", f"{latest['iqa']:.1f}", latest["iqa_class"])
    m2.metric("CCME-WQI da janela", f"{ccme['ccme_wqi']:.1f}", classify_ccme(ccme["ccme_wqi"]))
    m3.metric("F1 — escopo", f"{ccme['F1']:.1f}%")
    m4.metric("F2 — frequência", f"{ccme['F2']:.1f}%")

    st.caption(
        "Os objetivos do CCME-WQI desta versão são demonstrativos e configuráveis; "
        "não representam automaticamente enquadramento legal de uma água estuarina."
    )
    st.dataframe(objective_status_table(ds, DEFAULT_OBJECTIVES), use_container_width=True, hide_index=True)

    qcols = [c for c in latest.index if c.startswith("q_")]
    qtable = pd.DataFrame({
        "Subíndice IQA": [c.replace("q_","") for c in qcols],
        "q": [latest[c] for c in qcols],
        "Peso": [WEIGHTS.get(c.replace("q_",""), np.nan) for c in qcols],
    })
    qtable["Parâmetro"] = qtable["Subíndice IQA"].map(lambda x: PARAM_LABELS.get(x, x))
    st.dataframe(qtable[["Parâmetro","q","Peso"]].sort_values("q"), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Mapa multicritério para planejamento adaptativo de monitoramento")
    latest = data[data["date"] == data["date"].max()].copy()

    c1, c2, c3, c4 = st.columns(4)
    wr = c1.slider("Peso: criticidade", 0, 100, 35)
    wu = c2.slider("Peso: incerteza", 0, 100, 30)
    wp = c3.slider("Peso: pressão", 0, 100, 20)
    wd = c4.slider("Peso: distância", 0, 100, 15)

    priority = monitoring_priority_surface(
        latest,
        {"risk": wr, "uncertainty": wu, "pressure": wp, "distance": wd},
    )
    st.plotly_chart(priority_figure(priority, latest), use_container_width=True)
    st.caption(
        "A superfície combina criticidade ambiental, variância espacial, pressão antrópica e distância da rede. "
        "Ela pode alimentar posteriormente AHP/PROMETHEE e otimização de portfólio sob orçamento."
    )

with tabs[5]:
    st.subheader("Rastreabilidade, origem e flags de qualidade")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observações brutas", f"{len(raw_qc):,}".replace(",", "."))
    c2.metric("Excluídas por limite físico", f"{int(raw_qc['qc_exclude'].sum())}")
    c3.metric("Marcadas como suspeitas", f"{int(raw_qc['qc_suspect'].sum())}")

    st.dataframe(raw_qc.sort_values("date", ascending=False).head(500), use_container_width=True, hide_index=True)
    processed_csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar dados processados", processed_csv, "dados_processados_gemeo.csv", "text/csv")

with tabs[6]:
    st.subheader("Correspondência com a metodologia do artigo")
    st.markdown("""
1. **Estado digital espaço-temporal:** cada registro contém posição, tempo, variáveis ambientais, origem e flags.
2. **QA/QC:** limites físicos excluem leituras impossíveis; saltos plausíveis são apenas marcados como suspeitos.
3. **Índices em paralelo:** IQA, subíndices e CCME-WQI são mantidos separadamente.
4. **Modelagem espacial:** Kriging ordinário gera campo contínuo e variância; IDW funciona como fallback.
5. **Previsão:** Random Forest prevê os parâmetros; a persistência é usada como baseline.
6. **Propagação:** o IQA é recalculado após a previsão e recebe intervalo por Monte Carlo.
7. **Explicabilidade:** importância das entradas é mostrada separadamente da decomposição matemática do IQA.
8. **Cenários:** perturbações de chuva, vazão, maré, salinidade, nutrientes, turbidez e fonte pontual são experimentos what-if.
9. **Planejamento adaptativo:** risco, incerteza, pressão e distância geram uma superfície de prioridade.
10. **Versionamento:** cenários podem ser registrados em banco local SQLite; em produção, migrar para PostgreSQL/PostGIS.
""")
    st.info(
        "Maturidade atual: protótipo de simulação com modelagem e cenários. "
        "Para ser um gêmeo digital operacional completo, falta a ingestão automática de sensores/fontes reais, "
        "orquestração de atualização e validação contínua em ambiente de produção."
    )
