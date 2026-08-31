"""Dashboard interactivo local para explorar el tablero de modelos."""

from pathlib import Path

import pandas as pd
import streamlit as st

from reporte_excel import _metrics, _traces, weekly_status


ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_data():
    metrics = _metrics(ROOT)
    traces = _traces(ROOT)
    daily = []
    for model, frame in traces.items():
        frame = frame.copy()
        frame["fecha_objetivo"] = pd.to_datetime(frame["fecha_objetivo"])
        frame["acierto_pct"] = (1 - (frame["real"] - frame["proyectado"]) / frame["real"].where(frame["real"] != 0)).astype(float)
        frame["error_abs"] = (frame["proyectado"] - frame["real"]).abs()
        daily.append(frame)
    daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    weekly = (daily.groupby(["modelo", "finca", "fecha_origen"], as_index=False)
              .agg(real=("real", "sum"), proyectado=("proyectado", "sum"), dias=("fecha_objetivo", "nunique")))
    weekly["acierto_pct"] = 1 - (weekly["real"] - weekly["proyectado"]) / weekly["real"].where(weekly["real"] != 0)
    weekly["estado"] = weekly["acierto_pct"].map(weekly_status)
    return metrics, daily, weekly


st.set_page_config(page_title="Markov Freedom", page_icon=None, layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>""", unsafe_allow_html=True)
st.title("Markov Freedom | Rendimiento de modelos")
metrics, daily, weekly = load_data()

if daily.empty:
    st.error("No hay predicciones trazables. Ejecute las fases 2-7 primero.")
    st.stop()

models = sorted(daily.modelo.unique())
model = st.sidebar.selectbox("Modelo", ["Todos"] + models)
farms = sorted(daily.finca.unique())
farm = st.sidebar.selectbox("Finca", ["Todas"] + farms)
horizons = sorted(daily.horizonte_dia.dropna().unique()) if "horizonte_dia" in daily else []
horizon = st.sidebar.selectbox("Horizonte", ["Todos"] + horizons)
filtered = daily.copy()
if model != "Todos": filtered = filtered[filtered.modelo.eq(model)]
if farm != "Todas": filtered = filtered[filtered.finca.eq(farm)]
if horizon != "Todos": filtered = filtered[filtered.horizonte_dia.eq(horizon)]

denom = filtered.real.abs().sum()
wape = filtered.error_abs.sum() / denom if denom else float("nan")
accuracy = filtered.acierto_pct.mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("WAPE", f"{wape:.2%}" if pd.notna(wape) else "N.A.")
c2.metric("Acierto relativo medio", f"{accuracy:.2%}" if pd.notna(accuracy) else "N.A.")
c3.metric("MAE", f"{filtered.error_abs.mean():,.0f}" if len(filtered) else "N.A.")
c4.metric("Días trazables", f"{len(filtered):,}")

st.subheader("Proyectado contra real")
chart = filtered.sort_values("fecha_objetivo").set_index("fecha_objetivo")[["real", "proyectado"]]
st.line_chart(chart, height=320)

left, right = st.columns(2)
with left:
    st.subheader("Acierto semanal por finca")
    week = weekly.copy()
    if model != "Todos": week = week[week.modelo.eq(model)]
    if farm != "Todas": week = week[week.finca.eq(farm)]
    st.dataframe(week[["modelo", "finca", "fecha_origen", "real", "proyectado", "acierto_pct", "estado"]]
                 .sort_values("fecha_origen"), use_container_width=True, hide_index=True)
with right:
    st.subheader("Ranking primario")
    ranking = metrics[metrics.comparacion_primaria].sort_values("wape").drop_duplicates("experiment_id")
    st.dataframe(ranking[["experiment_id", "wape", "mae", "rmse", "bias_pct", "n"]],
                 use_container_width=True, hide_index=True)

st.subheader("Detalle diario")
st.download_button("Descargar detalle filtrado CSV", filtered.to_csv(index=False), "detalle_modelos.csv", "text/csv")
if st.checkbox("Mostrar detalle diario", value=False):
    st.dataframe(filtered.sort_values("fecha_objetivo"), use_container_width=True, hide_index=True)

st.info("Acierto = 1 - (real - proyectado) / real. Para real=0 se reporta N.A. Las métricas de modelos sin predicciones trazables no se inventan.")
