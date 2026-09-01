"""Dashboard interactivo local para explorar el tablero de modelos."""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from evaluation.metrics import metrics as calculate_metrics
from reporte_excel import (PREDICTION_FILES_FIXED, PREDICTION_FILES_ROLLING,
                           _metrics, _traces, weekly_status)


ROOT = Path(__file__).resolve().parents[1]


def add_weekly_status(weekly):
    weekly = weekly.copy()
    weekly["estado_ventana"] = pd.Series(pd.NA, index=weekly.index, dtype="string")
    weekly.loc[weekly.filas_reales.eq(0), "estado_ventana"] = "PENDIENTE_REAL"
    weekly.loc[weekly.filas_reales.eq(weekly.filas_pronosticadas), "estado_ventana"] = "VALIDA"
    weekly.loc[weekly.filas_reales.between(1, weekly.filas_pronosticadas - 1), "estado_ventana"] = "PARCIAL"
    weekly["proyectado_modelo"] = weekly["proyectado"]
    weekly["proyectado_total_modelo"] = weekly["proyectado_total"]
    weekly["proyectado_total"] = np.ceil(weekly["proyectado_total"])
    weekly["proyectado"] = np.ceil(weekly["proyectado"])
    weekly["acierto_pct"] = 1 - (weekly["real"] - weekly["proyectado_modelo"]) / weekly["real"].where(
        weekly.filas_reales.gt(0) & weekly["real"].ne(0))
    weekly["estado"] = weekly["acierto_pct"].map(weekly_status)
    return weekly


def aggregate_weekly(daily, by_block=False):
    keys = ["modelo", "finca", "semana_proyeccion"]
    if by_block:
        keys.insert(2, "bloque")
    grouped = daily.groupby(keys, as_index=False, dropna=False)
    weekly = grouped.agg(
        fecha_origen=("fecha_origen", "min"),
        real=("real", lambda s: s.sum(min_count=1)),
        proyectado_total=("proyectado_modelo", "sum"),
        dias=("fecha_objetivo", "nunique"),
        filas_pronosticadas=("real", "size"),
        filas_reales=("real", "count"),
    )
    observed = daily[daily.real.notna()].groupby(keys, as_index=False, dropna=False).agg(
        proyectado=("proyectado_modelo", "sum"))
    weekly = weekly.merge(observed, on=keys, how="left", validate="one_to_one")
    weekly["proyectado"] = weekly["proyectado"].fillna(0.0)
    return add_weekly_status(weekly)


@st.cache_data
def load_data(evaluation_mode):
    metrics = _metrics(ROOT)
    files = PREDICTION_FILES_FIXED if evaluation_mode == "Validación fija" else PREDICTION_FILES_ROLLING
    traces = _traces(ROOT, files)
    daily = []
    for model, frame in traces.items():
        frame = frame.copy()
        frame["fecha_objetivo"] = pd.to_datetime(frame["fecha_objetivo"])
        if "semana_proyeccion" not in frame:
            frame["semana_proyeccion"] = frame["fecha_objetivo"].dt.isocalendar().year * 100 + frame["fecha_objetivo"].dt.isocalendar().week
        frame["acierto_pct"] = (1 - (frame["real"] - frame["proyectado_modelo"]) / frame["real"].where(frame["real"] != 0)).astype(float)
        frame["error_abs"] = (frame["proyectado_modelo"] - frame["real"]).abs()
        daily.append(frame)
    daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    return metrics, daily, aggregate_weekly(daily), aggregate_weekly(daily, by_block=True)


@st.cache_data
def load_hyperparameter_results():
    path = ROOT / "outputs/evaluation/metrics_hyperparametros.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


st.set_page_config(page_title="Markov Freedom", page_icon=None, layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>""", unsafe_allow_html=True)
st.title("Markov Freedom | Rendimiento de modelos")
evaluation_mode = st.sidebar.radio("Evaluación", ["Rolling-origin", "Validación fija"], index=0,
                                   help="No mezclar métricas: la validación fija compara 714 registros; rolling usa todos los orígenes causales posibles.")
metrics, daily, weekly, weekly_block = load_data(evaluation_mode)
hyperparameter_results = load_hyperparameter_results()

if daily.empty:
    st.error("No hay predicciones trazables. Ejecute las fases 2-7 primero.")
    st.stop()

models = sorted(daily.modelo.unique())
preferred_model = ("BEST_WEEKLY_WAPE" if evaluation_mode == "Validación fija"
                   else "RF_H1_H7_FENO")
default_model = preferred_model if preferred_model in models else models[0]
model = st.sidebar.selectbox("Modelo", models + ["Todos"], index=models.index(default_model))
farms = sorted(daily.finca.unique())
farm = st.sidebar.selectbox("Finca", ["Todas"] + farms)
blocks = sorted(daily.loc[daily.finca.eq(farm), "bloque"].dropna().unique()) if farm != "Todas" else []
block = st.sidebar.selectbox("Bloque", ["Todos"] + blocks, disabled=farm == "Todas")
horizons = sorted(daily.horizonte_dia.dropna().unique()) if "horizonte_dia" in daily else []
horizon = st.sidebar.selectbox("Horizonte", ["Todos"] + horizons)
filtered = daily.copy()
if model != "Todos": filtered = filtered[filtered.modelo.eq(model)]
if farm != "Todas": filtered = filtered[filtered.finca.eq(farm)]
if block != "Todos": filtered = filtered[filtered.bloque.eq(block)]
if horizon != "Todos": filtered = filtered[filtered.horizonte_dia.eq(horizon)]

denom = filtered.real.abs().sum()
wape = filtered.error_abs.sum() / denom if denom else float("nan")
accuracy = (1 - (filtered.real.sum() - filtered.proyectado_modelo.sum()) / filtered.real.sum()) if filtered.real.sum() else float("nan")
r2 = calculate_metrics(filtered.real, filtered.proyectado_modelo)["r2"]
c_week = weekly_block.copy() if block != "Todos" else weekly.copy()
if model != "Todos": c_week = c_week[c_week.modelo.eq(model)]
if farm != "Todas": c_week = c_week[c_week.finca.eq(farm)]
weeks_observed = c_week[c_week.filas_reales.gt(0)]
weeks_valid = c_week[c_week.estado_ventana.eq("VALIDA")]
weeks_partial = c_week[c_week.estado_ventana.eq("PARCIAL")]
hits = int(weeks_observed.acierto_pct.between(.93, 1.07, inclusive="both").sum())
near = int((weeks_observed.acierto_pct.between(.90, .93, inclusive="left") | weeks_observed.acierto_pct.between(1.07, 1.10, inclusive="right")).sum())
miss = int(len(weeks_observed) - hits - near)
pending = int(c_week.estado_ventana.eq("PENDIENTE_REAL").sum())
partial = int(c_week.estado_ventana.eq("PARCIAL").sum())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WAPE", f"{wape:.2%}" if pd.notna(wape) else "N.A.")
c2.metric("Acierto relativo medio", f"{accuracy:.2%}" if pd.notna(accuracy) else "N.A.")
c3.metric("MAE", f"{filtered.error_abs.mean():,.0f}" if len(filtered) else "N.A.")
c4.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "N.A.", help="Coeficiente calculado sobre las filas filtradas. Puede ser negativo.")
c5.metric("Semanas acertadas", f"{hits} / {len(weeks_observed)}")
c5, c6, c7 = st.columns(3)
c5.metric("Cercanas", near); c6.metric("No acertadas", miss); c7.metric("Pendientes / parciales", f"{pending} / {len(weeks_partial)}")

st.subheader("Proyectado contra real")
chart = (filtered.groupby("fecha_objetivo", as_index=True)[["real", "proyectado_modelo"]].sum(min_count=1)
          .sort_index())
chart = chart.rename(columns={"proyectado_modelo": "proyectado"})
st.line_chart(chart, height=320)
residual = (chart["proyectado"] - chart["real"]).rename("residuo").dropna()
st.subheader("Residuo diario: picos y valles")
st.bar_chart(residual, height=180)
worst = (filtered.assign(residuo=filtered.proyectado_modelo - filtered.real)
         .dropna(subset=["real"])
         .assign(error_abs=lambda x: x.residuo.abs())
         .sort_values("error_abs", ascending=False)
         [["fecha_objetivo", "finca", "bloque", "horizonte_dia", "real", "proyectado_modelo", "residuo"]]
         .head(10))
st.caption("H1-H7 es la posición del día dentro de la ventana semanal. Se grafican valores sin redondear y se agregan solo los bloques de la selección.")
with st.expander("Mayores errores diarios"):
    st.dataframe(worst, use_container_width=True, hide_index=True)

st.subheader("Pronostico semanal: todas las semanas")
weekly_view = weekly_block.copy() if block != "Todos" else weekly.copy()
if model != "Todos":
    weekly_view = weekly_view[weekly_view.modelo.eq(model)]
if farm != "Todas":
    weekly_view = weekly_view[weekly_view.finca.eq(farm)]
if block != "Todos" and "bloque" in weekly_view:
    weekly_view = weekly_view[weekly_view.bloque.eq(block)]
complete_weeks = weekly_view[weekly_view.estado_ventana.eq("VALIDA")]
if len(complete_weeks):
    weekly_scores = calculate_metrics(complete_weeks.real, complete_weeks.proyectado_modelo)
    w1, w2, w3, w4 = st.columns(4)
    weekly_wape = (complete_weeks.proyectado_modelo - complete_weeks.real).abs().sum() / complete_weeks.real.abs().sum()
    w1.metric("WAPE semanal", f"{weekly_wape:.2%}")
    w2.metric("R² semanal", f"{weekly_scores['r2']:.3f}")
    w3.metric("MAE semanal", f"{weekly_scores['mae']:,.0f}")
    w4.metric("Semanas completas", len(complete_weeks))
weekly_chart = (weekly_view.groupby("semana_proyeccion", as_index=True)[["real", "proyectado_modelo"]]
                .sum(min_count=1).rename(columns={"proyectado_modelo": "proyectado"}).sort_index())
st.line_chart(weekly_chart, height=320)

left, right = st.columns(2)
with left:
    st.subheader("Acierto semanal por finca")
    week = weekly_block.copy() if block != "Todos" else weekly.copy()
    if model != "Todos": week = week[week.modelo.eq(model)]
    if farm != "Todas": week = week[week.finca.eq(farm)]
    if block != "Todos" and "bloque" in week: week = week[week.bloque.eq(block)]
    st.caption("Se muestran todas las semanas; las semanas parciales se identifican por separado.")
    table_cols = ["modelo", "finca"] + (["bloque"] if "bloque" in week else []) + [
        "semana_proyeccion", "fecha_origen", "real", "proyectado", "proyectado_total",
        "acierto_pct", "estado_ventana", "estado"]
    st.dataframe(week[table_cols].sort_values("fecha_origen"), use_container_width=True, hide_index=True)
with right:
    st.subheader("Ranking primario")
    ranking = metrics[metrics.comparacion_primaria].sort_values("wape").drop_duplicates("experiment_id")
    st.dataframe(ranking[["experiment_id", "wape", "mae", "rmse", "r2", "bias_pct", "n"]],
                 use_container_width=True, hide_index=True)
    if not hyperparameter_results.empty:
        st.subheader("Ranking exploratorio semanal")
        weekly_ranking = hyperparameter_results.sort_values("weekly_wape").head(10)
        st.dataframe(weekly_ranking[["model", "features", "weekly_wape", "weekly_r2",
                                     "weekly_mae", "weekly_rmse", "weekly_bias_pct",
                                     "selection_score"]], use_container_width=True, hide_index=True)

if not hyperparameter_results.empty:
    best_hyper = hyperparameter_results.sort_values("weekly_wape").iloc[0]
    st.subheader("Mejor modelo de la busqueda de hiperparametros")
    st.caption("Seleccionado por WAPE semanal en validacion fija; no sustituye el ranking formal de Fase 8.")
    info_cols = st.columns(6)
    info_cols[0].metric("Modelo", str(best_hyper["model"]))
    info_cols[1].metric("Variables", str(best_hyper["features"]))
    info_cols[2].metric("WAPE diario", f"{best_hyper['daily_wape']:.2%}")
    info_cols[3].metric("R2 diario", f"{best_hyper['daily_r2']:.3f}")
    info_cols[4].metric("WAPE semanal", f"{best_hyper['weekly_wape']:.2%}")
    info_cols[5].metric("R2 semanal", f"{best_hyper['weekly_r2']:.3f}")
    parameters = pd.DataFrame([{
        "modelo": best_hyper["model"],
        "n_estimators": best_hyper["n_estimators"],
        "max_depth": best_hyper["max_depth"],
        "min_samples_leaf": best_hyper["min_samples_leaf"],
        "min_samples_split": best_hyper["min_samples_split"],
        "max_features": best_hyper["max_features"],
        "criterion": best_hyper["criterion"],
    }])
    st.dataframe(parameters, use_container_width=True, hide_index=True)

st.subheader("Detalle diario")
st.download_button("Descargar detalle filtrado CSV", filtered.to_csv(index=False), "detalle_modelos.csv", "text/csv")
if st.checkbox("Mostrar detalle diario", value=False):
    st.dataframe(filtered.sort_values("fecha_objetivo"), use_container_width=True, hide_index=True)

st.info("Acierto = 1 - (real - proyectado) / real. Para real=0 se reporta N.A. Las métricas de modelos sin predicciones trazables no se inventan.")
