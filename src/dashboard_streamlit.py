"""Dashboard interactivo local para explorar el tablero de modelos."""

from pathlib import Path
import json

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from evaluation.metrics import metrics as calculate_metrics
from evaluation.split import temporal_masks
from reporte_excel import (PREDICTION_FILES_FIXED, PREDICTION_FILES_ROLLING,
                           _metrics, _traces, weekly_status)
from models.supervised import feature_groups
from models.supervised import build_supervised_dataset
from models.bayes import HierarchicalNB
from models.selection import estimator_from_spec, read_selection
from canonical import load_config


ROOT = Path(__file__).resolve().parents[1]


def add_weekly_status(weekly):
    weekly = weekly.copy()
    weekly["semana_label"] = weekly["semana_proyeccion"].map(
        lambda value: str(int(value)) if pd.notna(value) else "N.A.")
    weekly["estado_ventana"] = pd.Series(pd.NA, index=weekly.index, dtype="string")
    weekly.loc[weekly.filas_reales.eq(0), "estado_ventana"] = "PENDIENTE_REAL"
    weekly.loc[weekly.filas_reales.eq(weekly.filas_pronosticadas) &
               weekly.estado_fuente.eq("VALIDA"), "estado_ventana"] = "VALIDA"
    weekly.loc[weekly.filas_reales.between(1, weekly.filas_pronosticadas - 1), "estado_ventana"] = "PARCIAL"
    weekly["proyectado_modelo"] = weekly["proyectado"]
    weekly["proyectado_total_modelo"] = weekly["proyectado_total"]
    weekly["proyectado_total"] = np.ceil(weekly["proyectado_total"])
    weekly["proyectado"] = np.ceil(weekly["proyectado"])
    weekly["acierto_pct"] = 1 - (weekly["real"] - weekly["proyectado_modelo"]) / weekly["real"].where(
        weekly.filas_reales.gt(0) & weekly["real"].ne(0))
    weekly["estado"] = weekly["acierto_pct"].map(weekly_status)
    return weekly


def week_label(value):
    return str(int(value)) if pd.notna(value) else "N.A."


def aggregate_weekly(daily, by_block=False):
    keys = ["modelo", "split", "finca", "semana_proyeccion"]
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
        estado_fuente=("estado_ventana", lambda s: "VALIDA" if s.eq("VALIDA").all() else
                       ("PENDIENTE_REAL" if s.eq("PENDIENTE_REAL").all() else "PARCIAL")),
    )
    observed = daily[daily.real.notna()].groupby(keys, as_index=False, dropna=False).agg(
        proyectado=("proyectado_modelo", "sum"))
    weekly = weekly.merge(observed, on=keys, how="left", validate="one_to_one")
    weekly["proyectado"] = weekly["proyectado"].fillna(0.0)
    return add_weekly_status(weekly)


def complete_validation_daily(daily, weekly_view):
    """Conserva solo filas diarias de combinaciones finca-semana completas."""
    key_cols = ["modelo", "split", "finca", "semana_proyeccion"]
    if "bloque" in weekly_view:
        key_cols.insert(2, "bloque")
    valid_keys = weekly_view.loc[
        weekly_view.estado_ventana.eq("VALIDA"), key_cols
    ].drop_duplicates()
    if valid_keys.empty:
        return daily.iloc[0:0].copy()
    return (daily.merge(valid_keys.assign(_valid=True), on=key_cols, how="inner")
            .drop(columns="_valid")
            .dropna(subset=["real"]))


@st.cache_data
def load_training_traces():
    """Genera trazas in-sample para completar la tabla por particion."""
    datasets = ROOT / "outputs/datasets"
    required = ["forecast_windows.parquet", "fact_bloque_dia.parquet",
                "transition_intervals_tradicional.parquet", "poda_features.parquet",
                "clima_features.parquet"]
    if not all((datasets / name).exists() for name in required):
        return []
    cfg = load_config(ROOT / "config/pipeline.yaml")
    windows, fact = (pd.read_parquet(datasets / name) for name in required[:2])
    intervals, pruning, climate = (pd.read_parquet(datasets / name) for name in required[2:])
    frame = build_supervised_dataset(windows, fact, intervals, cfg, pruning, climate)
    train_mask, _ = temporal_masks(frame, cfg, origin_values=windows["fecha_origen"])
    train = pd.Series(train_mask, index=frame.index)
    base = frame.loc[train].copy()
    common = ["finca", "bloque", "fecha_origen", "fecha_objetivo",
              "semana_proyeccion", "horizonte_dia", "target"]
    traces = []

    m3 = base[common + ["M3_pred_bloque"]].rename(
        columns={"target": "real", "M3_pred_bloque": "proyectado"})
    m3["modelo"], m3["split"] = "E00_M3_BASE", "TRAIN"
    traces.append(m3)

    selection_path = ROOT / "outputs/evaluation/selected_model_manifest.json"
    if not selection_path.exists():
        return traces
    selected = read_selection(selection_path)
    group = feature_groups(frame)[selected["features"]]
    cols = [c for c in dict.fromkeys(group) if c in frame.select_dtypes(include=[np.number]).columns]
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = estimator_from_spec(selected, cfg)
    model.fit(x.loc[train], frame.loc[train, "target"])
    selected_trace = base[common].rename(columns={"target": "real"})
    selected_trace["proyectado"] = model.predict(x.loc[train])
    selected_trace["modelo"], selected_trace["split"] = "BEST_COMPOSITE", "TRAIN"
    traces.append(selected_trace)

    bayes = HierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"]).fit(base)
    bayes_trace = base[common].rename(columns={"target": "real"})
    bayes_trace["proyectado"] = bayes.predict(base)
    bayes_trace["modelo"], bayes_trace["split"] = "NB_JERARQUICO", "TRAIN"
    traces.append(bayes_trace)
    return traces


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
        frame["semana_label"] = frame["semana_proyeccion"].map(week_label)
        frame["split"] = "ROLLING" if evaluation_mode == "Rolling origin" else "VALIDATION"
        frame["acierto_pct"] = (1 - (frame["real"] - frame["proyectado_modelo"]) / frame["real"].where(frame["real"] != 0)).astype(float)
        frame["error_abs"] = (frame["proyectado_modelo"] - frame["real"]).abs()
        daily.append(frame)
    training = load_training_traces() if evaluation_mode == "Validación fija" else []
    daily.extend(training)
    for frame in training:
        frame["fecha_objetivo"] = pd.to_datetime(frame["fecha_objetivo"])
        frame["semana_label"] = frame["semana_proyeccion"].map(week_label)
        frame["proyectado_modelo"] = frame["proyectado"]
        frame["acierto_pct"] = (1 - (frame["real"] - frame["proyectado_modelo"]) /
                                 frame["real"].where(frame["real"] != 0)).astype(float)
        frame["error_abs"] = (frame["proyectado_modelo"] - frame["real"]).abs()
    daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    return metrics, daily, aggregate_weekly(daily), aggregate_weekly(daily, by_block=True)


@st.cache_data
def load_hyperparameter_results():
    path = ROOT / "outputs/evaluation/metrics_hyperparametros.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_model_inputs():
    path = ROOT / "outputs/datasets/dataset_supervisado_diario.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    cols = list(dict.fromkeys(c for c in feature_groups(frame)["FENO"]
                             if c in frame.select_dtypes(include=[np.number]).columns
                             and c != "target"))
    base_cols = ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                 "semana_proyeccion", "horizonte_dia", "target"]
    return frame[base_cols + [c for c in cols if c not in base_cols]]


@st.cache_data
def load_diagnostics():
    def read(name):
        path = ROOT / "outputs/evaluation" / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    return {"overfit": read("diagnostico_sobreajuste.csv"),
            "correlations": read("diagnostico_correlaciones_altas.csv"),
            "vif": read("diagnostico_vif.csv"),
            "features": read("diagnostico_features.csv"),
            "importance": read("diagnostico_importancia_horizonte.csv"),
            "leakage": read("diagnostico_leakage.csv"),
            "ablation": read("metrics_rf_ablation_m3.csv")}


st.set_page_config(page_title="Markov Freedom", page_icon=None, layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
[data-testid="stMetric"] {background: #f4f7f8; border: 1px solid #dce5e8; padding: .75rem; border-radius: .65rem;}
[data-testid="stExpander"] {border-color: #dce5e8;}
</style>""", unsafe_allow_html=True)
st.title("Markov Freedom")
st.caption("Centro de validación causal, diagnóstico y trazabilidad de pronósticos")
evaluation_mode = st.sidebar.radio("Evaluación", ["Rolling origin", "Validación fija"])
st.sidebar.info("Rolling origin es la comparación formal; la validación fija conserva el diagnóstico de selección.")
metrics, daily, weekly, weekly_block = load_data(evaluation_mode)
hyperparameter_results = load_hyperparameter_results()

if daily.empty:
    st.error("No hay predicciones trazables. Ejecute las fases 2-7 primero.")
    st.stop()

models = sorted(daily.modelo.unique())
preferred_model = ("MODELO_SELECCIONADO_ROLLING" if evaluation_mode == "Rolling origin"
                   else "BEST_COMPOSITE")
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

# La validación compara únicamente ventanas completas, igual que las métricas
# semanales. Las pendientes y parciales se conservan en la tabla de detalle.
selected_weekly = aggregate_weekly(filtered, by_block=block != "Todos")
evaluation_split = "ROLLING" if evaluation_mode == "Rolling origin" else "VALIDATION"
validation_weekly = selected_weekly[selected_weekly.split.eq(evaluation_split)]
filtered = complete_validation_daily(filtered[filtered.split.eq(evaluation_split)], validation_weekly)

denom = filtered.real.abs().sum()
wape = filtered.error_abs.sum() / denom if denom else float("nan")
accuracy = (1 - (filtered.real.sum() - filtered.proyectado_modelo.sum()) / filtered.real.sum()) if filtered.real.sum() else float("nan")
r2 = calculate_metrics(filtered.real, filtered.proyectado_modelo)["r2"]
weeks_observed = validation_weekly[validation_weekly.estado_ventana.eq("VALIDA")]
weeks_valid = weeks_observed
weeks_partial = validation_weekly[validation_weekly.estado_ventana.eq("PARCIAL")]
hits = int(weeks_observed.acierto_pct.between(.93, 1.07, inclusive="both").sum())
near = int((weeks_observed.acierto_pct.between(.90, .93, inclusive="left") | weeks_observed.acierto_pct.between(1.07, 1.10, inclusive="right")).sum())
miss = int(len(weeks_observed) - hits - near)
pending = int(validation_weekly.estado_ventana.eq("PENDIENTE_REAL").sum())
partial = int(validation_weekly.estado_ventana.eq("PARCIAL").sum())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WAPE", f"{wape:.2%}" if pd.notna(wape) else "N.A.")
c2.metric("Acierto relativo medio", f"{accuracy:.2%}" if pd.notna(accuracy) else "N.A.")
c3.metric("MAE", f"{filtered.error_abs.mean():,.0f}" if len(filtered) else "N.A.")
c4.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "N.A.", help="Coeficiente calculado sobre las filas filtradas. Puede ser negativo.")
c5.metric("Finca-semanas acertadas", f"{hits} / {len(weeks_observed)}")
c5, c6, c7 = st.columns(3)
c5.metric("Cercanas", near); c6.metric("No acertadas", miss); c7.metric("Pendientes / parciales", f"{pending} / {len(weeks_partial)}")
st.caption("WAPE, MAE, R² y los gráficos de validación usan únicamente ventanas finca-semana completas. Las pendientes y parciales no se incluyen.")

st.subheader("Proyectado contra real")
chart_with_week = (filtered.groupby(["fecha_objetivo", "semana_label"], as_index=False)
                   [["real", "proyectado_modelo"]].sum(min_count=1).sort_values("fecha_objetivo"))
chart_for_display = chart_with_week.melt(
    id_vars=["fecha_objetivo", "semana_label"], var_name="serie", value_name="valor")
daily_chart = alt.Chart(chart_for_display).mark_line().encode(
    x=alt.X("fecha_objetivo:T", title="Día"),
    y=alt.Y("valor:Q", title="Cantidad"),
    color=alt.Color("serie:N", title=None),
    tooltip=[
        alt.Tooltip("fecha_objetivo:T", title="Día", format="%a %d %b %Y"),
        alt.Tooltip("semana_label:N", title="Semana"),
        alt.Tooltip("serie:N", title="Serie"),
        alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
    ],
).properties(height=320)
st.altair_chart(daily_chart, use_container_width=True)
st.caption("El eje muestra el día; al pasar el cursor se muestran también el día y la semana en formato YYYYWW.")
chart = (chart_with_week.groupby("fecha_objetivo", as_index=True)
         [["real", "proyectado_modelo"]].sum(min_count=1)
         .rename(columns={"proyectado_modelo": "proyectado"}))
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

st.subheader("Pronostico semanal: validacion")
weekly_view = selected_weekly
complete_weeks = weekly_view[weekly_view.estado_ventana.eq("VALIDA")]
if len(complete_weeks):
    weekly_scores = calculate_metrics(complete_weeks.real, complete_weeks.proyectado_modelo)
    w1, w2, w3, w4 = st.columns(4)
    weekly_wape = (complete_weeks.proyectado_modelo - complete_weeks.real).abs().sum() / complete_weeks.real.abs().sum()
    w1.metric("WAPE semanal", f"{weekly_wape:.2%}")
    w2.metric("R² semanal", f"{weekly_scores['r2']:.3f}")
    w3.metric("MAE semanal", f"{weekly_scores['mae']:,.0f}")
    w4.metric("Semanas calendario completas", complete_weeks.semana_proyeccion.nunique())
    st.caption(f"Combinaciones finca-semana completas: {len(complete_weeks)}")
weekly_chart = (complete_weeks.groupby(["semana_proyeccion", "semana_label"], as_index=True)
                [["real", "proyectado_modelo"]].sum(min_count=1).sort_index())
weekly_chart.index = weekly_chart.index.get_level_values("semana_label")
weekly_chart = weekly_chart.rename(columns={"proyectado_modelo": "proyectado"})
st.line_chart(weekly_chart, height=320)

left, right = st.columns(2)
with left:
    st.subheader("Acierto semanal por finca")
    week = selected_weekly
    st.caption("Se muestran todas las semanas de TRAIN y VALIDATION; el split identifica cada fila.")
    table_cols = ["modelo", "split", "finca"] + (["bloque"] if "bloque" in week else []) + [
        "semana_label", "fecha_origen", "real", "proyectado", "proyectado_total",
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
    default_hyper = hyperparameter_results.sort_values("weekly_wape").iloc[0]["model"]
    hyper_models = hyperparameter_results["model"].tolist()
    selected_hyper_model = st.selectbox("Modelo de hiperparametros", hyper_models,
                                        index=hyper_models.index(default_hyper))
    selected_hyper = hyperparameter_results[
        hyperparameter_results.model.eq(selected_hyper_model)].iloc[0]
    st.subheader("Detalle del modelo seleccionado de hiperparametros")
    st.caption(f"Seleccionado por: {selected_hyper_model}; validacion fija. No sustituye el ranking formal de Fase 8.")
    info_cols = st.columns(6)
    info_cols[0].metric("Modelo", str(selected_hyper["model"]))
    info_cols[1].metric("Variables", str(selected_hyper["features"]))
    info_cols[2].metric("WAPE diario", f"{selected_hyper['daily_wape']:.2%}")
    info_cols[3].metric("R2 diario", f"{selected_hyper['daily_r2']:.3f}")
    info_cols[4].metric("WAPE semanal", f"{selected_hyper['weekly_wape']:.2%}")
    info_cols[5].metric("R2 semanal", f"{selected_hyper['weekly_r2']:.3f}")
    parameters = json.loads(selected_hyper["hyperparameters"])
    parameters = pd.DataFrame([{"parametro": key, "valor": value}
                               for key, value in parameters.items()])
    st.dataframe(parameters, use_container_width=True, hide_index=True)

st.subheader("Variables de entrada y diagnóstico")
inputs = load_model_inputs()
diagnostics = load_diagnostics()
if inputs.empty:
    st.info("No existe el dataframe supervisado. Ejecute fase6_supervisado.py.")
else:
    input_tab, comparison_tab, health_tab, collinearity_tab, quality_tab = st.tabs([
        "Variables de entrada", "RF con / sin M3", "Sobreajuste", "Colinealidad", "Calidad y explicación"])
    with input_tab:
        input_weeks = sorted(inputs.semana_proyeccion.dropna().unique())
        selected_input_week = st.selectbox("Semana de origen / proyección", ["Todas"] + input_weeks,
                                           key="input_week")
        input_view = inputs.copy()
        if selected_input_week != "Todas":
            input_view = input_view[input_view.semana_proyeccion.eq(selected_input_week)]
        if farm != "Todas":
            input_view = input_view[input_view.finca.eq(farm)]
        if block != "Todos":
            input_view = input_view[input_view.bloque.eq(block)]
        if horizon != "Todos":
            input_view = input_view[input_view.horizonte_dia.eq(horizon)]
        st.caption("target es el corte real observado y no es una variable de entrada.")
        st.download_button("Descargar variables de entrada CSV", input_view.to_csv(index=False),
                           "variables_entrada_rf.csv", "text/csv", key="download_inputs")
        st.dataframe(input_view, use_container_width=True, hide_index=True)

    with comparison_tab:
        ablation = diagnostics["ablation"]
        if ablation.empty:
            st.info("Ejecute ablation_rf_m3.py para generar la comparación.")
        else:
            base = metrics[(metrics.experiment_id == "RF_H1_H7_FENO") & metrics.split.eq("VALIDATION")]
            no_m3 = ablation[(ablation.experiment_id == "RF_H1_H7_FENO_SIN_M3") &
                             ablation.horizonte.astype(str).eq("TODOS")]
            comparison = pd.DataFrame([
                {"modelo": "RF H1-H7 con M3", "wape": base.wape.iloc[0] if len(base) else np.nan},
                {"modelo": "RF H1-H7 sin M3", "wape": no_m3.wape.iloc[0] if len(no_m3) else np.nan},
                {"modelo": "M3 baseline", "wape": metrics.loc[metrics.experiment_id.eq("E00_M3_BASE"), "wape"].iloc[0]
                 if metrics.experiment_id.eq("E00_M3_BASE").any() else np.nan},
            ])
            chart = alt.Chart(comparison).mark_bar().encode(
                x=alt.X("wape:Q", title="WAPE", axis=alt.Axis(format=".0%")),
                y=alt.Y("modelo:N", sort="-x", title=None),
                color=alt.Color("modelo:N", legend=None),
                tooltip=["modelo", alt.Tooltip("wape:Q", format=".2%")],
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)
            horizon_days = int(load_config(ROOT / "config/pipeline.yaml")["forecast"]["horizon_days"])
            no_m3_rows = ablation[ablation.horizonte.astype(str).isin(
                [str(i) for i in range(1, horizon_days + 1)])]
            with st.expander("Comparación por horizonte"):
                st.dataframe(no_m3_rows, use_container_width=True, hide_index=True)
            st.caption("Esta comparación usa el mismo split temporal y los mismos hiperparámetros. La variante sin M3 fue reentrenada, no simulada eliminando columnas.")

    if diagnostics["overfit"].empty:
        st.info("Ejecute diagnostico_modelos.py para cargar sobreajuste, colinealidad y leakage.")
    else:
        with health_tab:
            overfit = diagnostics["overfit"]
            validation = overfit[overfit.scope.eq("VALIDATION")]
            high_risk = int(validation.riesgo_sobreajuste.eq("ALTO").sum())
            medium_risk = int(validation.riesgo_sobreajuste.eq("MEDIO").sum())
            h1, h2, h3 = st.columns(3)
            h1.metric("Horizontes alto riesgo", high_risk)
            h2.metric("Horizontes riesgo medio", medium_risk)
            h3.metric("Mayor gap WAPE", f"{validation.gap_wape.max():.1%}")
            plot = overfit.copy()
            plot["horizonte_label"] = "H" + plot.horizonte.astype(str)
            chart = alt.Chart(plot).mark_bar().encode(
                x=alt.X("horizonte_label:N", title="Horizonte"),
                y=alt.Y("wape:Q", title="WAPE", axis=alt.Axis(format=".0%")),
                color=alt.Color("scope:N", title="Muestra"),
                tooltip=["horizonte_label", "scope", alt.Tooltip("wape:Q", format=".2%"),
                         alt.Tooltip("r2:Q", format=".3f"), "riesgo_sobreajuste"],
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            gap_chart = alt.Chart(validation).mark_bar().encode(
                x=alt.X("horizonte:N", title="Horizonte"),
                y=alt.Y("gap_wape:Q", title="Gap WAPE", axis=alt.Axis(format=".0%")),
                color=alt.Color("riesgo_sobreajuste:N", scale=alt.Scale(
                    domain=["BAJO", "MEDIO", "ALTO"], range=["#2e8b57", "#d99b23", "#c94c4c"]),
                    title="Riesgo"),
                tooltip=["horizonte", alt.Tooltip("gap_wape:Q", format=".2%"), "riesgo_sobreajuste"],
            ).properties(height=220)
            st.altair_chart(gap_chart, use_container_width=True)
            st.caption("El gap se interpreta junto con el tamaño de muestra y el comportamiento temporal; no constituye una prueba aislada de sobreajuste.")
            with st.expander("Detalle numérico train-validation"):
                st.dataframe(overfit, use_container_width=True, hide_index=True)
        with collinearity_tab:
            st.caption("La colinealidad no invalida un Random Forest, pero sí puede repartir la importancia entre variables redundantes.")
            vif = diagnostics["vif"].head(15).sort_values("vif")
            vif_chart = alt.Chart(vif).mark_bar().encode(
                x=alt.X("vif:Q", title="VIF", scale=alt.Scale(type="log")),
                # Una escala logarítmica no admite el cero que usaría por defecto
                # una barra; VIF=1 es la base natural del diagnóstico.
                x2=alt.X2(datum=1),
                y=alt.Y("variable:N", sort="-x", title=None),
                color=alt.Color("riesgo:N", scale=alt.Scale(
                    domain=["BAJO", "MEDIO", "ALTO"], range=["#2e8b57", "#d99b23", "#c94c4c"]), title="Riesgo"),
                tooltip=["variable", "vif", "riesgo"],
            ).properties(height=420)
            st.altair_chart(vif_chart, use_container_width=True)
            top_variables = vif.variable.tolist()[:12]
            corr_values = inputs[top_variables].corr().reindex(
                index=top_variables, columns=top_variables)
            corr_array = np.array(corr_values, dtype=float, copy=True)
            np.fill_diagonal(corr_array, 1.0)
            corr_values = pd.DataFrame(
                corr_array, index=top_variables, columns=top_variables)
            # La diagonal representa cada variable consigo misma y debe ser
            # exactamente 1, incluso con columnas casi constantes o redondeos.
            corr = (corr_values.rename_axis("variable_1").reset_index()
                    .melt(id_vars="variable_1", var_name="variable_2",
                          value_name="correlacion"))
            heatmap = alt.Chart(corr).mark_rect().encode(
                x=alt.X("variable_2:N", title=None, sort=top_variables),
                y=alt.Y("variable_1:N", title=None, sort=top_variables),
                color=alt.Color("correlacion:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                                 title="r"),
                tooltip=["variable_1", "variable_2", alt.Tooltip("correlacion:Q", format=".3f")],
            ).properties(height=420)
            st.altair_chart(heatmap, use_container_width=True)
            with st.expander("Detalle de pares correlacionados"):
                st.dataframe(diagnostics["correlations"].head(50), use_container_width=True, hide_index=True)
        with quality_tab:
            features = diagnostics["features"].copy()
            top_missing = features.sort_values("faltantes_pct", ascending=False).head(15)
            missing_chart = alt.Chart(top_missing).mark_bar().encode(
                x=alt.X("faltantes_pct:Q", title="Faltantes", axis=alt.Axis(format=".0%")),
                y=alt.Y("variable:N", sort="-x", title=None),
                color=alt.Color("faltantes_pct:Q", scale=alt.Scale(scheme="yelloworangered"), title="%"),
                tooltip=["variable", alt.Tooltip("faltantes_pct:Q", format=".2%"), "unicos"],
            ).properties(height=360)
            st.altair_chart(missing_chart, use_container_width=True)
            if not diagnostics["importance"].empty:
                selected_importance = diagnostics["importance"]
                if horizon != "Todos":
                    selected_importance = selected_importance[selected_importance.horizonte.eq(horizon)]
                selected_importance = selected_importance.head(20).sort_values("importance")
                importance_chart = alt.Chart(selected_importance).mark_bar().encode(
                    x=alt.X("importance:Q", title="Importancia estructural"),
                    y=alt.Y("variable:N", sort="-x", title=None),
                    color=alt.Color("horizonte:N", title="Horizonte"),
                    tooltip=["variable", "horizonte", "importance"],
                ).properties(height=420)
                st.altair_chart(importance_chart, use_container_width=True)
            leakage = diagnostics["leakage"]
            leakage_ok = leakage.resultado.eq("OK").all()
            st.success("Auditoría básica de leakage: OK") if leakage_ok else st.error("Auditoría básica de leakage: revisar")
            with st.expander("Detalle de calidad y leakage"):
                st.dataframe(features, use_container_width=True, hide_index=True)
                st.dataframe(leakage, use_container_width=True, hide_index=True)

st.subheader("Detalle diario")
st.download_button("Descargar detalle filtrado CSV", filtered.to_csv(index=False), "detalle_modelos.csv", "text/csv")
if st.checkbox("Mostrar detalle diario", value=False):
    st.dataframe(filtered.sort_values("fecha_objetivo"), use_container_width=True, hide_index=True)

st.info("Acierto = 1 - (real - proyectado) / real. Para real=0 se reporta N.A. Las métricas de modelos sin predicciones trazables no se inventan.")
