"""Dashboard tecnico de validacion semanal, incertidumbre y diagnosticos."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from canonical import load_config
from dashboard_validation import (complete_windows, load_validation_predictions, model_scores,
                                  operational_summary, operational_weekly, operational_weekly_selected,
                                  weekly_status)
from evaluacion_entrada import evaluate_input
from reporte_excel import PREDICTION_FILES, PREDICTION_FILES_ROLLING


ROOT = Path(__file__).resolve().parents[1]
FIXED_VALIDATION_FILES = {
    **PREDICTION_FILES,
    "E00_M3_BASE": "outputs/predictions/E00_M3_BASE.csv",
    "RF_H1_H7_FENO_SIN_M3": "outputs/evaluation/predictions_rf_h1_h7_feno_sin_m3.csv",
}
BAYES_FILES = {
    "NB_JERARQUICO": "outputs/evaluation/predictions_nb_jerarquico.csv",
    "NB_JERARQUICO_COVARIABLES": "outputs/evaluation/predictions_nb_jerarquico_covariables.csv",
    "M3_DIRICHLET_MULTINOMIAL": "outputs/evaluation/predictions_m3_dirichlet.csv",
}


def _pct(value, signed=False):
    if pd.isna(value):
        return "N.A."
    return f"{value:+.2%}" if signed else f"{value:.2%}"


@st.cache_data
def load_data(evaluation_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(ROOT / "config/pipeline.yaml")
    if evaluation_mode == "Rolling origin":
        files, split = PREDICTION_FILES_ROLLING, "ROLLING_VALIDATION"
    else:
        files, split = FIXED_VALIDATION_FILES, "FIXED_VALIDATION"
    daily = load_validation_predictions(ROOT, files, split)
    if daily.empty:
        return pd.DataFrame(), daily
    return complete_windows(daily, int(cfg["forecast"]["horizon_days"]))


@st.cache_data
def load_bayesian_weekly() -> pd.DataFrame:
    path = ROOT / "outputs/evaluation/bayes_weekly_intervals.csv"
    return pd.read_csv(path, parse_dates=["fecha_origen"]) if path.exists() else pd.DataFrame()


@st.cache_data
def load_bayesian_draws() -> pd.DataFrame:
    path = ROOT / "outputs/evaluation/bayes_posterior_draws.csv"
    return pd.read_csv(path, parse_dates=["fecha_origen"]) if path.exists() else pd.DataFrame()


@st.cache_data
def load_diagnostics() -> dict[str, pd.DataFrame]:
    names = {
        "overfit": "diagnostico_sobreajuste.csv",
        "importance": "diagnostico_importancia_horizonte.csv",
        "pairs": "diagnostico_correlaciones_altas.csv",
        "lag_pairs": "diagnostico_correlaciones_rezagos.csv",
        "lag_target": "diagnostico_correlacion_target_rezagos.csv",
        "vif": "diagnostico_vif.csv",
        "features": "diagnostico_features.csv",
        "leakage": "diagnostico_leakage.csv",
        "bayes_metrics": "metrics_fase7_bayes.csv",
    }
    result = {}
    for key, name in names.items():
        path = ROOT / "outputs/evaluation" / name
        result[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return result


@st.cache_data
def load_manifests() -> dict[str, dict]:
    result = {}
    for key, name in {
        "selected": "selected_model_manifest.json",
        "champion": "champion_manifest.json",
        "diagnostic": "diagnostico_manifest.json",
    }.items():
        path = ROOT / "outputs/evaluation" / name
        if path.exists():
            result[key] = json.loads(path.read_text(encoding="utf-8"))
    return result


@st.cache_data
def load_rf_optimization() -> tuple[pd.DataFrame, dict]:
    evaluation = ROOT / "outputs/evaluation"
    results_path = evaluation / "rf_mejores_por_grupo.csv"
    plan_path = evaluation / "rf_hardware_plan.json"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    return results, plan


@st.cache_data
def load_input_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = ROOT / "outputs/evaluation"
    daily_path = evaluation / "predictions_entrada_diarias.csv"
    weekly_path = evaluation / "predictions_entrada_semanales.csv"
    if not daily_path.exists() or not weekly_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(daily_path, parse_dates=["fecha_origen", "fecha_objetivo"]), pd.read_csv(weekly_path, parse_dates=["fecha_origen"])


def apply_filters(frame: pd.DataFrame, farm: str, block: str) -> pd.DataFrame:
    result = frame
    if farm != "Todas":
        result = result[result["finca"].eq(farm)]
    if block != "Todos":
        result = result[result["bloque"].astype(str).eq(block)]
    return result


def weekly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    result = model_scores(frame).rename(columns={
        "wape": "wape_semanal", "mae": "mae_semanal", "rmse": "rmse_semanal",
        "bias_pct": "sesgo_pct", "r2": "r2_semanal",
    })
    if result.empty:
        return result
    result["desviacion_abs_pct"] = result["wape_semanal"]
    ratios = []
    for model_name, group in frame.groupby("modelo"):
        real_total = group["real"].sum()
        ratios.append({"modelo": model_name,
                       "ratio_pred_real": group["proyectado_modelo"].sum() / real_total
                       if real_total else np.nan})
    ratios = pd.DataFrame(ratios)
    result = result.merge(ratios, on="modelo", how="left")
    result = result.drop(columns=["acierto_global"], errors="ignore")
    result["poblacion"] = np.where(
        result["modelo"].astype(str).str.match(r"E0[2-7]_"),
        "RETROSPECTIVO_ORACLE_NO_CAUSAL", "CAUSAL")
    return result


def aggregate_bayesian_intervals(weekly: pd.DataFrame, draws: pd.DataFrame, model: str, farm: str, block: str) -> pd.DataFrame:
    """Calcula cuantiles de sumas de draws para el filtro actual, nunca sumas de cuantiles."""
    filtered_weekly = apply_filters(weekly[weekly.model.eq(model)], farm, block)
    filtered_draws = apply_filters(draws[draws.model.eq(model)], farm, block)
    if filtered_weekly.empty or filtered_draws.empty:
        return pd.DataFrame()
    real = filtered_weekly.groupby("semana_proyeccion", as_index=False).real.sum()
    totals = filtered_draws.groupby(["semana_proyeccion", "draw"], as_index=False)["sample"].sum()
    rows = []
    for week, group in totals.groupby("semana_proyeccion"):
        samples = group["sample"].to_numpy(float)
        actual = float(real.loc[real.semana_proyeccion.eq(week), "real"].iloc[0])
        rows.append({"semana_proyeccion": week, "real": actual, "pred": samples.mean(),
                     "low80": np.quantile(samples, .1), "high80": np.quantile(samples, .9),
                     "low95": np.quantile(samples, .025), "high95": np.quantile(samples, .975)})
    result = pd.DataFrame(rows).sort_values("semana_proyeccion")
    result["coverage80"] = ((result.real >= result.low80) & (result.real <= result.high80)).astype(float)
    result["coverage95"] = ((result.real >= result.low95) & (result.real <= result.high95)).astype(float)
    result["width80"] = result.high80 - result.low80
    result["width95"] = result.high95 - result.low95
    return result


st.set_page_config(page_title="Markov Freedom", page_icon=None, layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stMetric"] {background: #f4f7f8; border: 1px solid #dce5e8; padding: .65rem; border-radius: .65rem;}
</style>""", unsafe_allow_html=True)
st.title("Markov Freedom")
st.caption("Centro tecnico de validacion semanal, incertidumbre, diagnostico y trazabilidad")

evaluation_mode = st.sidebar.radio("Evaluacion", ["Rolling origin", "Validacion fija"])
weekly, daily = load_data(evaluation_mode)
diagnostics = load_diagnostics()
manifests = load_manifests()
bayes = load_bayesian_weekly()
bayes_draws = load_bayesian_draws()
rf_best, hardware_plan = load_rf_optimization()
if weekly.empty:
    st.error("No hay ventanas semanales completas para esta evaluacion.")
    st.stop()

models = sorted(weekly["modelo"].unique())
preferred = "MODELO_SELECCIONADO_ROLLING" if evaluation_mode == "Rolling origin" else "E00_M3_BASE"
model = st.sidebar.selectbox("Modelo", models, index=models.index(preferred) if preferred in models else 0)
farms = sorted(weekly["finca"].dropna().unique())
farm = st.sidebar.selectbox("Finca", ["Todas"] + farms)
available_blocks = weekly if farm == "Todas" else weekly[weekly["finca"].eq(farm)]
blocks = sorted(available_blocks["bloque"].dropna().astype(str).unique())
block = st.sidebar.selectbox("Bloque", ["Todos"] + blocks, disabled=not blocks)

filtered_weekly = apply_filters(weekly, farm, block)
model_weekly = filtered_weekly[filtered_weekly["modelo"].eq(model)].copy()
if model_weekly.empty:
    st.warning("No hay ventanas para los filtros seleccionados.")
    st.stop()

score = model_scores(model_weekly).iloc[0]
weekly_wape = float(score["wape"])
sesgo = float(score["bias_pct"])
summary = weekly_summary(filtered_weekly)
operational = operational_weekly_selected(daily, by_block=block != "Todos")
operational = apply_filters(operational, farm, block)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("WAPE semanal", _pct(weekly_wape))
c2.metric("Sesgo porcentual", _pct(sesgo, signed=True), help="Positivo sobreestima; negativo subestima; 0% es exacto.")
c3.metric("MAE semanal", f"{score['mae']:,.0f}")
c4.metric("RMSE semanal", f"{score['rmse']:,.0f}")
c5.metric("R2 semanal", f"{score['r2']:.3f}" if pd.notna(score["r2"]) else "N.A.")
c6.metric("Ventanas", f"{len(model_weekly):,}")
st.caption(f"{model}: {len(model_weekly)} ventanas bloque-origen completas; split {model_weekly['split'].iloc[0]}.")
if evaluation_mode == "Validacion fija" and weekly["modelo"].astype(str).str.match(r"E0[2-7]_").any():
    st.warning("Los modelos E02-E07 usan informacion P32 retrospectiva y no son causales. Se muestran para auditoria, pero no deben compararse como champion causal.")

with st.expander("Definiciones y lectura metodologica"):
    st.markdown("""
    - **WAPE:** suma de errores absolutos dividida por la suma de reales.
    - **Sesgo porcentual:** suma de errores firmados dividida por la suma de reales absolutos.
    - **MAE/RMSE:** error absoluto y raiz del error cuadratico sobre totales semanales.
    - **R2:** variabilidad explicada frente a la media de los reales; puede ser negativo.
    - Las ventanas son completas H1-H7 y se mantienen separadas por finca, bloque, origen y semana.
    """)

tab_summary, tab_input, tab_rf, tab_bayes, tab_overfit, tab_features, tab_corr, tab_trace = st.tabs([
    "Resumen semanal", "Nueva entrada", "RF optimizados", "Intervalos Bayes", "Sobreajuste", "Variables y rezagos", "Correlacion y VIF", "Trazabilidad"
])

with tab_input:
    st.subheader("Evaluacion del archivo de entrada")
    input_path = ROOT / "resultados acutuales/conteos_vs_cortes_multifinca.xlsx"
    input_daily, input_weekly = load_input_evaluation()
    if st.button("Ejecutar evaluacion de la entrada", disabled=not input_path.exists()):
        with st.spinner("Generando proyecciones causales para todos los modelos..."):
            evaluate_input(ROOT, input_path)
        load_input_evaluation.clear()
        st.rerun()
    if input_weekly.empty:
        st.info("Aun no existe una evaluacion para la entrada. Use el boton para generarla.")
    else:
        input_models = sorted(input_weekly.modelo.unique())
        input_model = st.selectbox("Modelo de entrada", input_models, key="input_model")
        input_farms = sorted(input_weekly.finca.unique())
        input_farm = st.selectbox("Finca de entrada", ["Todas"] + input_farms, key="input_farm")
        granularity = st.radio("Granularidad", ["Finca + semana", "Finca + bloque + semana"], horizontal=True)
        data = input_weekly[input_weekly.modelo.eq(input_model)].copy()
        if input_farm != "Todas":
            data = data[data.finca.eq(input_farm)]
        if granularity == "Finca + semana":
            data = (data.groupby(["modelo", "finca", "semana_proyeccion"], as_index=False)
                    .agg(real=("real", "sum"), proyectado=("proyectado", "sum"), dias=("dias", "sum")))
            data["diferencia"] = data.proyectado - data.real
            data["error_abs"] = data.diferencia.abs()
            data["razon_proyectado_real"] = np.where(data.real.ne(0), data.proyectado / data.real, np.nan)
            data["desviacion_pct"] = data.razon_proyectado_real - 1
            data["indicador"] = data.razon_proyectado_real.map(weekly_status)
        else:
            data = data.sort_values(["finca", "bloque", "semana_proyeccion"])
        st.caption("La semana corresponde a la ventana objetivo lunes-domingo posterior al ultimo conteo semanal.")
        display = data[[c for c in ["modelo", "finca", "bloque", "semana_proyeccion", "real", "proyectado",
                                    "diferencia", "error_abs", "razon_proyectado_real", "desviacion_pct", "indicador"] if c in data]]
        styled = display.style.map(
            lambda value: "background-color: #c6efce" if value == "ACIERTO" else
            ("background-color: #ffeb9c" if value == "CERCA" else
             ("background-color: #ffc7ce" if value == "NO ACIERTO" else "")), subset=["indicador"])
        st.dataframe(styled, width="stretch", hide_index=True,
                     column_config={"real": st.column_config.NumberColumn("Real", format="%,.0f"),
                                    "proyectado": st.column_config.NumberColumn("Proyectado", format="%,.0f"),
                                    "razon_proyectado_real": st.column_config.NumberColumn("Proyectado / real", format="0.0%"),
                                    "desviacion_pct": st.column_config.NumberColumn("Desviacion", format="+0.0%;-0.0%")})
        st.download_button("Descargar evaluacion semanal CSV", display.to_csv(index=False),
                           "evaluacion_entrada_semanal.csv", "text/csv")
        st.download_button("Descargar detalle diario CSV", input_daily[input_daily.modelo.eq(input_model)].to_csv(index=False),
                           "evaluacion_entrada_diaria.csv", "text/csv")
        xlsx_path = ROOT / "outputs/evaluation/evaluacion_entrada.xlsx"
        if xlsx_path.exists():
            st.download_button("Descargar evaluacion Excel", xlsx_path.read_bytes(),
                               "evaluacion_entrada.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_summary:
    weekly_by_iso = (model_weekly.groupby("semana_proyeccion", as_index=False)
                     [["real", "proyectado_modelo"]].sum(min_count=1)
                     .melt("semana_proyeccion", var_name="serie", value_name="cantidad"))
    st.subheader("Pronostico semanal contra real")
    chart = alt.Chart(weekly_by_iso).mark_line(point=True).encode(
        x=alt.X("semana_proyeccion:O", title="Semana ISO"),
        y=alt.Y("cantidad:Q", title="Cantidad semanal"),
        color=alt.Color("serie:N", title=None),
        tooltip=["semana_proyeccion", "serie", alt.Tooltip("cantidad:Q", format=",.0f")],
    ).properties(height=340)
    st.altair_chart(chart, width="stretch")
    st.subheader("Comparacion semanal por modelo")
    st.dataframe(summary.sort_values("wape_semanal"), width="stretch", hide_index=True)
    comparison_chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X("wape_semanal:Q", title="WAPE semanal", axis=alt.Axis(format=".0%")),
        y=alt.Y("modelo:N", sort="-x", title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=320)),
        color=alt.condition(alt.datum.modelo == model, alt.value("#1565c0"), alt.value("#90caf9")),
        tooltip=["modelo", "ventanas", alt.Tooltip("wape_semanal:Q", format=".2%"),
                 alt.Tooltip("sesgo_pct:Q", format="+.2%"), alt.Tooltip("r2_semanal:Q", format=".3f")],
    ).properties(height=max(320, len(summary) * 32))
    st.altair_chart(comparison_chart, width="stretch")
    st.subheader("Resumen de indicadores operativos")
    st.caption("Origen seleccionado: fecha más cercana al lunes de la semana proyectada. "
               "ACIERTO: 93%-107%; CERCA: 90%-<93% o >107%-110%; NO ACIERTO: fuera de esos rangos.")
    hit_summary = operational_summary(operational)
    st.dataframe(hit_summary, width="stretch", hide_index=True,
                 column_config={
                     "pct_acierto": st.column_config.NumberColumn("% acierto", format="0.0%"),
                     "pct_acierto_o_cerca": st.column_config.NumberColumn("% acierto o cerca", format="0.0%"),
                 })

    operational_level = "finca + bloque + semana" if block != "Todos" else "finca + semana"
    st.subheader(f"Semanas de validacion operativa: {model}")
    st.caption(f"Nivel mostrado: {operational_level}. Selecciona un bloque para segmentar el resultado.")
    operational_model = operational[operational["modelo"].eq(model)]
    detail_sort_columns = ["semana_proyeccion", "finca"]
    if "bloque" in operational_model.columns:
        detail_sort_columns.append("bloque")
    if "fecha_origen" in operational_model.columns:
        detail_sort_columns.append("fecha_origen")
    detail = operational_model.sort_values(detail_sort_columns)
    detail_columns = ["modelo", "split", "finca", "bloque", "semana_proyeccion",
                      "fecha_origen_seleccionada", "inicio_semana", "distancia_inicio_dias",
                      "real", "proyectado_modelo", "diferencia", "diferencia_abs",
                      "ratio_proyectado_real", "desviacion_pct", "indicador"]
    detail_view = detail[[column for column in detail_columns if column in detail]]
    st.dataframe(detail_view, width="stretch", hide_index=True,
                 column_config={
                     "real": st.column_config.NumberColumn("Real", format="%,.0f"),
                     "proyectado_modelo": st.column_config.NumberColumn("Proyectado", format="%,.0f"),
                     "diferencia": st.column_config.NumberColumn("Diferencia", format="%,.0f"),
                     "diferencia_abs": st.column_config.NumberColumn("Error absoluto", format="%,.0f"),
                     "ratio_proyectado_real": st.column_config.NumberColumn("Proyectado / real", format="0.0%"),
                     "desviacion_pct": st.column_config.NumberColumn("Desviacion", format="+0.0%;-0.0%"),
                 })
    with st.expander("Resumen agrupado por finca y semana"):
        audit_daily = daily.copy()
        if farm != "Todas":
            audit_daily = audit_daily[audit_daily["finca"].eq(farm)]
        if block != "Todos":
            audit_daily = audit_daily[audit_daily["bloque"].astype(str).eq(block)]
        audit = operational_weekly_selected(audit_daily, by_block=False)
        audit = audit[audit["modelo"].eq(model)].sort_values(["semana_proyeccion", "finca"])
        audit_columns = ["modelo", "split", "finca", "semana_proyeccion",
                         "fecha_origen_seleccionada", "inicio_semana", "real",
                         "proyectado_modelo", "diferencia", "diferencia_abs",
                         "ratio_proyectado_real", "desviacion_pct", "indicador"]
        audit_view = audit[[column for column in audit_columns if column in audit]]
        st.caption("Resumen global por finca y semana. La fecha de origen seleccionada mantiene la trazabilidad del cálculo.")
        st.dataframe(audit_view, width="stretch", hide_index=True)
    st.download_button("Descargar metricas semanales CSV", summary.to_csv(index=False),
                       "metricas_semanales_modelos.csv", "text/csv")
    st.download_button("Descargar detalle de semanas CSV", detail_view.to_csv(index=False),
                       "detalle_semanal_operativo.csv", "text/csv")

with tab_rf:
    st.subheader("Mejor Random Forest por grupo de variables")
    st.caption("Resultados sobre SELECTION temporal. Sirven para seleccionar hiperparametros; el ranking formal se valida luego con rolling origin.")
    if rf_best.empty:
        st.info("No existe la busqueda optimizada. Ejecute evaluacion_hyperparametros.py.")
    else:
        rf_cols = [c for c in ["features", "model", "n_estimators", "max_depth", "min_samples_leaf",
                               "min_samples_split", "max_features", "criterion", "weekly_wape", "weekly_mae",
                               "weekly_rmse", "weekly_bias_pct", "weekly_r2", "daily_wape", "selection_score"] if c in rf_best]
        st.dataframe(rf_best[rf_cols].sort_values("weekly_wape"), width="stretch", hide_index=True)
        rf_chart = alt.Chart(rf_best).mark_bar().encode(
            x=alt.X("weekly_wape:Q", title="WAPE semanal en SELECTION", axis=alt.Axis(format=".0%")),
            y=alt.Y("features:N", sort="-x", title="Grupo de variables"),
            color=alt.Color("features:N", legend=None),
            tooltip=["model", "features", alt.Tooltip("weekly_wape:Q", format=".2%"),
                     alt.Tooltip("weekly_r2:Q", format=".3f"), alt.Tooltip("weekly_bias_pct:Q", format="+.2%")]
        ).properties(height=260)
        st.altair_chart(rf_chart, width="stretch")
        group = st.selectbox("Grupo RF para contraste", sorted(rf_best.features.unique()))
        selected_rf = rf_best[rf_best.features.eq(group)].iloc[0]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("WAPE semanal", _pct(selected_rf.weekly_wape))
        r2.metric("R2 semanal", f"{selected_rf.weekly_r2:.3f}")
        r3.metric("Sesgo semanal", _pct(selected_rf.weekly_bias_pct, signed=True))
        r4.metric("WAPE diario", _pct(selected_rf.daily_wape))
        st.code(str(selected_rf.model), language=None)
    if hardware_plan:
        st.subheader("Plan de computo de la ultima busqueda")
        st.json(hardware_plan)

with tab_bayes:
    if evaluation_mode == "Rolling origin":
        st.info("Los intervalos bayesianos disponibles corresponden a validacion fija; no se mezclan con el ranking rolling.")
    elif bayes.empty:
        st.info("No existen predicciones bayesianas con intervalos.")
    else:
        bayes_filtered = apply_filters(bayes, farm, block)
        bayes_models = sorted(bayes_filtered.model.unique())
        bayes_model = st.selectbox("Modelo bayesiano", bayes_models)
        bayes_model_data = aggregate_bayesian_intervals(bayes, bayes_draws, bayes_model, farm, block)
        if bayes_model_data.empty:
            st.info("No hay draws posteriores para los filtros seleccionados. Ejecute fase7_bayes.py.")
            st.stop()
        coverage = bayes_model_data[["coverage80", "coverage95", "width80", "width95"]].mean()
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Cobertura 80%", _pct(coverage.coverage80))
        b2.metric("Cobertura 95%", _pct(coverage.coverage95))
        b3.metric("Ancho medio 80%", f"{coverage.width80:,.0f}")
        b4.metric("Ancho medio 95%", f"{coverage.width95:,.0f}")
        st.caption("Cobertura nominal: 80% y 95%. La cobertura observada debe aproximarse a esos valores.")
        base = alt.Chart(bayes_model_data).encode(x=alt.X("semana_proyeccion:O", title="Semana ISO"))
        band95 = base.mark_area(opacity=.18, color="#90caf9").encode(y="low95:Q", y2="high95:Q")
        band80 = base.mark_area(opacity=.30, color="#1565c0").encode(y="low80:Q", y2="high80:Q")
        lines = base.transform_fold(["real", "pred"], as_=["serie", "cantidad"]).mark_line(point=True).encode(
            y=alt.Y("cantidad:Q", title="Cantidad semanal"), color=alt.Color("serie:N", scale=alt.Scale(range=["#2e7d32", "#111827"])),
            tooltip=["semana_proyeccion", alt.Tooltip("serie:N"), alt.Tooltip("cantidad:Q", format=",.0f")])
        bands_chart = alt.layer(band95, band80, lines).properties(height=360)
        st.altair_chart(bands_chart, width="stretch")
        st.caption("Bandas calculadas desde sumas de draws posteriores para los filtros activos; no se suman cuantiles diarios.")
        bayes_cols = [c for c in ["experiment_id", "coverage_interval_80", "coverage_interval_95",
                                  "ancho_medio_intervalo", "wape", "mae", "rmse", "bias_pct", "r2", "n"]
                       if c in diagnostics["bayes_metrics"]]
        st.dataframe(diagnostics["bayes_metrics"][bayes_cols], width="stretch", hide_index=True)

with tab_overfit:
    overfit = diagnostics["overfit"]
    if overfit.empty:
        st.info("No existe diagnostico de sobreajuste.")
    else:
        current = overfit[overfit.modelo.eq("RF_H1_H7_FENO")]
        st.dataframe(current, width="stretch", hide_index=True)
        chart = alt.Chart(current).mark_line(point=True).encode(
            x=alt.X("horizonte:O", title="Horizonte"), y=alt.Y("wape:Q", title="WAPE", axis=alt.Axis(format=".0%")),
            color=alt.Color("scope:N", title="Muestra"), tooltip=["horizonte", "scope", "wape", "r2", "riesgo_sobreajuste"]
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        gap = current.drop_duplicates("horizonte")
        gap_chart = alt.Chart(gap).mark_bar().encode(
            x=alt.X("horizonte:O", title="Horizonte"), y=alt.Y("gap_wape:Q", title="Gap WAPE", axis=alt.Axis(format=".0%")),
            color=alt.Color("riesgo_sobreajuste:N"), tooltip=["horizonte", "gap_wape", "riesgo_sobreajuste"]
        ).properties(height=260)
        st.altair_chart(gap_chart, width="stretch")

with tab_features:
    importance = diagnostics["importance"]
    if not importance.empty:
        st.subheader("Importancia por horizonte")
        top = importance.sort_values("importance", ascending=False).groupby("horizonte", as_index=False).head(10)
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("importance:Q"), y=alt.Y("variable:N", sort="-x"), color=alt.Color("horizonte:N"),
            tooltip=["horizonte", "variable", "importance"]
        ).properties(height=500), width="stretch")
        st.dataframe(top, width="stretch", hide_index=True)
    if not diagnostics["lag_target"].empty:
        st.subheader("Relacion de variables rezagadas con el target en TRAIN")
        st.dataframe(diagnostics["lag_target"].head(30), width="stretch", hide_index=True)

with tab_corr:
    if not diagnostics["vif"].empty:
        st.subheader("VIF")
        vif = diagnostics["vif"].head(20).copy()
        vif["log10_vif"] = np.log10(vif["vif"].clip(lower=1))
        st.altair_chart(alt.Chart(vif).mark_bar().encode(
            x=alt.X("log10_vif:Q", title="log10(VIF)"), y=alt.Y("variable:N", sort="-x"),
            color=alt.Color("riesgo:N"), tooltip=["variable", "vif", "riesgo"]
        ).properties(height=500), width="stretch")
    st.subheader("Pares de alta correlacion")
    pairs = diagnostics["pairs"].head(100)
    if not pairs.empty:
        st.altair_chart(alt.Chart(pairs).mark_rect().encode(
            x=alt.X("variable_1:N", title=None, sort=None), y=alt.Y("variable_2:N", title=None, sort=None),
            color=alt.Color("correlacion:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
            tooltip=["variable_1", "variable_2", alt.Tooltip("correlacion:Q", format=".3f")]
        ).properties(height=520), width="stretch")
    st.subheader("Correlaciones entre variables rezagadas")
    lag_pairs = diagnostics["lag_pairs"].head(100)
    if not lag_pairs.empty:
        st.altair_chart(alt.Chart(lag_pairs).mark_rect().encode(
            x=alt.X("variable_1:N", title=None, sort=None), y=alt.Y("variable_2:N", title=None, sort=None),
            color=alt.Color("correlacion:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
            tooltip=["variable_1", "variable_2", alt.Tooltip("correlacion:Q", format=".3f")]
        ).properties(height=520), width="stretch")

with tab_trace:
    st.subheader("Modelo seleccionado")
    if manifests.get("selected"):
        st.json(manifests["selected"])
    st.subheader("Champion y poblacion formal")
    if manifests.get("champion"):
        st.json(manifests["champion"])
    if not diagnostics["leakage"].empty:
        st.subheader("Auditoria basica de leakage")
        st.dataframe(diagnostics["leakage"], width="stretch", hide_index=True)
    st.warning("Las correlaciones e importancias son diagnosticas; no prueban causalidad. Los modelos P32 se mantienen separados por su etiqueta retrospectiva.")
