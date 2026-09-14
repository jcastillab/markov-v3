from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "resultados acutuales"

RUTA_RESULTADOS = INPUT_DIR / "Resultados"
RUTA_ROSES = INPUT_DIR / "Roses.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSOLIDAR CSV
# ============================================================

def consolidar_resultados_csv(
    path_raiz,
    encoding="utf-8-sig",
    separador=",",
    incluir_nombre_archivo=False
):

    path_raiz = Path(path_raiz)

    if not path_raiz.exists():
        raise FileNotFoundError(
            f"La ruta no existe: {path_raiz}"
        )

    if not path_raiz.is_dir():
        raise NotADirectoryError(
            f"La ruta no corresponde a una carpeta: {path_raiz}"
        )

    dataframes = []

    for archivo_csv in path_raiz.rglob("*.csv"):

        bloque = archivo_csv.parent.name
        finca = archivo_csv.parent.parent.name
        semana = archivo_csv.parent.parent.parent.name

        try:

            df_archivo = pd.read_csv(
                archivo_csv,
                encoding=encoding,
                sep=separador
            )

            # La semana registrada en el CSV es la fuente de verdad. Algunas
            # carpetas contienen archivos cuya semana no coincide con la ruta.
            if "semana" in df_archivo.columns:
                semana_csv = (
                    df_archivo["semana"]
                    .astype("string")
                    .str.strip()
                    .replace({"": pd.NA})
                    .fillna(semana)
                )
            else:
                semana_csv = semana

            df_archivo["Semana"] = semana_csv
            df_archivo["Finca"] = finca
            df_archivo["Bloque"] = bloque

            if incluir_nombre_archivo:
                df_archivo["ArchivoOrigen"] = archivo_csv.name

            dataframes.append(df_archivo)

        except Exception as error:

            print(
                f"No se pudo leer {archivo_csv}: {error}"
            )

    if not dataframes:

        return pd.DataFrame(
            columns=["Semana", "Finca", "Bloque"]
        )

    return pd.concat(
        dataframes,
        ignore_index=True
    )


# ============================================================
# PROCESAR CONTEOS
# ============================================================

print("Leyendo CSV...")

conteos = consolidar_resultados_csv(
    RUTA_RESULTADOS,
    incluir_nombre_archivo=True
)

print(
    f"Registros CSV consolidados: {len(conteos):,}"
)

conteos = conteos[
    conteos["fecha"].notna()
].copy()


def parsear_fecha_conteo(valores):
    """Parsea fechas ISO y fechas locales sin invertir fechas ISO ambiguas."""

    valores = valores.astype("string").str.strip()
    fechas = pd.Series(pd.NaT, index=valores.index, dtype="datetime64[ns]")

    es_iso = valores.str.match(r"^\d{4}-\d{2}-\d{2}")

    fechas.loc[es_iso] = pd.to_datetime(
        valores.loc[es_iso],
        format="mixed",
        dayfirst=False,
        errors="coerce"
    )
    fechas.loc[~es_iso] = pd.to_datetime(
        valores.loc[~es_iso],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    return fechas


conteos["Fecha"] = parsear_fecha_conteo(conteos["fecha"])


# Normalizar finca

conteos["Finca"] = (
    conteos["Finca"]
    .astype(str)
    .str.upper()
    .str.strip()
)


conteos["Finca"] = conteos["Finca"].replace({
    "LA PRADERA": "PRADERA",
    "PRADERA": "PRADERA",
    "ALMER": "ALMER",
    "SANTA HELENA": "SANTA HELENA"
})


# Mantener únicamente las tres fincas del modelo

conteos = conteos[
    conteos["Finca"].isin([
        "PRADERA",
        "ALMER",
        "SANTA HELENA"
    ])
].copy()


# Normalizar identificadores

conteos["Bloque"] = (
    conteos["Bloque"]
    .astype(str)
    .str.strip()
)


conteos["Semana"] = (
    conteos["Semana"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# Consolidar todos los videos de la misma finca, bloque,
# semana y fecha

conteos = (
    conteos
    .groupby(
        [
            "Finca",
            "Bloque",
            "Semana",
            "Fecha"
        ],
        as_index=False
    )
    .agg(
        conteo_RC=("conteo_RC", "sum"),
        conteo_SS=("conteo_SS", "sum"),
        conteo_AP=("conteo_AP", "sum"),
        conteo_CO=("conteo_CO", "sum"),
        conteo_total=("conteo_total", "sum")
    )
)


conteos.to_excel(
    OUTPUT_DIR / "conteos.xlsx",
    index=False
)


print(
    f"Conteos agrupados: {len(conteos):,}"
)


# ============================================================
# PROCESAR CORTES REALES
# ============================================================

print("Leyendo Roses.xlsx...")

cortes = pd.read_excel(
    RUTA_ROSES,
    sheet_name="Hoja1"
)


# Normalizar campos de texto

for columna in [
    "Finca",
    "Flor",
    "Variedad",
    "Color"
]:

    cortes[columna] = (
        cortes[columna]
        .astype(str)
        .str.strip()
    )


# Filtrar Freedom Red

cortes = cortes[
    cortes["Finca"].isin([
        "LA PRADERA",
        "ALMER",
        "SANTA HELENA"
    ])
    &
    cortes["Variedad"].str.upper().eq("FREEDOM")
    &
    cortes["Color"].str.upper().eq("RED")
].copy()


# Homologar finca

cortes["Finca"] = cortes["Finca"].replace({
    "LA PRADERA": "PRADERA"
})


# ============================================================
# CONVERTIR FECHA XLSB
# ============================================================

cortes["Fecha"] = pd.to_datetime(
    cortes["Fecha"],
    errors="coerce"
)


# ============================================================
# AGRUPAR CORTES
# ============================================================

cortes["Block"] = (
    cortes["Block"]
    .astype(str)
    .str.strip()
)


cortes = (
    cortes
    .groupby(
        [
            "Finca",
            "Block",
            "Flor",
            "Variedad",
            "Color",
            "Fecha"
        ],
        as_index=False
    )
    .agg(
        Cantidad=("Cantidad", "sum")
    )
)


# ============================================================
# CALCULAR SEMANA ISO
# ============================================================

iso = cortes["Fecha"].dt.isocalendar()


cortes["semana"] = (
    iso["year"].astype(int) * 100
    +
    iso["week"].astype(int)
)


cortes["Semana"] = (
    "S"
    +
    iso["week"]
    .astype(int)
    .astype(str)
)


# Homologar nombre bloque

cortes.rename(
    columns={
        "Block": "Bloque"
    },
    inplace=True
)


# ============================================================
# MERGE CORTES + CONTEOS
# ============================================================

columnas_conteo = [
    "Finca",
    "Bloque",
    "Semana",
    "Fecha",
    "conteo_RC",
    "conteo_SS",
    "conteo_AP",
    "conteo_CO",
    "conteo_total"
]

# ============================================================
# NORMALIZAR LLAVES ANTES DEL MERGE
# ============================================================

def normalizar_bloque(valor):
    if pd.isna(valor):
        return None

    valor = str(valor).strip().upper()

    # Corrige casos provenientes de Excel como 10.0
    if valor.endswith(".0"):
        valor = valor[:-2]

    return valor


def normalizar_finca(valor):
    if pd.isna(valor):
        return None

    valor = str(valor).strip().upper()

    equivalencias = {
        "LA PRADERA": "PRADERA",
        "PRADERA": "PRADERA",
        "ALMER": "ALMER",
        "SANTA HELENA": "SANTA HELENA",
    }

    return equivalencias.get(valor, valor)


for df in [conteos, cortes]:

    df["Finca"] = df["Finca"].apply(normalizar_finca)

    df["Bloque"] = df["Bloque"].apply(normalizar_bloque)

    df["Semana"] = (
        df["Semana"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce"
    ).dt.normalize()

llaves = [
    "Finca",
    "Bloque",
    "Semana",
    "Fecha"
]

diagnostico = conteos.merge(
    cortes[llaves].drop_duplicates(),
    on=llaves,
    how="left",
    indicator=True
)

sin_corte_correspondiente = diagnostico[
    diagnostico["_merge"] == "left_only"
].copy()

sin_corte_correspondiente.to_excel(
    OUTPUT_DIR / "conteos_sin_match.xlsx",
    index=False
)

print(
    "Conteos sin correspondencia con cortes:",
    len(sin_corte_correspondiente)
)

df_final = cortes.merge(
    conteos[columnas_conteo],
    on=[
        "Finca",
        "Bloque",
        "Semana",
        "Fecha"
    ],
    how="left"
)


# Desde semana 17 de 2026

df_final = df_final[
    df_final["semana"] >= 202617
].copy()


df_final = df_final.sort_values(
    by=[
        "Finca",
        "Bloque",
        "Fecha"
    ]
).reset_index(drop=True)


# Orden exacto del archivo original

columnas_finales = [
    "Finca",
    "Bloque",
    "Flor",
    "Variedad",
    "Color",
    "Fecha",
    "Cantidad",
    "semana",
    "Semana",
    "conteo_RC",
    "conteo_SS",
    "conteo_AP",
    "conteo_CO",
    "conteo_total"
]


df_final = df_final[columnas_finales]


# ============================================================
# EXPORTAR
# ============================================================

ruta_salida = (
    OUTPUT_DIR
    / "conteos_vs_cortes_multifinca.xlsx"
)


df_final.to_excel(
    ruta_salida,
    index=False
)


print()
print("Proceso terminado.")
print(
    f"Filas finales: {len(df_final):,}"
)
print(
    f"Archivo: {ruta_salida}"
)
