"""Fase 0 - Sondas de contratos (prompt secciones 34-36).

Responde con evidencia:
1. Nombres de finca/bloque en cada fuente -> tabla de homologacion.
2. Significado de conteo_CO (distribucion, relacion con otros conteos).
3. Unidad de Cantidad vs Cantidad_proy en podas.
4. Fila anomala en camas_muestreadas.
5. Formatos de FechaHora en clima (texto vs serial Excel) y solape 2025/2026.
6. Codigos fenologicos crudos en las 3 fuentes (abril, julio, P32),
   incluidos SP aislado y pendientes.
7. Regla 'ultimo conteo valido de la semana': cuantos conteos por
   finca+bloque+semana existen realmente.

Salida: outputs/data_quality/contratos_fase0.md + CSVs de soporte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "data_quality"

pd.set_option("display.width", 200)


def sec(f, titulo):
    f.write(f"\n\n## {titulo}\n\n")


def tabla_md(f, df, max_rows=60):
    f.write(df.head(max_rows).to_markdown(index=False))
    if len(df) > max_rows:
        f.write(f"\n\n... ({len(df) - max_rows} filas mas)")
    f.write("\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "contratos_fase0.md"

    with md.open("w", encoding="utf-8") as f:
        f.write("# Contratos y sondas - Fase 0\n")

        # ------------------------------------------------------------------
        # 1. Fincas y bloques por fuente
        # ------------------------------------------------------------------
        sec(f, "1. Nombres de finca y bloque por fuente")

        cc = pd.read_excel(RAW / "conteos_vs_cortes_multifinca.xlsx")
        camas = pd.read_excel(RAW / "camas_muestreadas_semana.xlsx")
        plano = pd.read_excel(RAW / "plano_siembra.xlsx")
        podas = pd.read_excel(RAW / "Podas 10.xlsx")

        fincas = []
        for nombre, df, col in [
            ("conteos_cortes", cc, "Finca"),
            ("camas_muestreadas", camas, "Finca"),
            ("plano_siembra", plano, "Finca"),
            ("podas", podas, "Finca"),
        ]:
            for finca, n in df[col].astype(str).value_counts().items():
                fincas.append({"fuente": nombre, "finca_raw": finca,
                               "n_filas": n})
        fincas_df = pd.DataFrame(fincas)
        fincas_df.to_csv(OUT / "qa_fincas_por_fuente.csv", index=False)
        tabla_md(f, fincas_df)

        # bloques por finca en conteos
        sec(f, "1b. Bloques por finca en conteos_vs_cortes")
        bl = (cc.assign(Bloque=cc["Bloque"].astype(str))
                .groupby(["Finca", "Bloque"]).size().reset_index(name="n"))
        bl.to_csv(OUT / "qa_bloques_conteos.csv", index=False)
        tabla_md(f, bl)

        # ------------------------------------------------------------------
        # 2. conteo_CO
        # ------------------------------------------------------------------
        sec(f, "2. conteo_CO")
        conteos = cc[cc["conteo_total"].notna()].copy()
        f.write(f"Filas con conteo_total no nulo: {len(conteos)}\n\n")
        if len(conteos):
            cols = ["conteo_RC", "conteo_SS", "conteo_AP", "conteo_CO",
                    "conteo_total"]
            desc = conteos[cols].describe().round(2)
            tabla_md(f, desc.reset_index().rename(columns={"index": "stat"}))
            # CO = total - RC - SS - AP ?
            calc = (conteos["conteo_total"] - conteos["conteo_RC"]
                    - conteos["conteo_SS"] - conteos["conteo_AP"])
            cmp_df = pd.DataFrame({
                "total_menos_RCSA": calc,
                "conteo_CO": conteos["conteo_CO"],
            })
            cmp_df["diferencia"] = cmp_df["conteo_CO"] - cmp_df["total_menos_RCSA"]
            f.write("\n¿conteo_CO == conteo_total - (RC+SS+AP)?\n\n")
            f.write(f"- Filas donde coincide exactamente: "
                    f"{(cmp_df['diferencia'] == 0).sum()} de {len(cmp_df)}\n")
            f.write(f"- Filas donde CO no nulo pero total-RCSA = 0: "
                    f"{((cmp_df['conteo_CO'] > 0) & (cmp_df['total_menos_RCSA'] == 0)).sum()}\n")
            f.write(f"- Filas donde CO = 0 o nulo pero total-RCSA > 0: "
                    f"{((cmp_df['conteo_CO'].fillna(0) == 0) & (cmp_df['total_menos_RCSA'] > 0)).sum()}\n")
            f.write("\nEstadistica de la diferencia (CO - (total-RCSA)):\n\n")
            tabla_md(f, cmp_df["diferencia"].describe().round(2).to_frame().reset_index().rename(columns={"index": "stat"}))
            # distribucion de CO
            f.write("\nDistribucion de conteo_CO:\n\n")
            tabla_md(f, conteos["conteo_CO"].describe().round(2).to_frame().reset_index().rename(columns={"index": "stat"}))

        # ------------------------------------------------------------------
        # 3. Podas: Cantidad vs Cantidad_proy, Destinos
        # ------------------------------------------------------------------
        sec(f, "3. Podas: destinos y unidades")
        podas_f = podas[
            podas["Variedad"].astype(str).str.upper().str.contains("FREEDOM", na=False)
        ].copy()
        f.write(f"Filas FREEDOM: {len(podas_f)}\n\n")
        f.write("Destinos (FREEDOM):\n\n")
        tabla_md(f, podas_f["Destino"].value_counts().to_frame().reset_index())
        f.write("\nColumnas: " + ", ".join(map(str, podas.columns)) + "\n\n")
        f.write("Cantidad vs Cantidad_proy (FREEDOM, fincas objetivo):\n\n")
        fincas_obj = ["ALMER", "PRADERA", "LA PRADERA", "SANTA HELENA"]
        pf = podas_f[podas_f["Finca"].astype(str).str.upper().isin(fincas_obj)]
        f.write(f"- Filas: {len(pf)}\n")
        f.write(f"- Iguales exactamente: {(pf['Cantidad'] == pf['Cantidad_proy']).sum()}\n")
        f.write(f"- Cantidad_proy nulo: {pf['Cantidad_proy'].isna().sum()}\n")
        f.write(f"- Cantidad negativa: {(pf['Cantidad'] < 0).sum()}, cero: {(pf['Cantidad'] == 0).sum()}\n\n")
        f.write("Descripcion de Cantidad por Destino (FREEDOM objetivo):\n\n")
        tabla_md(f, pf.groupby("Destino")["Cantidad"].describe().round(2).reset_index())
        f.write("\nRango de fechas podas FREEDOM: "
                f"{pd.to_datetime(pf['Fecha'], dayfirst=True, errors='coerce').min()} a "
                f"{pd.to_datetime(pf['Fecha'], dayfirst=True, errors='coerce').max()}\n")
        f.write("\nFincas en podas FREEDOM:\n\n")
        tabla_md(f, podas_f["Finca"].value_counts().to_frame().reset_index())
        f.write("\nColumna de bloque: 'Block'. Ejemplos por finca:\n\n")
        tabla_md(f, pf.groupby("Finca")["Block"].apply(
            lambda s: ", ".join(sorted(s.astype(str).unique())[:15])).reset_index())

        # ------------------------------------------------------------------
        # 4. Camas muestreadas: fila anomala
        # ------------------------------------------------------------------
        sec(f, "4. Camas muestreadas: filas anomalas")
        f.write("Valores unicos de la primera columna (nombre real: "
                f"'{camas.columns[0]}'):\n\n")
        tabla_md(f, camas[camas.columns[0]].value_counts().to_frame().reset_index())
        f.write("\nFincas distintas en camas:\n\n")
        tabla_md(f, camas["Finca"].astype(str).value_counts().to_frame().reset_index())
        # Semanas validas: formato S\d+
        sem = camas["Semana"].astype(str)
        invalidas = camas[~sem.str.match(r"^S\d+$", na=False)]
        f.write(f"\nFilas con Semana que NO cumple patron S<digitos>: {len(invalidas)}\n\n")
        if len(invalidas):
            tabla_md(f, invalidas)

        # ------------------------------------------------------------------
        # 5. Clima: formatos FechaHora, estaciones, solape
        # ------------------------------------------------------------------
        sec(f, "5. Clima: estaciones, formatos y solape")
        est = []
        for anio in ["2025", "2026"]:
            df = pd.read_excel(RAW / f"{anio}.xlsx")
            est.append((anio, df))
            f.write(f"\n### {anio}.xlsx ({len(df)} filas)\n\n")
            f.write(f"Estaciones: {df['IdEstacion'].nunique()} unicas\n\n")
            tabla_md(f, df.groupby(["IdEstacion", "Estacion"]).size().reset_index(name="n"), max_rows=25)
            fh = df["FechaHora"]
            f.write(f"dtype FechaHora: {fh.dtype}\n\n")
            # tipos de valores crudos
            tipos = fh.dropna().map(lambda v: type(v).__name__).value_counts()
            f.write("Tipos python de FechaHora:\n\n")
            tabla_md(f, tipos.to_frame().reset_index())
            num = pd.to_numeric(fh, errors="coerce")
            f.write(f"- Parseable como numero (serial Excel): {num.notna().sum()}\n")
            f.write(f"- Rango serial: {num.min()} - {num.max()}\n\n")
        f25, f26 = est[0][1], est[1][1]
        key25 = set(zip(f25["IdEstacion"], f25["FechaHora"].astype(str)))
        key26 = set(zip(f26["IdEstacion"], f26["FechaHora"].astype(str)))
        f.write(f"\nSolape exacto IdEstacion+FechaHora (texto crudo) entre archivos: "
                f"{len(key25 & key26)}\n")

        # ------------------------------------------------------------------
        # 6. Codigos fenologicos
        # ------------------------------------------------------------------
        sec(f, "6. Codigos fenologicos crudos por fuente")

        def codigos_tradicional(path, hojas):
            # columnas de tallo son enteros 1..30 (verificado en Fase 0)
            cod = {}
            xl = pd.ExcelFile(path)
            for h in hojas:
                if h not in xl.sheet_names:
                    continue
                df = xl.parse(h)
                tallo_cols = [c for c in df.columns
                              if isinstance(c, int) or str(c).isdigit()]
                for c in tallo_cols:
                    for v in df[c].dropna().astype(str).str.strip():
                        if v and v.lower() != "nan":
                            cod[v] = cod.get(v, 0) + 1
            return cod

        cod_abril = codigos_tradicional(
            RAW / "FENOLOGIAS ABRIL FREEDOM.xlsx", ["Abril", "Junio"])
        cod_julio = codigos_tradicional(
            RAW / "FENOLOGIAS JULIO FREEDOM.xlsx", ["JULIO"])
        f.write("\n### FENOLOGIAS ABRIL/JUNIO\n\n")
        tabla_md(f, pd.DataFrame(
            sorted(cod_abril.items(), key=lambda x: -x[1]),
            columns=["codigo", "n"]))
        f.write("\n### FENOLOGIAS JULIO\n\n")
        tabla_md(f, pd.DataFrame(
            sorted(cod_julio.items(), key=lambda x: -x[1]),
            columns=["codigo", "n"]))

        # P32: estructura no tabular - primera fila fechas, columnas por tallo
        sec(f, "6b. Codigos P32 (Fenologias13.08Final-1)")
        p32_codes: dict[str, int] = {}
        p32_meta = []
        xl = pd.ExcelFile(RAW / "Fenologias13.08Final-1.xlsx")
        for h in ["Garbanzo", "Rayando 1", "Separando S", "Definiendo P"]:
            raw = xl.parse(h, header=None)
            p32_meta.append({"hoja": h, "filas": raw.shape[0],
                             "cols": raw.shape[1]})
            # contar todos los strings que parezcan codigo en toda la hoja
            for v in raw.to_numpy().ravel():
                if isinstance(v, str):
                    v = v.strip()
                    if v and len(v) <= 8 and not v.startswith("="):
                        p32_codes[v] = p32_codes.get(v, 0) + 1
        tabla_md(f, pd.DataFrame(p32_meta))
        f.write("\nStrings detectados (crudo, incluye encabezados):\n\n")
        tabla_md(f, pd.DataFrame(
            sorted(p32_codes.items(), key=lambda x: -x[1]),
            columns=["valor", "n"]), max_rows=80)

        # ------------------------------------------------------------------
        # 7. Regla ultimo conteo de la semana
        # ------------------------------------------------------------------
        sec(f, "7. Conteos por finca+bloque+semana")
        cc2 = cc.copy()
        cc2["tiene_conteo"] = cc2["conteo_total"].notna()
        por_sem = (cc2[cc2["tiene_conteo"]]
                   .groupby(["Finca", "Bloque", "semana"])
                   .agg(n_conteos=("tiene_conteo", "sum"),
                        fechas=("Fecha", lambda s: ", ".join(
                            pd.to_datetime(s, dayfirst=True, errors="coerce")
                            .dt.strftime("%Y-%m-%d").dropna())))
                   .reset_index())
        f.write(f"Combinaciones finca+bloque+semana con conteo: {len(por_sem)}\n\n")
        f.write("Distribucion de numero de conteos por semana:\n\n")
        tabla_md(f, por_sem["n_conteos"].value_counts().sort_index()
                 .to_frame().reset_index())
        f.write("\nEjemplos con mas de 1 conteo en la semana:\n\n")
        tabla_md(f, por_sem[por_sem["n_conteos"] > 1].head(15))
        por_sem.to_csv(OUT / "qa_conteos_por_semana.csv", index=False)

        # ------------------------------------------------------------------
        # 8. Cortes: ceros y cobertura temporal
        # ------------------------------------------------------------------
        sec(f, "8. Corte comercial: ceros y cobertura")
        cc2["Cantidad_num"] = pd.to_numeric(cc2["Cantidad"], errors="coerce")
        f.write(f"- Dias con corte = 0: {(cc2['Cantidad_num'] == 0).sum()}\n")
        f.write(f"- Dias con corte nulo: {cc2['Cantidad_num'].isna().sum()}\n")
        f.write(f"- Dias con corte negativo: {(cc2['Cantidad_num'] < 0).sum()}\n\n")
        fechas_cc = pd.to_datetime(cc2["Fecha"], dayfirst=True, errors="coerce")
        f.write(f"Rango fechas: {fechas_cc.min()} a {fechas_cc.max()}\n\n")
        f.write("Filas por finca/variedad:\n\n")
        tabla_md(f, cc2.groupby(["Finca", "Variedad"]).size()
                 .reset_index(name="n"), max_rows=30)

    print(f"OK -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
