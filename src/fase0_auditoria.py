"""Fase 0 - Auditoria de fuentes.

Inventario real de cada archivo Excel de data/raw:
- hash SHA-256 del archivo
- hojas, columnas, granularidad inferida
- n_filas, nulos por columna, duplicados por clave
- rangos temporales cuando exista columna de fecha
- problemas de tipos (fechas mixtas texto/serial, etc.)

Salidas:
- outputs/data_quality/inventario_fuentes.csv   (una fila por hoja)
- outputs/data_quality/inventario_columnas.csv  (una fila por columna)
- outputs/data_quality/hashes_fuentes.csv
- outputs/data_quality/hallazgos_fase0.md       (resumen legible)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "data_quality"

# Claves candidatas para prueba de duplicados por fuente (prompt seccion 26)
KEYS = {
    "conteos_vs_cortes_multifinca.xlsx": [["Finca", "Bloque", "Fecha"]],
    "camas_muestreadas_semana.xlsx": [["Finca", "Bloque", "Semana"]],
    "2025.xlsx": [["IdEstacion", "FechaHora"]],
    "2026.xlsx": [["IdEstacion", "FechaHora"]],
    "Podas 10.xlsx": [],
    "plano_siembra.xlsx": [],
}

DATE_COLS = {"Fecha", "FechaHora", "Fecha Siembra", "Fecha Siembra Inicial",
             "Fecha Erradicacion"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fechas_mixed(s: pd.Series) -> pd.Series:
    """Parsea fechas que pueden venir como datetime, texto o serial Excel.

    Devuelve serie de datetime64 con NaT donde no se pueda parsear.
    Determinista: texto con dayfirst=True (formato dd/mm/yyyy del proyecto);
    numeros como serial Excel (origen 1899-12-30).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    non_null = s.dropna()
    if non_null.empty:
        return out
    # numericos -> serial Excel
    num = pd.to_numeric(non_null, errors="coerce")
    is_num = num.notna() & (num > 20000) & (num < 60000)
    if is_num.any():
        out.loc[num.index[is_num]] = pd.to_datetime(
            num[is_num], unit="D", origin="1899-12-30")
    # resto -> texto
    resto = non_null[~non_null.index.isin(num.index[is_num])]
    if not resto.empty:
        out.loc[resto.index] = pd.to_datetime(
            resto, dayfirst=True, errors="coerce")
    return out


def perfilar_hoja(xlsx: Path, sheet: str, df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Perfil de una hoja: fila resumen + filas por columna."""
    df = df.dropna(how="all")
    fila = {
        "archivo": xlsx.name,
        "hoja": sheet,
        "n_filas": len(df),
        "n_cols": df.shape[1],
        "columnas": json.dumps([str(c) for c in df.columns],
                               ensure_ascii=False),
        "sha256": HASHES[xlsx.name],
    }
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "archivo": xlsx.name,
            "hoja": sheet,
            "columna": str(c),
            "dtype": str(s.dtype),
            "n_no_nulos": int(s.notna().sum()),
            "n_nulos": int(s.isna().sum()),
            "pct_nulos": round(float(s.isna().mean()) * 100, 2),
            "n_unicos": int(s.nunique(dropna=True)),
        }
        if str(c) in DATE_COLS or "fecha" in str(c).lower():
            f = parse_fechas_mixed(s)
            info["fecha_min"] = (str(f.min().date()) if f.notna().any()
                                 else None)
            info["fecha_max"] = (str(f.max().date()) if f.notna().any()
                                 else None)
            info["n_fecha_no_parseable"] = int((s.notna() & f.isna()).sum())
        cols.append(info)
    # duplicados por clave declarada
    for key in KEYS.get(xlsx.name, []):
        if all(k in df.columns for k in key):
            fila[f"dup_{'_'.join(key)}"] = int(
                df.duplicated(subset=key).sum())
    return fila, cols


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    xlsxs = sorted(RAW.glob("*.xlsx"))
    if not xlsxs:
        print("No hay xlsx en data/raw", file=sys.stderr)
        return 1

    global HASHES
    HASHES = {p.name: sha256(p) for p in xlsxs}
    pd.DataFrame(
        [{"archivo": k, "sha256": v,
          "bytes": (RAW / k).stat().st_size} for k, v in HASHES.items()]
    ).to_csv(OUT / "hashes_fuentes.csv", index=False)

    filas, columnas = [], []
    for xlsx in xlsxs:
        t0 = datetime.now()
        print(f"[{t0:%H:%M:%S}] {xlsx.name} ...", flush=True)
        try:
            xl = pd.ExcelFile(xlsx, engine="openpyxl")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR abriendo: {e}", file=sys.stderr)
            continue
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR hoja {sheet}: {e}", file=sys.stderr)
                continue
            fila, cols = perfilar_hoja(xlsx, sheet, df)
            filas.append(fila)
            columnas.extend(cols)
            print(f"  - {sheet}: {fila['n_filas']} filas x "
                  f"{fila['n_cols']} cols")
        print(f"  ({(datetime.now() - t0).total_seconds():.1f}s)")

    inv = pd.DataFrame(filas)
    inv.to_csv(OUT / "inventario_fuentes.csv", index=False)
    pd.DataFrame(columnas).to_csv(OUT / "inventario_columnas.csv",
                                  index=False)

    # resumen legible
    with (OUT / "hallazgos_fase0.md").open("w", encoding="utf-8") as f:
        f.write("# Inventario de fuentes - Fase 0\n\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write("| Archivo | Hoja | Filas | Cols | Duplicados clave |\n")
        f.write("|---|---|---:|---:|---|\n")
        for _, r in inv.iterrows():
            dup = [f"{k.split('dup_')[1]}={v}" for k, v in r.items()
                   if str(k).startswith("dup_") and pd.notna(v)]
            f.write(f"| {r['archivo']} | {r['hoja']} | {r['n_filas']} | "
                    f"{r['n_cols']} | {'; '.join(dup)} |\n")

    print(f"\nOK -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
