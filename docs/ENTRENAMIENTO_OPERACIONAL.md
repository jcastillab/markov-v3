# Entrenamiento operacional

Las fases historicas se conservan como implementacion interna y auditoria. La

## Comandos

Consolidar RAW y construir datasets:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py consolidar
```

Evaluar modelos con validacion temporal, rolling-origin y ranking:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py evaluar
```

Entrenar y congelar bundles:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py entrenar --cutoff 2026-08-09
```

Ejecutar todo el ciclo historico:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py todo --cutoff 2026-08-09
```

Proyectar sin entrenar:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py operacional -- --semana S36
```

Preparar todos los artefactos que consume el dashboard, incluida la pestaña
`Nueva entrada` y la última corrida operacional:

```powershell
.\.venv\Scripts\python.exe scripts\modelos.py dashboard
```

## Paralelismo

Los parametros viven en `config/pipeline.yaml`:

- Entrenamiento RF operacional: `n_jobs: -1`.
- Busqueda de hiperparametros: todos los procesos disponibles, limitado por
  memoria disponible y sin paralelismo anidado.
- `reserve_cpus: 0` permite usar todos los CPUs detectados.

Si el equipo tiene poca RAM, aumentar `estimated_memory_per_worker_gb` evita
que la busqueda de hiperparametros sature el sistema. La configuracion puede
modificarse sin cambiar codigo.

## Compatibilidad

`scripts/run_historical.py` y `scripts/train_models_cmd.py` quedan como
wrappers de compatibilidad y reenvian sus argumentos a la nueva interfaz.
