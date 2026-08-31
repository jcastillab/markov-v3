# Informe de Consultoría Técnica

## Modelo de Pronóstico de Corte de Rosas Freedom

Versión ejecutiva ampliada para gerencia, operaciones y equipo analítico.

> Conversión limpia a Markdown del PDF original. Se eliminaron repeticiones idénticas presentes en varias páginas del documento fuente, sin alterar el contenido técnico de cada apartado.

## Índice

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Diagnóstico Ejecutivo](#diagnóstico-ejecutivo)
- [Evaluación Metodológica](#evaluación-metodológica)
- [Auditoría de Arquitectura](#auditoría-de-arquitectura)
- [Hallazgo 1: x0](#hallazgo-1-x0)
- [Hallazgo 2: Edad Latente](#hallazgo-2-edad-latente)
- [Hallazgo 3: Ingreso RC](#hallazgo-3-ingreso-rc)
- [Hallazgo 4: AP a PC](#hallazgo-4-ap-a-pc)
- [Riesgos Estadísticos](#riesgos-estadísticos)
- [Riesgos de Negocio](#riesgos-de-negocio)
- [Causalidad y Backtesting](#causalidad-y-backtesting)
- [Estrategia x0 y Edad](#estrategia-x0-y-edad)
- [Estrategia AP](#estrategia-ap)
- [Estrategia Ingresos RC](#estrategia-ingresos-rc)
- [Clima y Podas](#clima-y-podas)
- [Roadmap 30-60-90 días](#roadmap-306090-días)
- [Matriz Impacto vs Esfuerzo](#matriz-impacto-vs-esfuerzo)
- [Arquitectura Objetivo](#arquitectura-objetivo)
- [MLOps y Gobierno](#mlops-y-gobierno)
- [Diseño Modelo V2](#diseño-modelo-v2)
- [Estimación de Reducción de WAPE](#estimación-de-reducción-de-wape)
- [Conclusiones](#conclusiones)

## Resumen Ejecutivo

La arquitectura conceptual es sólida. El mayor potencial de mejora proviene de x0 por edad, edad latente, calibración de ingresos RC y estabilización de AP→PC. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Diagnóstico Ejecutivo

El proyecto evolucionó correctamente desde enfoques agregados hacia un sistema fenológico basado en estados, cohortes, conservación de masa y Semi-Markov. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Evaluación Metodológica

Se analizan Markov, Semi-Markov y Hazard. M3 debe mantenerse como benchmark obligatorio. El Semi-Markov es el enfoque biológicamente correcto, pero su beneficio está limitado por la representación de edad. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Auditoría de Arquitectura

Se observan fortalezas en modularidad conceptual y debilidades asociadas a dependencias históricas, parámetros hardcodeados y trazabilidad parcial. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Hallazgo 1: x0

Todo el stock observado entra con edad cero. Esto reduce significativamente el valor del Semi-Markov. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Hallazgo 2: Edad Latente

La mayoría de exposiciones dependen de inferencia. Requiere fortalecimiento de observaciones y validación de probabilidades. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Hallazgo 3: Ingreso RC

El ingreso RC fijo puede inducir sesgo acumulativo. Debe calibrarse mediante backtesting causal. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Hallazgo 4: AP a PC

AP aparece como el estado más sensible del sistema. Se recomienda mantener M3 como referencia para AP hasta consolidar evidencia. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Riesgos Estadísticos

Sobreajuste, leakage temporal, dependencia de edad latente y extrapolación de muestras. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Riesgos de Negocio

Impacto sobre mano de obra, oferta comercial, logística y credibilidad de los pronósticos. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Causalidad y Backtesting

Implementar rolling origin y congelar artefactos de entrenamiento por fecha. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Estrategia x0 y Edad

Construir P(edad|microestado) usando RC1-RC4, S1S-S5S y SP1-SP2. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Estrategia AP

Separar microestados y fortalecer observaciones AP→PC. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Estrategia Ingresos RC

Evaluar mallas de parámetros entre 0% y 20%, segmentadas por finca o bloque si existe soporte. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Clima y Podas

Mantener como challengers. Reincorporar después de estabilizar núcleo biológico. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Roadmap 30-60-90 días

30 días: bugs y QA. 60 días: x0 por edad e ingresos. 90 días: Modelo V2 con validación causal. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Matriz Impacto vs Esfuerzo

Mayor retorno esperado: x0, edad latente, ingresos RC y AP→PC. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Arquitectura Objetivo

Raw→QA→Normalización→Feature Store→Semi-Markov→Evaluación→Power BI. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## MLOps y Gobierno

Versionado de datos, modelos, parámetros, artefactos, datasets y experimentos. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Diseño Modelo V2

Conteo→Microestado→Distribución edad→x0→Semi-Markov→Corte→Extrapolación→Evaluación. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Estimación de Reducción de WAPE

La mayor reducción provendría de resolver edad y x0 antes que clima o IA avanzada. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.

## Conclusiones

No se recomienda invertir en modelos más complejos antes de consolidar el núcleo causal y biológico. Este apartado desarrolla implicaciones operativas, estadísticas y arquitectónicas, incluyendo recomendaciones de implementación, riesgos, prioridades de ejecución y métricas de seguimiento para garantizar reproducibilidad, mejora continua y reducción sostenible del error de pronóstico.
