# Bitácora de evaluación — Objetivo Específico 3 (Etapas 19-23)

**Proyecto:** Reconstrucción muográfica por transmisión asistida por aprendizaje profundo mediante mejora predictiva de imágenes de baja estadística para exploración volcánica.
**Institución:** Universidad Industrial de Santander, Escuela de Ingeniería de Sistemas e Informática, grupo GOTS.
**Alcance:** Fase D del plan de 30 etapas (Etapas 19 a 23, Objetivo Específico 3), incluyendo el reentreno E19.5 de las baselines U-Net L1.

## Nota de procedencia

Esta bitácora se compiló el 2026-09-28 a partir de registros primarios: el historial de commits del repositorio de trabajo `muography-thesis`, los informes de etapa en `docs/reports/`, el protocolo congelado y su etiqueta Git, la bitácora semanal de investigación y los archivos de ejecución en el clúster SC3 (`test_access_log.json` y los `.out` de SLURM). No existió una bitácora de evaluación escrita durante la ejecución de la Fase D; los eventos que siguen no fueron redactados en la fecha que llevan, sino reconstruidos después a partir de esas fuentes. Cada entrada cita su evidencia. Donde una fuente no registra un dato, se dice explícitamente que no está registrado. Las rutas con prefijo `results/`, `models/` y `figures/results/` se refieren a este repositorio publicado (`muography-dl-reconstruction`); los hashes de commit se refieren al repositorio de trabajo (`muography-thesis`).

## Resumen

La Fase D evaluó el esquema de mejora muográfica sobre el conjunto de prueba, abierto exactamente una vez (SLURM 29819, nodo `felix`, 1533,4 s de pared) bajo el protocolo v1.1 congelado a cero TBD. El cWGAN-GP final supera a la mejor clásica congelada (S1) y cruza los pisos absolutos de calidad (S2) en los cuatro niveles de estadística (50/10/5/1 %), con SSIM por celda de 0,9057 / 0,8080 / 0,7933 / 0,7798 y márgenes de +0,020 a +0,057 sobre los pisos. De las doce comparaciones primarias de la familia, diez excluyen 0 tras Holm; las dos excepciones son cWGAN − U-Net L1 al 50 % (Δ +0,0020, IC [−0,0012, +0,0046]) y al 1 % (Δ −0,0028, IC [−0,0057, +0,0007]), ambas con Holm p = 0,458 (en el dry run de validación las doce sí habían excluido 0). El modelo transfiere (SSIM de prueba igual o mayor que la de validación en cada nivel). El estadístico de detección `z_det` pre-registrado quedó dominado por el fondo estructurado de la montaña: la tasa formal de detección es aproximadamente 0 %, una limitación del detector, no del modelo. La medida salvable dentro del protocolo, preservación del signo del contraste físico, es positiva (98,5 % de las ROI). El factor de reducción de exposición alcanza hasta aproximadamente 31× al 1 % en SSIM cruda, pero la clásica congelada sola ya explica aproximadamente 27,6× de esa cifra; el aporte incremental atribuible al modelo aprendido es de aproximadamente 3,7×. El OE3 queda técnicamente completo; a la fecha de esta compilación resta ejecutar la inspección visual ciega.

## Entradas cronológicas

### 2026-09-05 — Etapa 19: plan de ejecución de la Fase D

**Actividad.** Se redactó `docs/planning/phase-d-execution-plan.md`: análisis de los hallazgos de la Fase C (H1-H12), diseño del protocolo v1.1, del pase único sobre prueba y de los reentrenos E19.5, con el orden de ejecución en dos pistas (trabajo sin gate y trabajo tras el gate) y el registro de riesgos D1-D12.

**Resultado.** Se fijó el plan con la regla de oro de la fase: el conjunto de prueba y `geo-gen-01` se leen una sola vez, con el protocolo congelado a cero TBD antes de esa lectura.

**Incidencias.** Ninguna registrada en esta entrada; los fixes documentales de citas erradas (H8) y la actualización de `.gitignore` y `KEEP_BEST_PHASES` se ejecutaron por separado el mismo día.

**Evidencia.** Commits `764ed03` (14:45) y `eb8a154` (15:31) en `muography-thesis`.

### 2026-09-05 — Etapa 19: módulo de evaluación implementado y revisado

**Actividad.** Se implementaron los catorce módulos de `evaluation/` (ROI, detectabilidad, textura, estadística, inferencia con verificación SHA-256, cache, equivalencia, techo estocástico, benchmark forward-only, driver único de la Etapa 20, calibración de umbrales, tablas y figuras, paneles ciegos, análisis del factor), cada uno con su suite `test_*.py`. El módulo se auditó con dos revisores de contexto fresco antes de tocar el conjunto de prueba.

**Resultado.** 15 suites sintéticas PASSED en local, más las verificaciones V1-V5 de geometría real pendientes de ejecutar en el clúster (entrada siguiente). La revisión adversarial encontró y corrigió 7 test-burners y 10 hallazgos de números equivocados, entre ellos un almacén sin inicializar que habría quemado el pase único, colisiones de llave de cache entre brazos, conteos de referencia no cacheados y una implementación de `equivalence.py` que nunca se invocaba.

**Incidencias.** Ninguna quedó sin corregir antes del commit.

**Evidencia.** Commit `4fe69ac` (17:32) en `muography-thesis`, `evaluation/*.py`; `docs/reports/stage-19-evaluation-module.md` §1 y §5 (añadido por el commit `77cb251`, 19:01).

### 2026-09-05 — Etapa 19.5: reentreno de las baselines U-Net L1

**Actividad.** Se reentrenaron los cinco checkpoints U-Net L1 podados por la política de cuota de la Fase C, con las configuraciones y semillas archivadas verbatim, más dos sondas opcionales del generador final (`unet_dc_5`) con `lambda_adv=0` al 5 % y al 1 %. Se enviaron dos trabajos SLURM en paralelo en el nodo `felix` (partición `gpu_titan`): 29811 (grupo a: f50, f10, f5) y 29812 (grupo b: f1, f1-shared, más las dos sondas opcionales).

**Resultado.** Las cinco corridas requeridas reprodujeron la SSIM de validación archivada dentro del criterio ±0,005:

| Corrida | SSIM reproducida | SSIM archivada | Δ | Época mejor | Duración |
|---|---|---|---|---|---|
| `e19-unetl1-f50` | 0,9014 | 0,9021 | −0,0007 | 19 | 2255 s |
| `e19-unetl1-f10` | 0,7821 | 0,7832 | −0,0011 | 11 | 1802 s |
| `e19-unetl1-f5` | 0,7775 | 0,7787 | −0,0012 | 1 | 2528 s |
| `e19-unetl1-f1` | 0,7736 | 0,7723 (`ref_equivalent`) | +0,0013 | 2 | 1361 s |
| `e19-unetl1-f1-shared` | 0,7750 | 0,7747 (`shared`) | +0,0003 | 2 | 1339 s |

Las dos sondas opcionales (`e19-dc5-l1only-f5`, SSIM 0,7836 en 91 épocas, 9299 s; `e19-dc5-l1only-f1`, SSIM 0,7691 en 55 épocas, 5618 s) completaron también, ambos trabajos con estado `COMPLETE`.

**Incidencias.** Ninguna reportada en los logs de esta corrida. Las dos sondas opcionales de `lambda_adv=0` no se citan en ningún informe de las Etapas 19 a 23 ni entran al manifiesto (ver Vacíos de registro, al final).

**Evidencia.** `cluster/sc3/e19_unetl1_retrain.sbatch`; logs de clúster `muo-e19-retrain-29811.out` y `muo-e19-retrain-29812.out` (recuperados de SC3, no versionados en el repositorio); commit `9f7f963` (17:34) con `model/weights-manifest-e19.json` (publicado como `models/weights-manifest-e19.json`); `docs/reports/stage-19-evaluation-module.md` §4.

### 2026-09-05 — Etapa 19: dry run de validación y congelamiento del protocolo v1.1

**Actividad.** Se ejecutó la ruta de código completa de la Etapa 20 sobre validación, para convertir en números medidos, antes de abrir el conjunto de prueba, la calibración de `z_thr` por (brazo, nivel), la tolerancia de preservación X, los pisos absolutos S2, el techo estocástico y el tiempo de pared. Se realizaron tres envíos sucesivos del mismo `sbatch` al nodo `felix` (partición `gpu_titan`).

**Resultado.** El tercer intento completó el pase de evaluación (wall 18,5 min, coincidente con el valor citado en `docs/reports/stage-19-dryrun-validation.md` §3) y produjo, tras las correcciones de esa misma noche, el protocolo congelado: `z_thr` por (brazo, nivel) con el brazo F (cWGAN-GP) en 11,561 / 14,336 / 16,521 / 18,067 (50/10/5/1 %); pisos S2 en SSIM 0,8854 / 0,7649 / 0,7405 / 0,7233; techo estocástico SSIM 0,9048 ± 0,0042 (PSNR 31,22 dB); techo de partición de media-estadística 0,6554. `docs/protocol/protocol-v1.1.md` quedó a cero TBD y etiquetado `protocol-v1.1` en Git.

**Incidencias.**

1. Job 29813 (22:34 UTC) falló en `evaluation/test_roi.py` con `ModuleNotFoundError: No module named 'surrogate'`: el paquete de despliegue del clúster no incluía `simulation/surrogate/`.
2. Job 29814 (22:37 UTC), tras corregir el despliegue, completó los tests unitarios pero el estimador de techo estocástico dio 0,8381 ± 0,0062 (delta 0,0669 respecto al 0,905 esperado): la construcción original de dos redraws independientes componía varianza de estimación y de remuestreo. `STAGE 19 CEILING: FAILED`.
3. Job 29815 (22:45 UTC), tras corregir el estimador de techo (commit `b43c330`, 22:45 UTC), completó el pase de evaluación completo (`STAGE 20 EVALUATION: PASSED`, wall 18,5 min) pero `report_tables.py` lanzó `TypeError: ufunc 'isfinite' not supported` en `detectability.classify`: `csv.DictReader` producía cadenas vacías en columnas de ROI que el conversor numérico mapeaba a `None`.

Tras las correcciones sucesivas (commits `833ad33`, `3f21eb7`, `42dcd2b`, `cebf5ca`, entre 23:21 y 23:39 UTC), las tablas y umbrales se regeneraron con éxito: `docs/protocol/protocol_thresholds.json` quedó generado a las 23:38:59 UTC, reutilizando el cache de evaluación que el job 29815 ya había escrito. El protocolo v1.1 se congeló con el commit `4364ed5` (18:53 −0500, 23:53 UTC) y la etiqueta `protocol-v1.1`.

**Evidencia.** `cluster/sc3/e19_dryrun.sbatch`; logs `muo-e19-dryrun-29813.out`, `-29814.out`, `-29815.out` (recuperados de SC3); commits `b43c330`, `833ad33`, `3f21eb7`, `42dcd2b`, `cebf5ca`, `4364ed5`; `docs/protocol/protocol_thresholds.json` (publicado como `results/evaluation/protocol_thresholds.json`); `docs/reports/stage-19-dryrun-validation.md` §4, §5 y §8 (añadido por el commit `77cb251`, 19:01); tag Git `protocol-v1.1`.

### 2026-09-05 — Etapa 20: pase único sobre el conjunto de prueba

**Actividad.** Se envió el `sbatch` de la Etapa 20 con el gate mecánico exigiendo protocolo congelado, umbrales no provisionales y el conjunto de prueba nunca abierto. Se realizaron dos envíos al nodo `felix` (partición `gpu_titan`).

**Resultado.** El pase único, ejecutado en el segundo intento, abrió y cerró el conjunto de prueba exactamente una vez: `metrics.csv` con sha256 `28eef9da36de8881a2d33b128a960c69ef7df1e0b6de478330e474ed8dbe1049` (149 478 filas) y `octiles.csv` con sha256 `a257b2581e0d9d525983e36e13846ac5c1e1600a5ea89e07a8ce5b140c13a0b2` (143 862 filas), idénticos byte a byte entre el clúster y el repositorio local. cWGAN-GP por celda: 0,9057 / 0,8080 / 0,7933 / 0,7798 (50/10/5/1 %), transfiriendo sobre validación en cada nivel (Δ +0,0003 / +0,0083 / +0,0099 / +0,0103). S1 cumple en los cuatro niveles (Δ +0,0108 / +0,0123 / +0,0063 / +0,0263 frente a la clásica congelada, IC excluyendo 0, Holm p = 9,16e-05). S2 cumple en los cuatro niveles, con márgenes de +0,020 a +0,057 sobre los pisos. El empate cWGAN vs. U-Net L1 al 50 % (Δ +0,0020, IC [−0,0012, +0,0046]) y al 1 % (Δ −0,0028, IC [−0,0057, +0,0007]) no rechaza H0, el caso pre-registrado que se dirime en la Etapa 21. El veredicto conjuntivo pre-registrado S1∧S2∧S3 es **NO CUMPLE** al 50 %, 10 % y 5 % porque S3 queda sin datos (el umbral calibrado por brazo excede el `|z|` máximo de toda referencia, ver Etapa 21), y **CUMPLE** solo al 1 %, donde S3 no se exige por diseño (límite de aplicabilidad declarado en el protocolo). Descompuesto, S1 y S2 cumplen en los cuatro niveles; el único componente no satisfecho de la tríada es S3, por la limitación del instrumento de detección, no por la mejora (stage-20 §5).

**Incidencias.** El primer envío (job 29817, 23:55 UTC del 5 de septiembre) falló en la verificación pre-vuelo V3/V4 de ROI sobre `geo-test-01`: el conducto y la cavidad de esa geometría resultaron solapados (centros a 57,1 m, radios 30,3 y 36,5 m, con rangos en z que se intersecan; la separación de 100 m del catálogo se cumplía para la geometría de generalización, no para esta de prueba), lo que hacía que la descomposición aditiva de longitud de camino contara dos veces el solapamiento (error relativo máximo 5,85e-3 frente al umbral de 1e-3). El ancho angular de la elipsoide también sobrestimaba entre 22 % y 26 % al usar `max(ax, ay)` en vez de proyectar sobre la dirección horizontal perpendicular a la línea de visión. El conjunto de prueba **no se abrió** en este intento: el fallo ocurrió en la verificación pre-vuelo, antes de `run_evaluation.py`, sin entrada en el *access log* ni lectura de *shards* (confirmado por el mensaje del commit de corrección y por el *access log*, que solo registra el job 29819). Corregido con el commit `316eb2e` (19:12 −0500), que re-marcha la opacidad reproducida evaluando `geo.density()` directamente en vez de sumar longitudes por anomalía, y proyecta el ancho de la elipsoide sobre la dirección transversal a la vista. El segundo envío (job 29819, abierto 00:20:25 UTC y cerrado 00:45:17 UTC del 6 de septiembre) pasó las verificaciones V1-V5 en las cuatro geometrías (V4 con error relativo máximo 5,93e-08 a 5,96e-08 en `geo-test-01`) y completó el pase único con wall = 1533,4 s.

**Evidencia.** `cluster/sc3/e20_test_pass.sbatch`; logs `muo-e20-test-29817.out` y `muo-e20-test-29819.out` (recuperados de SC3); `evaluation/test_access_log.json` (recuperado de SC3, no versionado; un único par apertura/cierre, `access_id 20260906T002025261640Z-662661`, `slurm_job_id 29819`, `host felix.sc3.uis.edu.co`); commit `316eb2e` (19:12); `evaluation/cache/test/summary.json` (publicado como `results/evaluation/test/summary.json`); commit `b7b5183` (21:40) con `docs/reports/stage-20-test-evaluation.md`.

### 2026-09-05 — Etapa 21: detectabilidad de anomalías (post-procesamiento del cache de la Etapa 20)

**Actividad.** Se analizaron las tasas de detección/preservación/alucinación, la curva de contraste físico por octiles, la adjudicación de textura del empate del 1 % y la galería de peores casos, todo sobre el cache ya escrito por el pase único, sin lecturas nuevas del conjunto de prueba.

**Resultado.** El estadístico `z_det` pre-registrado quedó dominado por el fondo estructurado de la montaña (la opacidad varía con elevación y azimut, y el anillo de 4-12 px absorbe ese gradiente en `σ_BG`). Sobre las geometrías de prueba, el `|z|` de anomalía real de referencia alcanza mediana 3,16 y máximo 10,11, prácticamente indistinguible del `|z|` de las ROI *sham* de referencia (mediana 3,03, p90 7,93); el `z_thr` calibrado por brazo para el cWGAN (11,561 a 18,067) supera todo `|z|` real medido, de modo que la tasa formal de detección/preservación del modelo es aproximadamente 0 %, una limitación del detector, no del modelo. La medida salvable dentro del protocolo, sin umbral de detección, es la preservación del contraste físico: el cWGAN preserva el signo en el 98,5 % de las ROI (2837/2880), con Pearson 0,958 en conductos (error relativo mediano 12,8 %), Spearman 0,778 en cuerpos densos (con magnitud inflada, mediana −0,403 frente a −0,252 de referencia) y Spearman 0,042 en cavidades (correlación débil). La adjudicación de textura del empate del 1 % favorece al cWGAN: razón de varianza 0,6727 frente a 0,5874 de la U-Net L1, razón de banda Nyquist 0,4942 frente a 0,3936, sin mayor error residual (DE 0,0322 frente a 0,0334).

**Incidencias.** Ninguna nueva; este análisis reutiliza el cache de la Etapa 20 sin abrir de nuevo el conjunto de prueba. La inspección visual ciega (60 paneles) quedó preparada, con los paneles renderizados, pero pendiente de ejecución humana a la fecha de esta compilación.

**Evidencia.** `docs/reports/stage-21-detectability.md` §1, §4 y §5; `docs/reports/fragments/04_detectability.md`; `figures/rendered/results/stage21_*.png` (publicadas como `figures/results/stage21_*.png`); commit `b7b5183` (21:40) y `61f6651` (21:41, figuras).

### 2026-09-05 — Etapa 22: factor de reducción del tiempo de adquisición

**Actividad.** Se ejecutó §6 del protocolo v1.1 sobre el cache de la Etapa 20: curva de equivalencia PCHIP sobre log₁₀ f', raíz de Brent, bootstrap cluster de dos etapas (B = 10 000), fila por método con columna de aporte incremental (hallazgo H9 pre-registrado).

**Resultado.** En la convención cruda (comparable con la literatura), el factor SSIM del cWGAN-GP es 1,17× / 3,58× / 6,66× / 31,26× (50/10/5/1 %, con IC 95 % por bootstrap, por ejemplo [30,67, 31,92] al 1 %). La clásica congelada sola ya alcanza 27,59× al 1 % (IC [27,15, 28,09]): la mayor parte del factor absoluto proviene de la restauración determinista de amplitud 1/f, no del aprendizaje. El aporte incremental del modelo aprendido sobre la mejor clásica es +0,07× / +0,21× / +0,19× / +3,67× en los cuatro niveles. En la convención corregida 1/f (cota conservadora): cWGAN 1,08× / 2,96× / 5,44× / 25,18×; clásica 1,00× / 2,76× / 5,25× / 21,76×. Por PSNR el factor es mayor: cWGAN 1,70× / 8,06× / 16,04× / 79,72× al 1 %, clásica 73,42×. El punto de comparación con O'Donnell et al. (2025, factor escalar 21-31×) es el 1 %, donde el cWGAN da 31,26× (SSIM cruda), del mismo orden de magnitud.

**Incidencias.** Ninguna; post-procesamiento puro del cache de la Etapa 20, sin lecturas adicionales del conjunto de prueba.

**Evidencia.** `docs/reports/stage-22-time-reduction.md` §2-5; `docs/reports/fragments/factor_tables.md` y `factor_tables.json`; `figures/rendered/results/stage22_equivalence.png` (publicada como `figures/results/stage22_equivalence.png`); commit `b7b5183` (21:40).

### 2026-09-05 — Etapa 23: síntesis, amenazas a la validez y cierre técnico del OE3

**Actividad.** Se redactó la síntesis de hallazgos E20-E22, el mapeo de cada afirmación a su evidencia y hash, el listado de amenazas a la validez (internas, de constructo y externas) y el trabajo futuro.

**Resultado.** La pregunta de investigación se responde en dos mitades: calidad comparable confirmada en datos nunca vistos (S1 y S2 en los cuatro niveles, con transferencia), y preservación de la detectabilidad parcial, sostenida por la fidelidad de contraste y la adjudicación de textura, con la limitación honesta del detector `z_det` documentada sin rodeos. Entre las amenazas registradas: el pre-registro genuino se limita a los §5.2-5.4 y §6 del protocolo (el resto es documentación de facto sobre artefactos ya congelados de la Fase B); el solapamiento conducto-cavidad detectado en `geo-test-01` (ver Etapa 20); los reentrenos U-Net L1 post-poda dentro de ±0,005; el desajuste de contrato de entrada al 1 % corregido con la comparación emparejada como primaria; el no determinismo de la GPU en la inferencia; la generalización subpotenciada por diseño (una sola geometría, n = 6); y la brecha sim-to-real, con la validación en datos reales (MuTe/Cerro Machín) como primera línea de trabajo futuro junto con un estadístico de detectabilidad con fondo detrendizado. La síntesis consolida una tabla de evidencia (10 afirmaciones, cada una con su fuente y hash de `metrics.csv`) y los criterios de aceptación por etapa.

**Incidencias.** El OE3 queda técnicamente completo; a la fecha de esta compilación resta ejecutar la inspección visual ciega de la Etapa 21.

**Evidencia.** `docs/reports/stage-23-synthesis.md`; commit `b7b5183` (21:40).

### 2026-09-14 — Auditoría previa al informe de avance y hallazgos sobre el estado documental

**Actividad.** Se redactó y compiló el informe de avance (proyecto LaTeX independiente, 42 páginas, 11 figuras, 9 tablas trazables a artefactos congelados), precedido de una auditoría del estado real del repositorio frente a lo declarado en el plan.

**Resultado.** La auditoría confirmó que el veredicto combinado de la Etapa 20 es «no cumple» en 50/10/5 % por el criterio conjuntivo S1∧S2∧S3 con S3 sin datos (ver Etapa 20), y que el informe reporta esa descomposición en vez de un titular optimista. La revisión sistemática (Etapa 2) está en aproximadamente 5 %, no en el 15 % que declaraba el plan.

**Incidencias.** Ninguna bloqueante para la Fase D; los hallazgos de esta auditoría son de índole documental, no de los resultados técnicos del OE3.

**Evidencia.** Commits `2585371` (21:48) y `c327507` (21:51) en `muography-thesis`; `docs/research-log/2026-W38.md`.

## Cierre

**Estado final.** El Objetivo Específico 3 queda técnicamente completo a la fecha de esta compilación: el conjunto de prueba se abrió exactamente una vez, bajo un protocolo congelado a cero TBD, y las tres etapas de análisis posteriores (21, 22 y 23) se completaron sin volver a tocarlo. Los resultados centrales (S1, S2, transferencia, preservación de contraste, adjudicación de textura, factor de reducción) están documentados con su dispersión o intervalo de confianza en cada informe de etapa.

**Acciones humanas pendientes, registradas en las fuentes.**

- Inspección visual ciega de la Etapa 21 (60 paneles, aproximadamente 30 min cada uno): paneles renderizados, sin ejecutar.

**Vacíos de registro declarados.**

- La Etapa 19 no tiene una adenda propia en la bitácora semanal de investigación: la única entrada de la Fase D en `docs/research-log/2026-W36.md` («Adenda 05/09 — Fase D ejecutada») resume los hallazgos de las Etapas 20 y 21, pero no documenta en el registro semanal la implementación del módulo, los reentrenos E19.5 ni los tres envíos sucesivos del dry run (dos con fallas distintas antes de que el tercero completara el pase de evaluación y tropezara, a su vez, con el error de `report_tables.py` corregido esa misma noche); esos detalles solo existen en los informes de etapa y en los logs del clúster.
- El *access log* del conjunto de prueba (`evaluation/test_access_log.json`) existía únicamente en el clúster SC3, no en el repositorio de trabajo ni en el publicado; se recuperó por `scp` para esta compilación.
- No hay bitácoras semanales para 2026-W33, W34, W35 ni W37: el directorio `docs/research-log/` solo contiene W32, W36 y W38.
- Las dos sondas opcionales de la Etapa 19.5 (`e19-dc5-l1only-f5` y `e19-dc5-l1only-f1`, generador final con `lambda_adv=0`) completaron su entrenamiento según el log del clúster, pero no aparecen citadas en ningún informe de las Etapas 19 a 23 ni en `model/weights-manifest-e19.json`; no está registrado si su resultado se usó para algo o si fueron descartadas.
- El commit exacto bajo el cual corrió el pase único de la Etapa 20 no quedó registrado: tanto `evaluation/cache/test/summary.json` como `evaluation/test_access_log.json` guardan `"git_commit": "unknown"`.
