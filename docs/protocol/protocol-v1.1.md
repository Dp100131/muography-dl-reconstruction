# Protocolo Experimental v1.1 — pre-registro congelado

**Versión:** 1.1 (CONGELADO — cero campos sin resolver)
**Estado:** PRE-REGISTRO CONGELADO · **Fecha:** 2026-09-05
**Deciden:** Daniel Pinto y Reinaldo Cárdenas (autores) — pendiente de ratificación del director (PA-4; precedente `docs/actas/decision-framework-dl/authors-provisional-decision.md`).

| Versión | Cambio | Autor | Fecha |
|---|---|---|---|
| 0.1 | Esqueleto inicial (todos los campos sin resolver) | — | 2026-08-05 |
| 1.1 | Resolución de cada campo a un valor concreto: §5.2–§5.4 y §6 medidos sobre el dry-run de validación (E19, paso 7) con la prueba intacta; §2–§4, §5.1 y §7 documentados desde artefactos congelados de las Fases B y C. Se añade la columna/etiqueta de procedencia (PRE-REGISTRADO / DE FACTO), la tabla de potencia medida y el plan de reporte honesto dentro del documento. | Daniel Pinto, Reinaldo Cárdenas | 2026-09-05 |

**Regla de cierre cumplida:** cero campos sin resolver. Todos los marcadores de valor pendiente que gobernaban la v1.0 fueron sustituidos por valores numéricos concretos; una búsqueda del marcador de campo pendiente sobre este archivo devuelve vacío. La v1.0 (`docs/protocol/protocol-v1.0.md`) queda como histórico. Al congelar se etiqueta `protocol-v1.1` en Git; todo cambio posterior pasa por §8.

**Nomenclatura de procedencia (obligatoria en cada campo resuelto):**

- **PRE-REGISTRADO** — regla o número fijado ANTES de tocar el conjunto de prueba, medido sobre validación en el dry-run del paso 7 de la Etapa 19. Constituye pre-registro genuino contra un conjunto de prueba intacto. Aplica a **§5.2, §5.3, §5.4 y §6**.
- **DE FACTO** — valor documentado *a posteriori* desde artefactos ya congelados de las Fases B y C (catálogo de geometrías, parametrización del detector, dataset, código de métricas, cronograma real). No es pre-registro: se transcribe para dejar el documento completo y trazable, y se declara como tal. Aplica a **§2, §3, §4, §5.1 y §7**.

> **Declaración para las amenazas a la validez (E23) — se traslada verbatim:** el pre-registro genuino de este protocolo se limita a §5.2–§5.4 y §6, cuyas reglas y números se fijaron sobre validación con el conjunto de prueba y `geo-gen-01` sin abrir. Las secciones §2, §3, §4, §5.1 y §7 se documentan a posteriori desde artefactos congelados de las Fases B y C y **no son pre-registro**; se etiquetan DE FACTO precisamente para no presentarlas como decisiones anteriores a los datos. Esta distinción es una amenaza interna a la validez y se narra sin rodeos en la síntesis de la Etapa 23.

---

## 1. Objetivo y alcance

**Procedencia: transversal.**

Este protocolo fija los criterios con los que se juzgará el éxito del esquema de mejora predictiva ANTES de la lectura irreversible del conjunto de prueba (Etapa 20). Es el artefacto metodológico que habilita (*gate*) esa lectura única: sin este documento congelado a cero campos sin resolver, `MUOGRAPHY_ALLOW_TEST=1` no puede activarse.

**Qué decide este documento:** las fórmulas de detectabilidad, los umbrales de éxito, el plan de análisis estadístico y la metodología del factor de reducción de tiempo — todo lo que en §5.2–§5.4 y §6 lleva la etiqueta PRE-REGISTRADO.

**Qué NO decide:** la implementación de los modelos (Etapas 13–18, ya cerradas) ni la interpretación posterior. La Etapa 20 mide contra estos umbrales sin ajustes post hoc; las Etapas 21 y 22 son post-procesamiento puro del cache de la Etapa 20 (cero lecturas adicionales del conjunto de prueba).

**Diferencia entre lo pre-registrado y lo documentado a posteriori.** Las secciones DE FACTO (§2–§4, §5.1, §7) describen decisiones tomadas y congeladas durante las Fases B y C. Se incluyen para que el documento sea autocontenido y auditable, pero un revisor externo debe leerlas como *documentación*, no como *pre-registro*. Solo §5.2–§5.4 y §6 anticipan lo que se medirá y cómo se juzgará antes de ver los datos de prueba.

---

## 2. Escenarios geométricos

**Procedencia: DE FACTO** (transcrito del catálogo congelado de la Etapa 7: `simulation/geometry_catalog/catalog.md` + 19 JSON versionados; reportado en `docs/reports/stage-07-geometry-catalog.md`).

- **Número total de geometrías:** **19** (mínimo pedido por el plan: 10–20).
- **Particiones por geometría** (sin fuga de información, aserto en la Etapa 12): **12 train / 3 val / 3 test / 1 generalización** (`geo-gen-01`, excluida por completo del entrenamiento).

**Anomalías del catálogo** (parámetros reales, semilla fija 20260805):

| Tipo de anomalía | Rango de tamaño | Contraste de densidad ρ [g/cm³] | Instancias en el conjunto de evaluación | Procedencia |
|---|---|---|---|---|
| Conducto vertical (baja densidad) | radio 26,1–30,3 m | 1,83–2,15 | 2 por tipo en prueba | DE FACTO |
| Cavidad | radio 24,1–42,4 m | 0,05–0,45 | 2 por tipo en prueba | DE FACTO |
| Cuerpo denso | semiejes 39,6–74,2 m | 3,09–3,24 | 2 por tipo en prueba | DE FACTO |

- **Perfil de montaña base (edificio cónico):** radio de base R = 519–671 m, altura H = 386–428 m, densidad de roca ρ_rock = **2,65 g/cm³** (escala de cientos de metros, inspirado en los perfiles simplificados de Asama/Sakurajima y en el contexto del Cerro Machín — MuTe).
- **Cobertura del catálogo:** todos los tipos de anomalía aparecen solos y combinados, más una geometría de control homogéneo. Las 12 geometrías de entrenamiento combinan tipos, por lo que el conteo por tipo del catálogo completo es ambiguo; el recuento verificable y no ambiguo es el del conjunto de evaluación: **2 instancias por tipo en prueba** (6 instancias reales de anomalía en las 3 geometrías de prueba) y **3 instancias en `geo-gen-01`**, para un total de **9 instancias** en el conjunto retenido. (Nota de resolución: la columna «# configuraciones» de la v1.0 se sustituye por este recuento verificado por instancia; es un dato del catálogo congelado, DE FACTO.)

---

## 3. Parametrización del detector virtual

**Procedencia: DE FACTO** (parametrizado en `simulation/surrogate/imaging.py`, Etapa 8; construido idénticamente por la app Geant4).

| Parámetro | Valor | Procedencia |
|---|---|---|
| Área efectiva | **1,44 m²** (plano de 1,2 × 1,2 m, tipo MuTe) | DE FACTO |
| Grilla del histograma 2D (azimut × elevación) | **256 × 256** | DE FACTO |
| Rango azimutal | **±30°** | DE FACTO |
| Rango de elevación | **2–47°** | DE FACTO |
| Distancia al eje del edificio volcánico | **1.000 m** | DE FACTO |
| Altura del detector | **1,5 m** | DE FACTO |
| Número de vistas por geometría | **6**, separadas 60° (rotación en azimut alrededor del edificio) | DE FACTO |

La grilla 256 × 256 es directamente la entrada de la U-Net (sin re-muestreo). El registro píxel a píxel entre el ray-caster de evaluación y la formación de imagen de Geant4 está verificado contra `binIndex` (nodo `linspace` más cercano, sin desfase de medio píxel).

---

## 4. Niveles de estadística y equivalencia temporal

**Procedencia: DE FACTO** (calibración S0 de la Etapa 6 con EcoMug real; thinning binomial exacto de la Etapa 11).

- **Calibración de referencia:** **S0 = 36,175 × 10⁶ incidentes/vista/30 días** (dirección = convención del protocolo; espectro = EcoMug real; flujo por tablas de la campaña Geant4). El 100 % corresponde a la referencia de **30 días** de exposición.
- **Submuestreo:** thinning **binomial** (selección exacta de eventos reales, no re-simulación), con **K = 20** realizaciones por fracción y por (geometría, vista).

| Nivel (% de eventos) | Incidentes/vista | Días de exposición equivalentes | Procedencia |
|---|---|---|---|
| 100 % (referencia) | 36,175 × 10⁶ | 30 | DE FACTO |
| 50 % | 18,088 × 10⁶ | **15** | DE FACTO |
| 10 % | 3,618 × 10⁶ | **3** | DE FACTO |
| 5 % | 1,809 × 10⁶ | **1,5** | DE FACTO |
| 1 % | 0,362 × 10⁶ | **0,3** | DE FACTO |

- **Tamaño del dataset:** **9.120 pares** (19 geometrías × 6 vistas × 4 niveles × K=20), con train = 5.760 ≥ 5.000. Dataset congelado en el tag `dataset-v1.0` (Etapa 12).

---

## 5. Plan de evaluación

### 5.1 Métricas primarias

**Procedencia: DE FACTO** (transcrito de `evaluation/metrics.py`, congelado y *golden-tested*; los mismos parámetros produjeron `dataset/stats/stage11_baseline.json`, por lo que toda cifra de la Fase C queda comparable con la línea base sin modelo).

- **SSIM** — `skimage.metrics.structural_similarity(a, b, data_range=1.0, win_size=7, K1=0.01, K2=0.03, gaussian_weights=False)` → ventana uniforme (caja) de 7 px, K1 = 0,01, K2 = 0,03, rango dinámico `data_range=1.0`. Prohibido activar `gaussian_weights`: la variante gaussiana 11×11 de Wang difiere ≤ 0,002 y la consistencia con la tabla congelada prevalece sobre la convención.
- **PSNR** — `skimage.metrics.peak_signal_noise_ratio(ref, x, data_range=1.0)`.
- **Implementación de referencia:** `evaluation/metrics.py` (única interfaz de medición de las Fases C y D; sus parámetros están congelados y no se re-ajustan).
- **Nota sobre el piso de cielo (corrección de la v1.0):** el piso de opacidad de un píxel de cielo abierto NO es una constante única. `evaluation/roi.py::sky_opacity` reproduce bit a bit el cómputo de `imaging.opacity_map`, y como `n_t · RAY_STEP_M` no coincide exactamente con la ventana marchada, el piso es **débilmente dependiente de la geometría: 367,41–367,59 g/cm² según la vista**, no el valor redondo. El plan de la Fase D describía este piso como la constante única 367,53 g/cm²; aquí se corrige explícitamente al cómputo por geometría de `sky_opacity` (verificado por la prueba V4).

### 5.2 Métricas secundarias (detectabilidad)

**Procedencia: PRE-REGISTRADO** (fórmulas y umbral de detección fijados sobre validación con la prueba intacta).

- **Espacio de medición:** todas las métricas se calculan sobre las imágenes preprocesadas en **[0, 1]** (el espacio que emite el modelo; monótono en opacidad, de modo que el signo del contraste se preserva).
- **ROI por anomalía:** `evaluation/roi.py` importa la grilla angular (`angular_grid` / `ray_directions`) de `simulation/surrogate/imaging.py` (NO la re-implementa) y marcha con `RAY_STEP_M = 2,0 m`, sustituyendo `density()` por `_anomaly_mask(a)`, para obtener la longitud de camino L_a [m] por píxel y por anomalía. La descomposición de la columna (V4) está verificada contra la opacidad surrogate almacenada con **error relativo 5,7 × 10⁻⁸** (piso de cielo por geometría según §5.1).
  - **Regla FWHM:** `ROI(a) = { p : L_a(p) ≥ max(20 m, 0,5 · máx L_a) }` (20 m = 10 pasos de rayo; el corte FWHM domina en las 9 instancias).
  - **Mínimo 64 px** (≥ una ventana SSIM 7×7 con margen; la instancia menor, cavidad r = 24,1 m, da ≈ 82 px). Si alguna (anomalía, vista) baja de 64 px, **se excluye Y se reporta**, nunca se descarta en silencio.
- **Anillo de fondo:** 4–12 px alrededor de la ROI, excluyendo cielo, borde de silueta dilatado 3 px, otras ROIs dilatadas 3 px, y píxeles con `|L_rock − mediana_ROI| > 50 %`; requiere ≥ 200 px (si no, se ensancha a 16 px y se registra).
- **Fórmulas:**
  - Contraste: `C = (μ_ROI − μ_BG) / μ_BG` (signo: cavidad/conducto > 0, cuerpo denso < 0).
  - CNR: `CNR = (μ_ROI − μ_BG) / σ_BG`.
  - Estadístico de detección: `z_det = (μ_ROI − μ_BG) / (σ_BG · √(1/n_ROI + 1/n_BG))`.
- **Umbral de detección CALIBRADO POR BRAZO (`z_thr`), no asumido:** es el percentil 99 de |z| sobre las **ROIs sham** (10 por (geometría, vista); rechazo sembrado dentro de la silueta, con la forma/área de la distribución empírica de ROIs reales, L_a = 0 ∀a, ≥ 12 px de toda ROI real y del borde), medido sobre las referencias de **VALIDACIÓN** en el dry-run y congelado como número. El umbral se calibra **por separado para cada (brazo, nivel)** porque el *denoising* encoge σ_BG y desplaza la distribución de |z| sham: un escalar único produjo ~76–80 % de positivos sham en validación (un artefacto de umbral, no alucinación real). Cada brazo carga así, por construcción, una FPR ≈ 1 %.

  Tabla congelada `z_thr_by_arm` (de `docs/protocol/protocol_thresholds.json`, `provisional:false`):

  | Brazo | 50 % | 10 % | 5 % | 1 % |
  |---|---|---|---|---|
  | C — crudo 1/f | 9,547 | 5,894 | 4,510 | 2,678 |
  | D — clásica congelada por nivel | 9,537 | 20,361 | 19,486 | 18,733 |
  | E — U-Net L1 | 11,708 | 17,007 | 15,652 | 16,515 |
  | F — cWGAN-GP | 11,561 | 14,336 | 16,521 | 18,067 |

  Rangos por brazo a través de niveles: C 2,68–9,55; D 9,54–20,36; E 11,71–17,01; F 11,56–18,07. Escalar de referencia `z_thr = 10,464` (calibración del brazo de referencia, solo para continuidad; la detección usa SIEMPRE `z_thr_by_arm` cuando el brazo tiene entrada). Denominador sham sobre prueba: 18 celdas × 10 ROIs sham × K=20 = **3.600 mediciones sham por (nivel, brazo)**.
- **Variante física:** `ΔX = Δρ · L · 100 [g/cm²]` (`evaluation/roi.py::delta_opacity`), con rango real **3.024–22.065 g/cm²** sobre una columna media ≈ 57.600 g/cm² (perturbaciones del 5–38 %).

### 5.3 Umbrales de éxito

**Procedencia: PRE-REGISTRADO.**

**S1 — criterio primario (superioridad sobre la clásica).** En cada nivel, el modelo vence a la mejor clásica congelada de ese nivel si el intervalo primario (§5.4) de la diferencia por celda **excluye el 0** Y la comparación **sobrevive Holm** (§5.4).

**S2 — pisos absolutos por fórmula (medidos, congelados).** `piso_SSIM(f) = SSIM_val(f) − max(0,02; 3 · SD_celdas)` (análogo para PSNR con `max(1,0 dB; 3 · SD)`). El ancla `SSIM_val(f)` es el SSIM de validación del modelo final por nivel (0,9054 / 0,7997 / 0,7834 / 0,7695); el SSIM se computa contra la imagen de referencia de 30 días. El término de 0,02 (≈ 13× la σ inter-semilla de 0,0015) domina al 50 %; el término 3·SD_celdas domina en 10/5/1 %.

| Nivel | Piso SSIM | Piso PSNR [dB] | Procedencia |
|---|---|---|---|
| 50 % | **0,8854** | 31,30 | PRE-REGISTRADO |
| 10 % | **0,7649** | 28,96 | PRE-REGISTRADO |
| 5 % | **0,7405** | 28,63 | PRE-REGISTRADO |
| 1 % | **0,7233** | 28,56 | PRE-REGISTRADO |

**S3 — detectabilidad.**

| Métrica | Umbral | Alcance | Procedencia |
|---|---|---|---|
| Anomalías preservadas | ≥ **90 %** | para f ≥ 5 % (los 36 clusters (anomalía, vista) solo resuelven ±7 %, así que 90 % es el piso más fino que el diseño distingue). **Al 1 % se reporta sin exigir: es el límite de aplicabilidad declarado.** | PRE-REGISTRADO |
| Tasa de alucinación | ≤ **5 %** | = 5× la calibración sham del propio brazo (FPR ≈ 1 % por construcción, §5.2); IC ≈ ±1,5 % con 3.600 shams | PRE-REGISTRADO |
| Anomalías perdidas | ≤ **10 %** | sobre el conjunto de prueba | PRE-REGISTRADO |

**Constantes de clasificación (pre-registradas):**

- `detectada` ⇔ |z| ≥ z_thr(brazo, nivel) Y signo(C) = signo(C_ref).
- `preservada` ⇔ detectada Y `|C − C_ref| / |C_ref| ≤ X`, con **X = 0,30**. Justificación explícita: la regla general es `X = max(30 %, 3 · rel-SD(C_ref sobre K))`, pero la referencia de 30 días es una **única realización sin jitter K que medir**, de modo que el término `3·rel-SD(C_ref)` medido en el dry-run es 0,0 y `X = max(0,30; 0,0) = 0,30` queda en el piso pre-registrado. Se documenta explícitamente que X no se ajustó por medición (no había dispersión de referencia que medir).
- `perdida` ⇔ detectada en la referencia y no en la mejorada.
- `inflada` ⇔ `|C| > (1 + Y) · |C_ref|`, con **Y = 0,50**.
- `alucinada estructural` ⇔ |z| ≥ z_thr(brazo, nivel) sobre una ROI sham.
- **Válvula «no detectable al 100 %»:** una anomalía indetectable ya en la referencia se excluye del denominador y se reporta aparte (candidato: el conducto de `geo-test-01`, Δρ = −0,499, el más débil de las 9 instancias).

### 5.4 Análisis estadístico

**Procedencia: PRE-REGISTRADO.**

- **Unidad inferencial = media por celda (geometría, vista):** **n = 18 en prueba**, **n = 6 en generalización**. Las K = 20 realizaciones comparten un transporte padre, así que **n = 360 es pseudo-replicación**: se usa solo como dispersión descriptiva, nunca como n inferencial (PA-5).
- **Intervalo primario = t de Student sobre las 18 diferencias pareadas por celda, condicionado por normalidad Shapiro–Wilk (α = 0,05).** Si Shapiro rechaza → percentil de bootstrap cluster BCa de 2 etapas. **Se reportan SIEMPRE ambos.**
  - **Justificación medida (chequeo de cobertura sintético del dry-run, n = 18):** el bootstrap cluster de 2 etapas cubrió **0,905** frente a **0,940** de la t de Student, contra un nominal de 0,95 — sub-cobertura del bootstrap en muestra pequeña. Por eso la **t de Student es el intervalo primario** y el bootstrap el acompañante; ambos números de cobertura se documentan aquí. (Esto refina la elección provisional del plan, que anticipaba BCa primario; la medición del dry-run la invierte, que es exactamente el propósito del dry-run.)
- **Prueba primaria = Wilcoxon de rangos con signo, exacta, bilateral** sobre las 18 diferencias (p mínimo alcanzable a n=18 = **7,63 × 10⁻⁶**, que despeja el umbral de Holm 0,00417; la restricción real es la consistencia de signo entre geometrías y vistas). **Normalidad por Shapiro–Wilk** (NO D'Agostino, inválido a n=18).
- **Bootstrap cluster de 2 etapas:** re-muestrear las 18 celdas con reemplazo, luego las K dentro de cada celda; **B = 10.000**, sembrado. IC BCa con percentil de respaldo (si los extremos difieren > 20 % del ancho, se reporta percentil y se anota).
- **Holm–Bonferroni α = 0,05** sobre la **familia primaria de 12 hipótesis** = {cWGAN vs crudo+1/f, vs mejor clásica, vs U-Net L1} × 4 niveles. La **familia secundaria** (PSNR, contraste, CNR, textura) lleva su propio Holm. **El IC porta la afirmación; el p la acompaña.**
- **`geo-gen-01`:** **sin test formal** (n = 6 → p mínimo 0,031, no sobrevive Holm); solo estimaciones puntuales con IC cluster y la etiqueta «subpotenciado por diseño (una geometría)».
- **Regla de uso único, mecánicamente verificable:** `evaluation/run_evaluation.py` es el ÚNICO archivo autorizado a poner `MUOGRAPHY_ALLOW_TEST=1` (asertado por grep en el CI de tests, `evaluation/test_single_callsite.py`). Escribe `evaluation/test_access_log.json` **append-only** (UTC, commit de Git, argv, host, job SLURM, y el **sha256 de cada shard leído y de cada archivo de cache escrito**), transcrito a la bitácora.
- **Protocolo de caída (pre-registrado):** una falla del *job* **ANTES de emitir cualquier estadístico** permite **exactamente UN reinicio**, registrado como un evento `restart` con la razón y el sha256 del cache parcial. El `--resume` conserva solo las geometrías cuyo cache y filas ya están completos y verificados contra el manifiesto, rehace el resto y **anexa** a las tablas (no las trunca). Una **segunda evaluación lógica** (re-correr después de que ya existen estadísticos) está **prohibida**.
- **Potencia (medida en el dry-run):** la σ_d de la diferencia pareada por celda es mucho menor que una σ por par, porque el emparejamiento con la misma entrada cancela la varianza estructural (σ entre celdas ≈ 0,014, que se cancela). Los IC de la tabla primaria del dry-run implicaron **σ_d ≈ 0,0006 al 5 %**, de modo que el **MDE al 80 % de potencia bajo Holm(12) ≈ 0,0018**. Todos los márgenes de validación quedan muy por encima:
  - vs clásica: **+0,0140** (50 %), **+0,0128** (10 %), **+0,0058** (5 %), **+0,0255** (1 %);
  - aporte adversarial **+0,0040**;
  - diferencia cWGAN − U-Net L1 al 1 % = **−0,0056**;
  - todos con IC que excluyó el 0 en validación (el de 1 % enteramente por debajo de 0, coherente con el veredicto de E18).

  **Plan de reporte honesto (pre-registrado, redacción exacta por comparación):**
  - **50 / 10 / 1 % vs clásica →** se espera significativo. Si no lo es en prueba, **se reporta tal cual**, con respaldo en los pisos absolutos S2.
  - **5 % vs clásica, y adversarial-vs-L1 →** si no alcanza significancia, la conclusión registrada es exactamente: *«consistente en signo con validación pero no independientemente significativo a la potencia pre-registrada»* — **nunca** re-análisis ni cambio de unidad.
  - **1 % cWGAN vs U-Net L1 →** el veredicto es el de la Etapa 18, adjudicado por **textura y detectabilidad** (E21), no por SSIM global.
  - Todo cambio de un umbral, prueba o unidad **después** de ver los números de prueba exige una **adenda fechada en §8 + visto bueno del director**. El intervalo de confianza porta la afirmación; nunca se re-analiza.
- **Tamaño del conjunto de prueba:** 3 geometrías (18 celdas), 15 % del dataset, particionado por geometría. Regla de uso único: una sola evaluación final (Etapa 20), registrada en bitácora.

---

## 6. Metodología del factor de reducción de tiempo

**Procedencia: PRE-REGISTRADO.** Se ejecuta verbatim sobre el cache del pase único de la Etapa 20.

1. **Ambas curvas del MISMO pase de la Etapa 20** (mejorada de fracción f contra cruda de fracción f'): el factor N = f'/f es el f' cruda que iguala la métrica de la mejorada de f.
2. **Grilla cruda `F' = {0,01; 0,02; 0,05; 0,10; 0,20; 0,30; 0,50; 0,75; 0,90}`**, con **K = 20** sembrado (mejora deliberada y documentada sobre el K=5 de E13); semillas `SeedSequence([seed(22, idx_frac, k), geo, vista])`.
3. **Ambas convenciones:** cruda (comparable con la literatura) y corregida 1/f (cota conservadora).
4. **Media por celda** (geometría, vista).
5. **Guard de monotonía por celda:** PAVA + conteo de celdas afectadas.
6. **Interpolación PCHIP sobre `log₁₀ f'`.**
7. **Brent** en [0,01; 0,90]; `N = f'/f`.
8. **Fronteras:** si la métrica de la mejorada supera la curva en f'=0,90 → cota inferior «≥ 0,90/f»; si cae por debajo de la curva en f'=0,01 → «sin ganancia medible». La **cota estructural `N ≤ 1/f` se imprime siempre** (2× / 10× / 20× / 100× para 50 / 10 / 5 / 1 %).
9. **Incertidumbre:** bootstrap cluster de 2 etapas, **B = 10.000**, re-derivando AMBAS curvas por réplica.
10. Días equivalentes = 30 · f'.
11. **Por métrica** (SSIM primaria, PSNR, CNR con el cross-check analítico `N_CNR = (razón de CNR)²` del conteo Poisson) **y por tipo de anomalía** (denominador de 2 instancias por tipo → IC ancho, reportado como tal).
12. **Una fila por método** (crudo / clásica / U-Net L1 / cWGAN-GP) **+ columna de factor incremental atribuible al modelo aprendido.**
13. **Límite de aplicabilidad** = el menor f que cumple S1 + S3.
14. **Contraste con O'Donnell et al. (2025):** su 21×/31× escalar frente a nuestro factor como **función de f**; el 1 % (cota estructural 100×) es el punto apto de comparación.

**Proyecciones de VALIDACIÓN (a re-derivar en prueba, ítem de honestidad H9/D11).** El componente aprendido incremental es pequeño y DEBE reportarse por método: al 1 %, la línea base clásica sola ya proyecta **≈ 21×**, la U-Net L1 **≈ 25×** y la cWGAN-GP **≈ 25×** — es decir, gran parte del factor absoluto es la **restauración de amplitud 1/f**, no el aprendizaje. Descubrir esto al redactar parecería ocultamiento; pre-registrarlo lo convierte en hallazgo. Lente interpretativo complementario: la salida mejorada equivale a ≈ 9–11 días crudos constantes para f ≤ 10 %.

**Techo estocástico de referencia (regenerado por script, `evaluation/ceiling.py`, `evaluation/dryrun-artifacts/ceiling.json`):** SSIM = **0,9048 ± 0,0042** (referencia de 30 días; PSNR = 31,22 dB), con el **techo de partición de media-estadística = 0,6554** como ancla exacta y libre de supuestos (cota exacta del techo de una exposición de 15 días vía *thinning* binomial complementario). El techo se toca al 50 % (0,9054 ≈ 0,905), por lo que no se requiere extrapolación.

---

## 7. Cronograma de la Fase B (ejecución real)

**Procedencia: DE FACTO.** El cronograma prospectivo de la v1.0 se sustituye por las fechas reales de ejecución. Estado completo y trazable en `docs/planning/plan-30-stages.md`.

| Hito | Fecha real | Referencia |
|---|---|---|
| Fase B ejecutada de punta a punta (motor surrogate analítico) — validación del diseño | 2026-08-05 | tag `dataset-v1.0-rc1-surrogate` |
| Validación del generador (Etapa 6, EcoMug real; calibración S0) | Fase B | `docs/reports/stage-06-flux-validation.md` |
| Catálogo de 19 geometrías (Etapa 7) | 2026-08-05 | `docs/reports/stage-07-geometry-catalog.md` |
| Pipeline integrado y V&V físico (Etapas 8–9) | Fase B | `docs/reports/stage-08-pipeline-pilot.md`, `stage-09-physics-vv.md` |
| Campaña masiva Geant4 en SC3 (Etapa 10): 114/114 vistas, 4,1 × 10⁹ muones, 30 días calibrados/vista | 2026-08-30/31 | `docs/reports/phase-b-sc3-runbook.md`, `stage-10-campaign.md` |
| Pares de entrenamiento por thinning binomial (Etapa 11): 9.120 pares | Fase B | `docs/reports/stage-11-dataset-pairs.md` |
| Congelamiento del dataset (Etapa 12) | 2026-08-31 | tag `dataset-v1.0` (commit c4487f3) |

La Fase B se re-ejecutó y **cerró con el motor definitivo Geant4 11.4.2** en el clúster SC3 de la UIS (2026-08-30/31), superando el rc1 surrogate del 2026-08-05. Las firmas de cierre de OE1 (director y codirector) quedan como acción humana pendiente (PA-4).

---

## 8. Control de cambios post-congelamiento

**Procedencia: transversal.**

Este documento queda **CONGELADO** el 2026-09-05 y se etiqueta `protocol-v1.1` en Git. Todo cambio posterior requiere, sin excepción:

1. Nueva fila en el historial de versiones (§ encabezado).
2. Justificación escrita como **adenda fechada** dentro de este documento.
3. **Visto bueno del director.**
4. Nuevo tag (`protocol-v1.2`, …).

Los cambios **NUNCA** se aplican retroactivamente a resultados ya medidos. En particular: cualquier cambio de un umbral, prueba estadística o unidad de análisis **después** de la lectura del conjunto de prueba (Etapa 20) exige adenda fechada + firma del director; el intervalo de confianza porta la afirmación y no se re-analiza (§5.4).

---

## 9. Aprobación

**Procedencia: transversal.**

| Rol | Nombre | Estado / fecha |
|---|---|---|
| Autor | Daniel Andrés Pinto Ortega | Congelado por los autores — 2026-09-05 |
| Autor | Reinaldo Cárdenas Tirado | Congelado por los autores — 2026-09-05 |
| Director | Andrés Leonardo González Gómez | Ratificación pendiente (PA-4) |
| Codirector | Jaime Enrique Meneses Fonseca | Ratificación pendiente (PA-4) |

Los autores congelan este pre-registro con efecto inmediato para no detener el cronograma, asumiendo el mismo tipo de riesgo de re-trabajo documentado en `docs/actas/decision-framework-dl/authors-provisional-decision.md`. La ratificación del director y del codirector es un requisito de cierre (PA-4) y **la lectura del conjunto de prueba (Etapa 20) espera de todos modos el GO explícito**; correr la Etapa 20 antes de ratificar cambiaría un activo irreversible por cronograma, decisión que es del usuario y queda documentada.

---

## Procedencia — resumen

| Sección | Contenido | Procedencia |
|---|---|---|
| §1 | Objetivo y alcance | Transversal |
| §2 | Escenarios geométricos (19 geometrías, anomalías, edificio) | **DE FACTO** |
| §3 | Parametrización del detector virtual | **DE FACTO** |
| §4 | Niveles de estadística y equivalencia temporal (S0, días, K, pares) | **DE FACTO** |
| §5.1 | Definición de SSIM/PSNR (transcripción del código congelado) | **DE FACTO** |
| §5.2 | Fórmulas de detectabilidad y umbral `z_thr` calibrado por brazo | **PRE-REGISTRADO** |
| §5.3 | Umbrales de éxito S1/S2/S3 y constantes de clasificación | **PRE-REGISTRADO** |
| §5.4 | Plan estadístico, uso único, protocolo de caída, potencia, reporte honesto | **PRE-REGISTRADO** |
| §6 | Metodología del factor de reducción de tiempo | **PRE-REGISTRADO** |
| §7 | Cronograma real de la Fase B | **DE FACTO** |
| §8 | Control de cambios post-congelamiento | Transversal |
| §9 | Aprobación | Transversal |

**Declaración de cierre:** únicamente **§5.2–§5.4 y §6** constituyen pre-registro genuino contra un conjunto de prueba intacto (medidos sobre validación en el dry-run del paso 7 de la Etapa 19). La transcripción de fórmulas de §5.1 y las secciones §2–§4 y §7 se documentan a posteriori desde artefactos congelados de las Fases B y C (DE FACTO) y no son pre-registro; esta distinción se traslada verbatim a las amenazas a la validez de la Etapa 23.
