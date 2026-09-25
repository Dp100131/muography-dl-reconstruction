| Nivel | Brazo | Preservadas | IC 95 % | Perdidas | Infladas | Alucinación (sham) | n ROI | n sham | Ref. no detectable | ROI excluidas (<64 px) |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 % | crudo 1/f | 0.0% | [0.0%, 0.0%] | 100.0% | 0.0% | 0.1% | 20 | 3600 | 700 | 0 |
| 50 % | clásica congelada | 0.0% | [0.0%, 0.0%] | 100.0% | 0.0% | 0.1% | 20 | 3600 | 700 | 0 |
| 10 % | crudo 1/f | 0.0% | [0.0%, 0.0%] | 81.2% | 18.8% | 0.2% | 80 | 3600 | 640 | 0 |
| 5 % | crudo 1/f | 0.0% | [0.0%, 0.0%] | 78.9% | 21.1% | 0.4% | 180 | 3600 | 540 | 0 |
| 1 % | crudo 1/f | 0.0% | [0.0%, 0.0%] | 82.3% | 17.7% | 0.9% | 440 | 3600 | 280 | 0 |
| — | **ROI excluidas (todas las filas)** | — | — | — | — | — | — | — | — | **0** |

z_thr calibrado por (brazo, nivel) — percentil 99 de |z| sobre ROIs sham de validación de CADA brazo (escalar de respaldo 10.464). «Ref. no detectable» sale del denominador y se reporta aparte (válvula de la sección 5.3). «ROI excluidas» son las ROI con `roi_valid = 0` (< 64 px o sin anillo medible, sección 5.2): quedan fuera de toda tasa y se cuentan aquí, nunca se descartan en silencio. Las tasas leen banderas independientes, de modo que un caso inflado cuenta como inflado aunque también esté dentro de la tolerancia X.
