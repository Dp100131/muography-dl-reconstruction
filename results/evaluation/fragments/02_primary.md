| Nivel | Comparación | Δ media por celda | IC 95 % | Método | Wilcoxon p | p ajustado (Holm) | Rechaza H0 | n celdas |
|---|---|---|---|---|---|---|---|---|
| 50 % | cWGAN − crudo 1/f | 0.0115 | [0.0079, 0.0143] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 50 % | cWGAN − clásica congelada | 0.0108 | [0.0070, 0.0139] | bootstrap-percentile (Shapiro rejected) | 1.53e-05 | 9.16e-05 | sí | 18 |
| 50 % | cWGAN − U-Net L1 | 0.0020 | [-0.0012, 0.0046] | bootstrap-percentile (Shapiro rejected) | 2.29e-01 | 4.58e-01 | no | 18 |
| 10 % | cWGAN − crudo 1/f | 0.2076 | [0.2036, 0.2127] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 10 % | cWGAN − clásica congelada | 0.0123 | [0.0115, 0.0131] | student-t | 7.63e-06 | 9.16e-05 | sí | 18 |
| 10 % | cWGAN − U-Net L1 | 0.0165 | [0.0147, 0.0181] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 5 % | cWGAN − crudo 1/f | 0.3440 | [0.3371, 0.3527] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 5 % | cWGAN − clásica congelada | 0.0063 | [0.0059, 0.0068] | student-t | 7.63e-06 | 9.16e-05 | sí | 18 |
| 5 % | cWGAN − U-Net L1 | 0.0080 | [0.0059, 0.0107] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 1 % | cWGAN − crudo 1/f | 0.6521 | [0.6412, 0.6656] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 1 % | cWGAN − clásica congelada | 0.0263 | [0.0258, 0.0270] | bootstrap-percentile (Shapiro rejected) | 7.63e-06 | 9.16e-05 | sí | 18 |
| 1 % | cWGAN − U-Net L1 | -0.0028 | [-0.0057, 0.0007] | bootstrap-percentile (Shapiro rejected) | 2.29e-01 | 4.58e-01 | no | 18 |

Familia primaria de 12 hipótesis, Holm–Bonferroni α = 0.05. El intervalo porta la afirmación; el valor p lo acompaña. Unidad de análisis = celda (geometría, vista).
