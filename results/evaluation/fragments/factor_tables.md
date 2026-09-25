### Factor por método — curva cruda, métrica SSIM

| Nivel | Método | f' | N | IC 95 % (bootstrap) | N incremental | Cota estructural | Días equiv. | Estado | n celdas |
|---|---|---|---|---|---|---|---|---|---|
| 50 % | crudo 1/f (identidad) | 0.5485 | 1.10× | [1.10, 1.10] | 0.00× | 2× | 16.5 | factor | 24 |
| 50 % | mejor clásica | 0.5501 | 1.10× | [1.10, 1.10] | 0.00× | 2× | 16.5 | factor | 24 |
| 50 % | U-Net L1 | 0.5748 | 1.15× | [1.15, 1.15] | 0.05× | 2× | 17.2 | factor | 24 |
| 50 % | cWGAN-GP | 0.5830 | 1.17× | [1.15, 1.18] | 0.07× | 2× | 17.5 | factor | 24 |
| 10 % | crudo 1/f (identidad) | 0.1369 | 1.37× | [1.36, 1.37] | 0.00× | 10× | 4.1 | factor | 24 |
| 10 % | mejor clásica | 0.3372 | 3.37× | [3.32, 3.43] | 2.00× | 10× | 10.1 | factor | 24 |
| 10 % | U-Net L1 | 0.3305 | 3.31× | [3.25, 3.37] | 1.94× | 10× | 9.9 | factor | 24 |
| 10 % | cWGAN-GP | 0.3576 | 3.58× | [3.53, 3.63] | 2.21× | 10× | 10.7 | factor | 24 |
| 5 % | crudo 1/f (identidad) | 0.0704 | 1.41× | [1.40, 1.41] | 0.00× | 20× | 2.1 | factor | 24 |
| 5 % | mejor clásica | 0.3234 | 6.47× | [6.36, 6.59] | 5.06× | 20× | 9.7 | factor | 24 |
| 5 % | U-Net L1 | 0.3213 | 6.43× | [6.38, 6.48] | 5.02× | 20× | 9.6 | factor | 24 |
| 5 % | cWGAN-GP | 0.3331 | 6.66× | [6.56, 6.78] | 5.25× | 20× | 10.0 | factor | 24 |
| 1 % | crudo 1/f (identidad) | 0.0125 | 1.25× | [1.25, 1.26] | 0.00× | 100× | 0.4 | factor | 24 |
| 1 % | mejor clásica | 0.2759 | 27.59× | [27.15, 28.09] | 26.34× | 100× | 8.3 | factor | 24 |
| 1 % | U-Net L1 | 0.3161 | 31.61× | [31.34, 31.93] | 30.36× | 100× | 9.5 | factor | 24 |
| 1 % | cWGAN-GP | 0.3126 | 31.26× | [30.67, 31.92] | 30.00× | 100× | 9.4 | factor | 24 |

### Factor por método — curva cruda, métrica PSNR [dB]

| Nivel | Método | f' | N | IC 95 % (bootstrap) | N incremental | Cota estructural | Días equiv. | Estado | n celdas |
|---|---|---|---|---|---|---|---|---|---|
| 50 % | crudo 1/f (identidad) | 0.8117 | 1.62× | [1.62, 1.63] | 0.00× | 2× | 24.4 | factor | 24 |
| 50 % | mejor clásica | 0.8120 | 1.62× | [1.62, 1.63] | 0.00× | 2× | 24.4 | factor | 24 |
| 50 % | U-Net L1 | 0.8500 | 1.70× | [1.70, 1.70] | 0.08× | 2× | 25.5 | factor | 24 |
| 50 % | cWGAN-GP | 0.8484 | 1.70× | [1.70, 1.70] | 0.07× | 2× | 25.5 | factor | 24 |
| 10 % | crudo 1/f (identidad) | 0.5888 | 5.89× | [5.82, 5.96] | 0.00× | 10× | 17.7 | factor | 24 |
| 10 % | mejor clásica | 0.8016 | 8.02× | [7.98, 8.05] | 2.13× | 10× | 24.0 | factor | 24 |
| 10 % | U-Net L1 | 0.8034 | 8.03× | [8.00, 8.07] | 2.15× | 10× | 24.1 | factor | 24 |
| 10 % | cWGAN-GP | 0.8060 | 8.06× | [8.03, 8.09] | 2.17× | 10× | 24.2 | factor | 24 |
| 5 % | crudo 1/f (identidad) | 0.5076 | 10.15× | [10.04, 10.27] | 0.00× | 20× | 15.2 | factor | 24 |
| 5 % | mejor clásica | 0.7936 | 15.87× | [15.81, 15.94] | 5.72× | 20× | 23.8 | factor | 24 |
| 5 % | U-Net L1 | 0.7928 | 15.86× | [15.79, 15.92] | 5.70× | 20× | 23.8 | factor | 24 |
| 5 % | cWGAN-GP | 0.8019 | 16.04× | [15.98, 16.10] | 5.89× | 20× | 24.1 | factor | 24 |
| 1 % | crudo 1/f (identidad) | 0.2654 | 26.54× | [26.11, 26.94] | 0.00× | 100× | 8.0 | factor | 24 |
| 1 % | mejor clásica | 0.7342 | 73.42× | [72.95, 73.93] | 46.88× | 100× | 22.0 | factor | 24 |
| 1 % | U-Net L1 | 0.7935 | 79.35× | [79.22, 79.48] | 52.80× | 100× | 23.8 | factor | 24 |
| 1 % | cWGAN-GP | 0.7972 | 79.72× | [79.46, 80.00] | 53.18× | 100× | 23.9 | factor | 24 |

### Factor por método — curva corregida 1/f, métrica SSIM

| Nivel | Método | f' | N | IC 95 % (bootstrap) | N incremental | Cota estructural | Días equiv. | Estado | n celdas |
|---|---|---|---|---|---|---|---|---|---|
| 50 % | crudo 1/f (identidad) | 0.4996 | 1.00× | [1.00, 1.00] | 0.00× | 2× | 15.0 | factor | 24 |
| 50 % | mejor clásica | 0.5014 | 1.00× | [1.00, 1.00] | 0.00× | 2× | 15.0 | factor | 24 |
| 50 % | U-Net L1 | 0.5292 | 1.06× | [1.06, 1.06] | 0.06× | 2× | 15.9 | factor | 24 |
| 50 % | cWGAN-GP | 0.5385 | 1.08× | [1.06, 1.09] | 0.08× | 2× | 16.2 | factor | 24 |
| 10 % | crudo 1/f (identidad) | 0.0999 | 1.00× | [1.00, 1.00] | 0.00× | 10× | 3.0 | factor | 24 |
| 10 % | mejor clásica | 0.2758 | 2.76× | [2.70, 2.82] | 1.76× | 10× | 8.3 | factor | 24 |
| 10 % | U-Net L1 | 0.2693 | 2.69× | [2.63, 2.76] | 1.69× | 10× | 8.1 | factor | 24 |
| 10 % | cWGAN-GP | 0.2962 | 2.96× | [2.91, 3.02] | 1.96× | 10× | 8.9 | factor | 24 |
| 5 % | crudo 1/f (identidad) | 0.0500 | 1.00× | [1.00, 1.00] | 0.00× | 20× | 1.5 | factor | 24 |
| 5 % | mejor clásica | 0.2623 | 5.25× | [5.13, 5.38] | 4.25× | 20× | 7.9 | factor | 24 |
| 5 % | U-Net L1 | 0.2602 | 5.20× | [5.14, 5.27] | 4.20× | 20× | 7.8 | factor | 24 |
| 5 % | cWGAN-GP | 0.2718 | 5.44× | [5.32, 5.57] | 4.44× | 20× | 8.2 | factor | 24 |
| 1 % | crudo 1/f (identidad) | 0.0100 | 1.00× | [1.00, 1.00] | 0.00× | 100× | 0.3 | factor | 24 |
| 1 % | mejor clásica | 0.2176 | 21.76× | [21.29, 22.29] | 20.76× | 100× | 6.5 | factor | 24 |
| 1 % | U-Net L1 | 0.2552 | 25.52× | [25.19, 25.91] | 24.52× | 100× | 7.7 | factor | 24 |
| 1 % | cWGAN-GP | 0.2518 | 25.18× | [24.56, 25.89] | 24.18× | 100× | 7.6 | factor | 24 |

### Factor por método — curva corregida 1/f, métrica PSNR [dB]

| Nivel | Método | f' | N | IC 95 % (bootstrap) | N incremental | Cota estructural | Días equiv. | Estado | n celdas |
|---|---|---|---|---|---|---|---|---|---|
| 50 % | crudo 1/f (identidad) | 0.4999 | 1.00× | [1.00, 1.00] | 0.00× | 2× | 15.0 | factor | 24 |
| 50 % | mejor clásica | 0.5010 | 1.00× | [1.00, 1.00] | 0.00× | 2× | 15.0 | factor | 24 |
| 50 % | U-Net L1 | 0.6245 | 1.25× | [1.24, 1.25] | 0.25× | 2× | 18.7 | factor | 24 |
| 50 % | cWGAN-GP | 0.6194 | 1.24× | [1.22, 1.26] | 0.24× | 2× | 18.6 | factor | 24 |
| 10 % | crudo 1/f (identidad) | 0.0999 | 1.00× | [1.00, 1.00] | 0.00× | 10× | 3.0 | factor | 24 |
| 10 % | mejor clásica | 0.4683 | 4.68× | [4.67, 4.70] | 3.68× | 10× | 14.1 | factor | 24 |
| 10 % | U-Net L1 | 0.4739 | 4.74× | [4.72, 4.76] | 3.74× | 10× | 14.2 | factor | 24 |
| 10 % | cWGAN-GP | 0.4822 | 4.82× | [4.80, 4.84] | 3.82× | 10× | 14.5 | factor | 24 |
| 5 % | crudo 1/f (identidad) | 0.0499 | 1.00× | [1.00, 1.00] | 0.00× | 20× | 1.5 | factor | 24 |
| 5 % | mejor clásica | 0.4442 | 8.88× | [8.85, 8.92] | 7.89× | 20× | 13.3 | factor | 24 |
| 5 % | U-Net L1 | 0.4416 | 8.83× | [8.78, 8.89] | 7.83× | 20× | 13.2 | factor | 24 |
| 5 % | cWGAN-GP | 0.4695 | 9.39× | [9.35, 9.43] | 8.39× | 20× | 14.1 | factor | 24 |
| 1 % | crudo 1/f (identidad) | 0.0100 | 1.00× | [1.00, 1.00] | 0.00× | 100× | 0.3 | no_gain | 24 |
| 1 % | mejor clásica | 0.2986 | 29.86× | [29.67, 30.04] | 28.86× | 100× | 9.0 | factor | 24 |
| 1 % | U-Net L1 | 0.4437 | 44.37× | [43.58, 45.05] | 43.37× | 100× | 13.3 | factor | 24 |
| 1 % | cWGAN-GP | 0.4550 | 45.50× | [45.16, 45.83] | 44.50× | 100× | 13.7 | factor | 24 |

### Factor CNR por tipo de anomalía (ancla analítica N = (razón CNR)², conteo Poisson)

| Nivel | Método | Tipo de anomalía | Razón CNR | N = (razón)² | IC 95 % | Cota estructural | n celdas |
|---|---|---|---|---|---|---|---|
| 50 % | mejor clásica | cavity | 1.000 | 1.00× | [1.00, 1.00] | 2× | 18 |
| 50 % | mejor clásica | conduit | 1.000 | 1.00× | [1.00, 1.00] | 2× | 18 |
| 50 % | mejor clásica | dense_body | 0.999 | 1.00× | [0.99, 1.00] | 2× | 12 |
| 50 % | U-Net L1 | cavity | 1.676 | 2.81× | [1.85, 4.10] | 2× | 18 |
| 50 % | U-Net L1 | conduit | 1.054 | 1.11× | [0.92, 1.38] | 2× | 18 |
| 50 % | U-Net L1 | dense_body | 1.711 | 2.93× | [2.05, 4.27] | 2× | 12 |
| 50 % | cWGAN-GP | cavity | 1.675 | 2.81× | [1.91, 4.09] | 2× | 18 |
| 50 % | cWGAN-GP | conduit | 1.000 | 1.00× | [0.80, 1.29] | 2× | 18 |
| 50 % | cWGAN-GP | dense_body | 1.998 | 3.99× | [2.69, 6.08] | 2× | 12 |
| 10 % | mejor clásica | cavity | 2.159 | 4.66× | [0.02, 13.03] | 10× | 18 |
| 10 % | mejor clásica | conduit | 1.116 | 1.25× | [0.82, 1.63] | 10× | 18 |
| 10 % | mejor clásica | dense_body | 6.716 | 45.10× | [19.77, 89.49] | 10× | 12 |
| 10 % | U-Net L1 | cavity | -0.865 | 0.75× | [0.01, 141.69] | 10× | 18 |
| 10 % | U-Net L1 | conduit | 0.794 | 0.63× | [0.08, 1.25] | 10× | 18 |
| 10 % | U-Net L1 | dense_body | 4.065 | 16.53× | [6.53, 31.19] | 10× | 12 |
| 10 % | cWGAN-GP | cavity | 0.174 | 0.03× | [0.00, 58.07] | 10× | 18 |
| 10 % | cWGAN-GP | conduit | 0.762 | 0.58× | [0.06, 1.20] | 10× | 18 |
| 10 % | cWGAN-GP | dense_body | 3.568 | 12.73× | [7.13, 20.25] | 10× | 12 |
| 5 % | mejor clásica | cavity | 6.581 | 43.31× | [16.31, 117.18] | 20× | 18 |
| 5 % | mejor clásica | conduit | 1.034 | 1.07× | [0.24, 1.90] | 20× | 18 |
| 5 % | mejor clásica | dense_body | 8.379 | 70.21× | [31.51, 138.73] | 20× | 12 |
| 5 % | U-Net L1 | cavity | 9.693 | 93.96× | [12.75, 378.51] | 20× | 18 |
| 5 % | U-Net L1 | conduit | 0.782 | 0.61× | [0.02, 1.42] | 20× | 18 |
| 5 % | U-Net L1 | dense_body | 5.307 | 28.17× | [13.57, 47.58] | 20× | 12 |
| 5 % | cWGAN-GP | cavity | 13.911 | 193.53× | [33.06, 724.40] | 20× | 18 |
| 5 % | cWGAN-GP | conduit | 0.558 | 0.31× | [0.00, 1.44] | 20× | 18 |
| 5 % | cWGAN-GP | dense_body | 5.551 | 30.81× | [13.16, 55.94] | 20× | 12 |
| 1 % | mejor clásica | cavity | 16.847 | 283.83× | [0.26, 2056.52] | 100× | 18 |
| 1 % | mejor clásica | conduit | -9.754 | 95.14× | [2.17, 1071.44] | 100× | 18 |
| 1 % | mejor clásica | dense_body | 8.163 | 66.63× | [43.98, 93.27] | 100× | 12 |
| 1 % | U-Net L1 | cavity | 77.743 | 6044.01× | [104.43, 31196.67] | 100× | 18 |
| 1 % | U-Net L1 | conduit | -7.289 | 53.14× | [3.37, 689.24] | 100× | 18 |
| 1 % | U-Net L1 | dense_body | 11.767 | 138.46× | [61.21, 249.12] | 100× | 12 |
| 1 % | cWGAN-GP | cavity | 100.346 | 10069.34× | [144.74, 52747.66] | 100× | 18 |
| 1 % | cWGAN-GP | conduit | -6.026 | 36.31× | [2.60, 488.92] | 100× | 18 |
| 1 % | cWGAN-GP | dense_body | 12.900 | 166.41× | [64.65, 323.38] | 100× | 12 |


Incertidumbre: bootstrap cluster de dos etapas (celdas con reemplazo, luego K dentro de cada celda sorteada), B = 10000, ambas curvas re-derivadas por réplica (sección 6, paso 9).

**Cotas estructurales N ≤ 1/f: f=0.01 → 100×, f=0.05 → 20×, f=0.1 → 10×, f=0.5 → 2×.** Ningún método puede valer más que la cota: el 100 % de los eventos es todo lo que hay.

La columna «N incremental» es el factor del método menos el de la identidad cruda 1/f: separa la restauración determinista de amplitud de lo aportado por el modelo aprendido (hallazgo H9 del plan de ejecución, pre-registrado).
