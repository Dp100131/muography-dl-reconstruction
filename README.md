# Muography DL Reconstruction

Código y resultados del trabajo de grado **"Reconstrucción muográfica por transmisión asistida por aprendizaje profundo mediante mejora predictiva de imágenes de baja estadística para exploración volcánica"**, presentado por Reinaldo Cárdenas Tirado y Daniel Andrés Pinto Ortega, bajo la dirección de Andrés Leonardo González Gómez (PhD(c)) y la codirección de Jaime Enrique Meneses Fonseca (PhD), en la Universidad Industrial de Santander, Escuela de Ingeniería de Sistemas e Informática, grupo GOTS (Óptica y Tratamiento de Señales, Escuela de Física). El proyecto entrena un cWGAN-GP condicional que reconstruye imágenes muográficas de alta estadística a partir de adquisiciones cortas, usando pares generados con Geant4 y evaluados contra líneas base clásicas y un U-Net entrenado solo con pérdida L1.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `models/` | Arquitecturas del generador y del crítico, más los manifiestos de pesos congelados (`weights-manifest.json`, `weights-manifest-e19.json`) |
| `scripts/analysis/` | Construcción y congelamiento del dataset, validación de flujo, tablas del generador Geant4 |
| `scripts/simulation/` | Aplicación Geant4, catálogo de geometrías, entorno Docker y modelo surrogado |
| `scripts/training/` | Motor de entrenamiento cWGAN-GP, lanzadores, configuraciones y pruebas |
| `scripts/evaluation/` | Inferencia, métricas, calibración de umbrales, generación de tablas y figuras |
| `scripts/cluster/` | Scripts SLURM y utilidades de despliegue en el clúster SC3 |
| `results/dataset/` | Manifiesto de la campaña, particiones (`splits.json`) y estadísticos base |
| `results/training/` | Registros de entrenamiento por corrida (`config.json`, `summary.json`, `metrics.jsonl`) |
| `results/evaluation/` | Métricas de validación y prueba, umbrales congelados del protocolo, bitácora de acceso al split de prueba |
| `results/simulation/` | Comparación de motores de simulación |
| `figures/results/` | Figuras generadas de resultados por etapa |
| `figures/architecture/` | Diagramas de arquitectura |
| `docs/protocol/` | Protocolo experimental congelado |

## Archivos pesados (Google Drive)

Los datos binarios (shards del dataset, campaña Geant4 cruda, checkpoints y logs SLURM) no se versionan en Git. Están disponibles en:

**Enlace:** <https://drive.google.com/drive/folders/1pYpfEOt1tkg9aLmRyut8_1ls-VgKKvz_?usp=sharing>

El paquete descargado sigue la misma jerarquía que la raíz del repositorio: cada carpeta de nivel superior se coloca directamente sobre la carpeta homónima ignorada por `.gitignore` (`dataset/`, `data/`, `runs/`, `logs/`).

| Carpeta del paquete | Colocar en | Contenido | Tamaño aprox. |
|---|---|---|---|
| `dataset/shards/*.npz` | `dataset/shards/` | 19 shards del dataset (campaña Geant4, sin el surrogate) | 405 MB |
| `data/campaign-geant4/` | `data/campaign-geant4/` | 19 geometrías x 6 vistas, pares `.npz` + `.meta.json` | 9.6 MB |
| `runs/e17-final-*/checkpoints/best.pt` | `runs/e17-final-*/checkpoints/` | 6 checkpoints del cWGAN-GP final (4 niveles + 2 semillas extra al 5 %) | ~280 MB cada uno |
| `runs/e19-unetl1-*/checkpoints/best.pt` | `runs/e19-unetl1-*/checkpoints/` | 5 checkpoints del U-Net L1 reentrenado en la Etapa 19.5 | ~352 MB cada uno |
| `logs/simulation/*.out` | `logs/simulation/` | 199 logs SLURM de la campaña de simulación (incluye los `muo-smoke-*.out`) | 
| `logs/training-evaluation/*.out` | `logs/training-evaluation/` | 97 logs SLURM de entrenamiento y evaluación |

Tamaño total del paquete: 3.7 GB en 554 archivos.

Verificación de integridad, ejecutada desde la raíz del paquete descargado:

```bash
sha256sum -c SHA256SUMS
```

Todas las líneas deben reportar `OK`. Los hashes de referencia también viven en el repositorio, por si se quiere comprobar un archivo individual sin descargar el resto del paquete:

- `results/dataset/splits.json` (campo `shard_sha256`) para los shards del dataset.
- `models/weights-manifest.json` para los checkpoints `e17-final-*`.
- `models/weights-manifest-e19.json` para los checkpoints `e19-unetl1-*`.

## Qué modelo es cada checkpoint

- **`e17-final-f50` / `f10` / `f5` / `f1`**: cWGAN-GP final por nivel de estadística (50 %, 10 %, 5 %, 1 % de conteos), Etapa 17.
- **`e17-final-f5-s1` / `f5-s2`**: dos semillas adicionales del mismo cWGAN-GP al 5 %, para estimar variabilidad entre corridas.
- **`e19-unetl1-f50` / `f10` / `f5` / `f1` / `f1-shared`**: línea base U-Net entrenada solo con pérdida L1 (sin crítico), reentrenada en la Etapa 19.5 para servir de comparación en el mismo protocolo. `f1-shared` usa la convención de escala "shared" en vez de "ref_equivalent"; las demás usan `ref_equivalent`.
- Los checkpoints efectivamente usados en el pase de prueba de la Etapa 20 están listados en `results/evaluation/test/summary.json` (campo `model_provenance`); ese archivo es la fuente de verdad sobre qué arma corresponde a cada modelo en la evaluación final.

## Entorno

Python 3.9 o superior (los checkpoints publicados se entrenaron con Python 3.9 en el clúster; el desarrollo local usa Python 3.13).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` ya incluye `--extra-index-url https://download.pytorch.org/whl/cu118` y fija `torch==2.6.0+cu118`. Para una GPU con otra versión de CUDA, instalar primero el wheel de PyTorch correspondiente desde pytorch.org y luego correr `pip install -r requirements.txt` comentando la línea de `torch`.

## Uso

### Evaluar con los pesos publicados

```bash
python scripts/evaluation/run_evaluation.py --split val --skip-gpu
```

El split `val` es de uso libre y repetible; así se calibró el protocolo v1.1. El split `test` se abre una sola vez (Etapa 20) y esa apertura ya quedó registrada en `results/evaluation/test/test_access_log.json`; no debe repetirse fuera de una nueva ronda de evaluación planeada y documentada.

### Reentrenar

cWGAN-GP, a partir de una configuración congelada:

```bash
python scripts/training/train_cwgan_gp.py --config scripts/training/configs/e17_final.json
```

U-Net L1 (línea base sin crítico, mismo dataset y semillas):

```bash
python scripts/training/train_unet_l1.py --fraction 0.05 --seed 20260805130200
```

Ambos comparten las mismas banderas (`--fraction`, `--epochs`, `--lr`, `--seed`, `--run-id`, etc.); ver `scripts/training/configs/` para las configuraciones de referencia de cada etapa.

### Reconstruir el dataset a partir de las vistas de campaña

```bash
MUOGRAPHY_ENGINE=geant4 python scripts/analysis/build_dataset.py
```

### Ejecutar la simulación Geant4

```bash
docker compose -f scripts/simulation/docker/docker-compose.yml run --rm sim-batch
```

### Regenerar tablas y figuras

```bash
python scripts/evaluation/report_figures.py --cache results/evaluation/val
python scripts/evaluation/report_tables.py --cache results/evaluation/val --out results/evaluation/fragments
```

### Ejecutar las pruebas

Cada archivo `test_*.py` es un script autocontenido y verificable por sí mismo (sin pytest); se ejecuta directamente con Python:

```bash
python scripts/evaluation/test_metrics.py
python scripts/training/test_model_units.py
```

Para correr toda la batería:

```bash
for f in scripts/evaluation/test_*.py scripts/training/test_*.py; do python "$f" || break; done
```

## Configuraciones de la Etapa 19.5

Los cinco reentrenamientos U-Net L1 (`e19-unetl1-f50`, `f10`, `f5`, `f1`, `f1-shared`) tienen su configuración completa en `results/training/records/e19-unetl1-*/config.json`. El campo `config_sha8` de cada uno está verificado contra `models/weights-manifest-e19.json`.

## Protocolo de evaluación

El protocolo congelado vigente es `docs/protocol/protocol-v1.1.md`. Los umbrales derivados de él están en `results/evaluation/protocol_thresholds.json`.

El documento se incluye textual, tal como se congeló antes del pase de prueba, así que sus rutas siguen la organización del repositorio de trabajo donde se redactó. En este repositorio corresponden a:

| Ruta citada en el protocolo | Ruta en este repositorio |
|---|---|
| `evaluation/…`, `training/…`, `analysis/…`, `simulation/…` | `scripts/evaluation/…`, `scripts/training/…`, `scripts/analysis/…`, `scripts/simulation/…` |
| `cluster/sc3/…` | `scripts/cluster/…` |
| `model/weights-manifest*.json` | `models/weights-manifest*.json` |
| `docs/protocol/protocol_thresholds.json` | `results/evaluation/protocol_thresholds.json` |
| `evaluation/dryrun-artifacts/…` | `results/evaluation/fragments/…` y `results/evaluation/*.json` |
| `docs/reports/…`, `docs/planning/…`, `protocol-v1.0.md` | Documentos del informe de tesis; no se incluyen en este repositorio |

## Cita

Cárdenas Tirado, R., & Pinto Ortega, D. A. (2026). *Reconstrucción muográfica por transmisión asistida por aprendizaje profundo mediante mejora predictiva de imágenes de baja estadística para exploración volcánica* [Trabajo de grado, Universidad Industrial de Santander, Escuela de Ingeniería de Sistemas e Informática].
