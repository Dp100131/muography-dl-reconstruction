# Licencias y alcance

El repositorio contiene materiales sujetos a licencias diferentes. No se aplica una licencia única a todo su contenido. Las asignaciones siguientes se limitan a los derechos de los titulares del material propio; las licencias, avisos y condiciones de terceros prevalecen para sus respectivos componentes.

| Material | Licencia | Alcance |
|---|---|---|
| Código fuente propio | [Apache-2.0](LICENSES/Apache-2.0.txt) | Archivos de código y scripts de construcción propios, salvo el componente EcoMug indicado abajo y código de terceros. |
| Integración directa de EcoMug | [GPL-3.0-or-later](LICENSES/GPL-3.0.txt) | `scripts/simulation/ecomug_standalone/`. La distribución del programa combinado con EcoMug debe cumplir GPL. |
| EcoMug.h incorporado | [GPL-3.0-or-later](LICENSES/GPL-3.0.txt) | `scripts/simulation/third_party/ecomug/EcoMug.h`, conforme a su cabecera original. |
| nlohmann/json | [MIT](LICENSES/nlohmann-json-MIT.txt) y avisos internos | `scripts/simulation/third_party/json/json.hpp`; se conservan todos sus avisos internos. |
| Datos sintéticos y resultados propios | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) | Matrices, imágenes simuladas, tablas, figuras y predicciones producidas por el proyecto, incluidas las de `results/`, `figures/` y la distribución externa de archivos pesados enlazada en el [README](README.md#archivos-pesados-google-drive). |
| Pesos propios | [Apache-2.0](LICENSES/Apache-2.0.txt) | Parámetros numéricos de los modelos identificados en `models/weights-manifest.json` y `models/weights-manifest-e19.json`. No se extiende a código de terceros que pudiera estar incluido en un contenedor. |
| Documentación propia | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) | README, bitácoras y documentos de `docs/`. Los fragmentos de código conservan su licencia de software. |

La atribución del trabajo de grado corresponde a Reinaldo Cárdenas Tirado y Daniel Andrés Pinto Ortega. La dirección y codirección se reconocen según el README; esa función no sustituye la autoría ni implica titularidad exclusiva sobre todos los materiales. Se conservan las atribuciones específicas de cada archivo y contribución.

Quedan fuera de las concesiones sobre material propio las obras citadas, figuras ajenas, logotipos, marcas, plantillas, fuentes tipográficas y dependencias externas. Su presencia o mención no implica autorización para reutilizarlas bajo Apache-2.0 o CC BY 4.0. Las excepciones legales aplicables no se restringen.

Estas licencias permiten usos comerciales conforme a sus condiciones y no certifican la validez científica ni la aptitud de los resultados para decisiones operativas. Las limitaciones científicas se mantienen en el README.

La asignación se refiere a los artefactos del proyecto identificados por sus manifiestos, no a futuras adiciones arbitrarias a la carpeta externa. Los permisos de descarga de Drive son independientes de la licencia de reutilización.

Los textos de licencia de [LICENSES/](LICENSES/) son copias sin modificar de las publicadas en el repositorio documental del proyecto, [muografia-documentada](https://github.com/anlegogo/muografia-documentada), que a su vez proceden de:

- https://www.gnu.org/licenses/gpl-3.0.txt
- https://www.apache.org/licenses/LICENSE-2.0.txt
- https://creativecommons.org/licenses/by/4.0/legalcode.txt
- https://raw.githubusercontent.com/nlohmann/json/v3.11.3/LICENSE.MIT

Véanse los [avisos de terceros](THIRD_PARTY_NOTICES.md).
