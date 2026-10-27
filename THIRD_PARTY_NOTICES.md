# Avisos de terceros

Este inventario identifica los componentes incorporados en el repositorio. No reemplaza sus textos de licencia ni constituye una auditoría exhaustiva de todas las dependencias transitivas.

## EcoMug

- Archivo: `scripts/simulation/third_party/ecomug/EcoMug.h`.
- Aviso conservado: Copyright (C) 2023 Davide Pagano.
- Licencia de la copia incorporada: GNU GPL versión 3 o posterior; [texto GPL v3](LICENSES/GPL-3.0.txt).
- Proyecto: https://github.com/dr4kan/EcoMug
- Referencia: D. Pagano et al., “EcoMug: an Efficient COsmic MUon Generator for cosmic-ray muons applications”, DOI: https://doi.org/10.1016/j.nima.2021.165732
- `scripts/simulation/ecomug_standalone/main.cc` incluye directamente EcoMug.h. El Makefile descarga por defecto la referencia `main` (`ECOMUG_REF ?= main`), que es mutable: no acredita una versión fija de EcoMug.
- La publicación del programa combinado debe conservar la GPL y su código fuente correspondiente. No se declara que cualquier módulo independiente del repositorio quede sujeto a GPL por compartir el mismo repositorio.

## JSON for Modern C++

- Archivo: `scripts/simulation/third_party/json/json.hpp`.
- Versión declarada: 3.11.3.
- Cabecera conservada: SPDX-FileCopyrightText: 2013-2023 Niels Lohmann; SPDX-License-Identifier: MIT.
- [Texto MIT del proyecto para v3.11.3](LICENSES/nlohmann-json-MIT.txt), obtenido de https://raw.githubusercontent.com/nlohmann/json/v3.11.3/LICENSE.MIT
- El texto oficial de esa versión conserva el intervalo 2013–2022; no se ha alterado para sustituir el aviso 2013–2023 de la cabecera.
- La cabecera incluye contribuciones y avisos adicionales, entre ellos Evan Nemerson, Florian Loitsch y fragmentos de Abseil bajo Apache-2.0. Se conservan íntegramente; se incluye también [Apache-2.0](LICENSES/Apache-2.0.txt).

## Geant4

Geant4 se localiza mediante `find_package(Geant4 REQUIRED)`; su licencia no se sustituye por la del código propio. La receta `scripts/simulation/docker/Dockerfile` descarga y compila Geant4 desde su repositorio oficial. La redistribución de Geant4, de una imagen de contenedor que lo incluya o de sus modificaciones requiere conservar los términos de la versión redistribuida y los reconocimientos correspondientes.

Reconocimiento: “This product includes software developed by Members of the Geant4 Collaboration (http://cern.ch/geant4).”

Texto oficial: https://geant4.web.cern.ch/download/license

El uso de Geant4 no supone respaldo de la colaboración a las conclusiones científicas del trabajo.

## Dependencias y material documental

Las dependencias instaladas mediante `requirements.txt` conservan sus respectivas licencias. La licencia del repositorio no concede derechos adicionales sobre ellas. Antes de distribuir un entorno, ejecutable o contenedor que las incluya se deben incorporar los avisos y fuentes que correspondan.

Las referencias bibliográficas no equivalen a permisos de reproducción de figuras o textos ajenos. Las marcas institucionales no se licencian como material propio. Para reutilizar un elemento externo debe acudirse a su titular o a la licencia aplicable.
