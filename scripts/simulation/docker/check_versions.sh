#!/usr/bin/env bash
# Print the latest stable Geant4 release tag and the current EcoMug main commit.
# Use these to FREEZE versions in Dockerfile and the geant4_app CMakeLists,
# then record them somewhere durable.
set -euo pipefail

echo "== Geant4 latest release =="
curl -fsSL https://api.github.com/repos/Geant4/geant4/releases/latest \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['tag_name'], '-', r['name'])"

echo
echo "== EcoMug main HEAD commit =="
git ls-remote https://github.com/dr4kan/EcoMug.git HEAD
