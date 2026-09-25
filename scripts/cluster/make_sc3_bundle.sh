#!/usr/bin/env bash
# Build the SC3 deployment bundle: ONLY the simulation payload, never the
# thesis documents, research logs, meeting minutes or datasets.
# Output: sc3-deploy.tar.gz (extracts into muography-sim/)
set -euo pipefail

cd "$(dirname "$0")/../.."
COMMIT=$(git rev-parse HEAD)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git archive --format=tar HEAD \
    scripts/simulation/geant4_app \
    scripts/simulation/geometry_catalog/geometries \
    scripts/simulation/third_party \
    scripts/cluster \
  | tar -x -C "$TMP"

printf 'commit=%s\nbundled_at=%s\n' "$COMMIT" "$STAMP" > "$TMP/VERSION"
mkdir -p "$TMP/logs" "$TMP/results"

tar -czf sc3-deploy.tar.gz -C "$TMP" .
sha256sum sc3-deploy.tar.gz
echo "deploy:  scp sc3-deploy.tar.gz sc3-guane:"
echo "extract: ssh sc3-guane 'mkdir -p muography-sim && tar -xzf sc3-deploy.tar.gz -C muography-sim && cat muography-sim/VERSION'"
