# =============================================================================
# Geant4 + EcoMug simulation environment
#
# Targets:
#   sim-batch : Geant4 multithreaded, no visualization. Production + CI.
#   sim-viz   : Geant4 with Qt6/OpenGL viewer for geometry inspection (WSLg).
#
# Build:  docker compose build sim-batch
# REPRODUCIBILITY RULE: after the first successful build, freeze the exact
# GEANT4_VERSION tag and ECOMUG_REF commit hash below and record them
# somewhere durable. Never leave them floating.
# Check latest stable: ./check_versions.sh
# =============================================================================

# TODO(freeze): pin after first successful build
ARG GEANT4_VERSION=v11.3.2

# ---------------------------------------------------------------- builder base
FROM ubuntu:24.04 AS builder-common
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      libexpat1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
ARG GEANT4_VERSION
RUN git clone --branch ${GEANT4_VERSION} --depth 1 \
      https://github.com/Geant4/geant4.git /opt/geant4-src

# ------------------------------------------------------------- batch (no viz)
FROM builder-common AS builder-batch
RUN cmake -S /opt/geant4-src -B /opt/geant4-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/geant4 \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_USE_QT=OFF \
      -DGEANT4_USE_OPENGL_X11=OFF \
    && cmake --build /opt/geant4-build -j"$(nproc)" \
    && cmake --install /opt/geant4-build \
    && rm -rf /opt/geant4-build

# ---------------------------------------------------------------- viz (Qt6/GL)
FROM builder-common AS builder-viz
RUN apt-get update && apt-get install -y --no-install-recommends \
      qt6-base-dev libgl1-mesa-dev libglu1-mesa-dev libxmu-dev libxt-dev \
    && rm -rf /var/lib/apt/lists/*
RUN cmake -S /opt/geant4-src -B /opt/geant4-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/geant4 \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_USE_QT=ON \
      -DGEANT4_USE_OPENGL_X11=ON \
    && cmake --build /opt/geant4-build -j"$(nproc)" \
    && cmake --install /opt/geant4-build \
    && rm -rf /opt/geant4-build

# =========================================================== runtime: sim-batch
FROM ubuntu:24.04 AS sim-batch
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential git ca-certificates \
      libexpat1 zlib1g python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder-batch /opt/geant4 /opt/geant4
ENV Geant4_DIR=/opt/geant4/lib/cmake/Geant4 \
    PATH=/opt/geant4/bin:$PATH \
    LD_LIBRARY_PATH=/opt/geant4/lib
WORKDIR /workspace
COPY geant4_app/ /workspace/geant4_app/
COPY geometry_catalog/ /workspace/geometry_catalog/
RUN . /opt/geant4/bin/geant4.sh \
    && cmake -S geant4_app -B /workspace/build \
    && cmake --build /workspace/build -j"$(nproc)"
CMD ["/workspace/build/muography_sim", \
     "--geometry", "/workspace/geometry_catalog/geometries/geo-train-01.json", \
     "--tables", "/workspace/geant4_app/tables", \
     "--out", "/workspace/data/output"]

# ============================================================= runtime: sim-viz
FROM ubuntu:24.04 AS sim-viz
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential git ca-certificates \
      libexpat1 zlib1g python3 \
      libqt6opengl6 libqt6widgets6 libgl1 libglu1-mesa mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder-viz /opt/geant4 /opt/geant4
ENV Geant4_DIR=/opt/geant4/lib/cmake/Geant4 \
    PATH=/opt/geant4/bin:$PATH \
    LD_LIBRARY_PATH=/opt/geant4/lib
WORKDIR /workspace
COPY geant4_app/ /workspace/geant4_app/
COPY geometry_catalog/ /workspace/geometry_catalog/
RUN . /opt/geant4/bin/geant4.sh \
    && cmake -S geant4_app -B /workspace/build \
    && cmake --build /workspace/build -j"$(nproc)"
# geant4_app is batch-only by design (no UI/vis driver); this stage still
# builds it so the Qt6/OpenGL runtime libraries are available if a UI build
# is added later.
CMD ["/workspace/build/muography_sim", \
     "--geometry", "/workspace/geometry_catalog/geometries/geo-train-01.json", \
     "--tables", "/workspace/geant4_app/tables", \
     "--out", "/workspace/data/output"]
