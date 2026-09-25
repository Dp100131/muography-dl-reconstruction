# SC3 environment for muography_sim builds and runs.
# Source from every job script and interactive session:  source scripts/cluster/env.sh
# gnu14 FIRST: the geant4/11.4.2 libraries are built against gcc 14's
# libstdc++ (GLIBCXX_3.4.32); linking with the system gcc 11 fails.
module purge
module load gnu14/14.2.0 geant4/11.4.2

export MUOGRAPHY_SIM_HOME="${MUOGRAPHY_SIM_HOME:-$HOME/muography-sim}"
# Scratch lives under the RESEARCH GROUP directory (/scratch/<GROUP>/...),
# which only the admin can create. Until /scratch/SIMON exists, fall back to
# a home-based dir — acceptable for smoke tests (tiny I/O), NOT for the
# campaign (request the group dir before launching it).
if [ -d "/scratch/$(id -gn)" ]; then
  export MUOGRAPHY_SCRATCH="${MUOGRAPHY_SCRATCH:-/scratch/$(id -gn)/$USER/muography}"
else
  export MUOGRAPHY_SCRATCH="${MUOGRAPHY_SCRATCH:-$HOME/muography-scratch}"
fi
