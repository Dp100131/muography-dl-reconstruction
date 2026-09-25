// Standalone EcoMug generator (no Geant4) — Stage 6 real-generator validation.
// Generates N muons on a sky surface and writes (p, theta, phi, charge) CSV
// consumed by analysis/validate_ecomug.py.
//
// Build (WSL Ubuntu, after build-essential + cmake are installed):
//   cd simulation/ecomug_standalone && make && ./ecomug_gen 10000000 > muons.csv

#include <cstdio>
#include <cstdlib>

#include "EcoMug.h"

int main(int argc, char** argv)
{
  const long n = (argc > 1) ? std::atol(argv[1]) : 1000000L;
  const unsigned seed = (argc > 2) ? static_cast<unsigned>(std::atol(argv[2]))
                                   : 20260805U;

  EcoMug gen;
  gen.SetUseSky();
  gen.SetSkySize({{10.0, 10.0}});
  gen.SetSkyCenterPosition({{0.0, 0.0, 0.0}});
  gen.SetSeed(seed);
  gen.SetMinimumMomentum(0.5);    // GeV/c
  gen.SetMaximumMomentum(10000.); // GeV/c

  std::printf("p_gev,theta_rad,phi_rad,charge\n");
  for (long i = 0; i < n; ++i) {
    gen.Generate();
    std::printf("%.6g,%.6g,%.6g,%d\n",
                gen.GetGenerationMomentum(),
                gen.GetGenerationTheta(),
                gen.GetGenerationPhi(),
                gen.GetCharge());
  }
  return 0;
}
