// OE1 stage 1 (P2): legacy-inertness probe. It uses ONLY the pre-P2 API of the
// legacy classes, so the same source compiles against the tree at 9520170 (no
// P2) and against the P2 tree; tests/g4/legacy_inertness.sh builds both and
// compares their outputs. No run manager, no BeamOn, no transport.
//
//   legacy_inertness pga TABLES_DIR GEOMETRY_JSON N SEED
//       Runs PrimaryGeneratorAction::GeneratePrimaries (legacy sampler, no list)
//       N times on a private MixMaxRng seeded with SEED and prints one line
//       with the FNV-1a-64 of every primary (position, direction, energy, pdg).
//       Equal lines mean the legacy sampler draws the same numbers in the same
//       order (including the unspecified evaluation order of the two
//       G4UniformRand() operands of the impact point).
//   legacy_inertness run OUT_DIR
//       Feeds RunData with a fixed synthetic set of hits and finishes the run
//       with the legacy RunAction on the master; leaves
//       <OUT_DIR>/geo-train-06_view0_hist.csv and .meta.json.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "CLHEP/Random/MixMaxRng.h"
#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "FluxTables.hh"
#include "GeometryConfig.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "RunData.hh"

namespace {

void Mix(uint64_t& h, const void* data, std::size_t n)
{
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
}

void MixD(uint64_t& h, double v) { Mix(h, &v, sizeof v); }

int RunPga(int argc, char** argv)
{
  if (argc != 6) return 2;
  const std::string tablesDir = argv[2];
  const long n = std::atol(argv[4]);
  const long seed = std::atol(argv[5]);
  const GeometryConfig cfg = GeometryConfig::FromJsonFile(argv[3]);
  FluxTables tables(tablesDir);

  const long key[4] = {seed, 6L, 0L, 0x4F455331L};
  CLHEP::MixMaxRng engine;
  engine.setSeeds(key, 4);
  G4Random::setTheEngine(&engine);
  G4ParticleTable::GetParticleTable()->SetReadiness();

  uint64_t h = 14695981039346656037ULL;
  {
    PrimaryGeneratorAction pga(0, cfg, &tables);
    for (long id = 0; id < n; ++id) {
      G4Event event(static_cast<G4int>(id));
      pga.GeneratePrimaries(&event);
      const G4PrimaryVertex* v = event.GetPrimaryVertex(0);
      const G4PrimaryParticle* p = v->GetPrimary(0);
      MixD(h, v->GetX0());
      MixD(h, v->GetY0());
      MixD(h, v->GetZ0());
      MixD(h, p->GetMomentumDirection().x());
      MixD(h, p->GetMomentumDirection().y());
      MixD(h, p->GetMomentumDirection().z());
      MixD(h, p->GetKineticEnergy());
      const int pdg = p->GetPDGcode();
      Mix(h, &pdg, sizeof pdg);
    }
  }
  std::printf("legacy sampler: n=%ld seed=%ld fnv1a64=%016llx\n", n, seed,
              static_cast<unsigned long long>(h));
  return 0;
}

int RunLegacyRunAction(int argc, char** argv)
{
  if (argc != 3) return 2;
  // G4UserRunAction's constructor aborts (Run0041) unless the particle table
  // is flagged ready, which a run-manager kernel normally does.
  G4ParticleTable::GetParticleTable()->SetReadiness();
  GeometryConfig cfg;
  cfg.geometry_id = "geo-train-06";
  const std::string outDir = argv[2];

  std::mt19937_64 gen(12345);
  const int nGenerated = 6000, nHits = 5000;
  for (int i = 0; i < nGenerated; ++i) RunData::Instance().RecordGenerated();
  for (int i = 0; i < nHits; ++i) {
    const int az = static_cast<int>(gen() % 256), el = static_cast<int>(gen() % 256);
    const double ekin = 0.5 + static_cast<double>(gen() % 100000) / 977.0;
    RunData::Instance().RecordHit(az, el, ekin);
  }
  auto* run = new G4Run();  // its destructor needs a run manager: never deleted
  G4Event event(0);
  for (int i = 0; i < nGenerated; ++i) run->RecordEvent(&event);

  RunAction ra(cfg, 0, 2026092900ULL, outDir, 8, -1.0);  // the legacy constructor
  ra.BeginOfRunAction(run);
  ra.EndOfRunAction(run);
  std::fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc >= 2 && std::strcmp(argv[1], "pga") == 0) return RunPga(argc, argv);
  if (argc >= 2 && std::strcmp(argv[1], "run") == 0) return RunLegacyRunAction(argc, argv);
  std::fprintf(stderr, "usage: legacy_inertness pga TABLES GEOMETRY N SEED | run OUT_DIR\n");
  return 2;
}
