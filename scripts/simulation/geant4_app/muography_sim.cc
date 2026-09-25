// Production muography transmission simulation (table-driven generator).
//
// Usage:
//   muography_sim --geometry <catalog.json> --view <0-5> --tables DIR \
//                 --seed S --out DIR [--days D | --events N] [--threads T]
//
// --days D  : exposure-calibrated run — N = Poisson(total_open_sky_30d * D/30)
// --events N: fixed event count (smoke tests, benchmarks)
//
// Output: <out>/<geometry_id>_view<k>_hist.csv + .meta.json
// (tools/csv_to_npz.py converts to the NPZ schema shared with the
// surrogate pipeline.)

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "FTFP_BERT.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "FluxTables.hh"
#include "GeometryConfig.hh"

namespace {
void usage(const char* prog)
{
  std::cerr << "usage: " << prog
            << " --geometry FILE.json --view 0-5 --tables DIR --seed S"
               " --out DIR [--days D | --events N] [--threads T]\n";
}
}  // namespace

int main(int argc, char** argv)
{
  std::string geomFile, outDir = ".", tablesDir;
  int view = -1, threads = 1;
  long long events = -1;
  double days = -1.0;
  uint64_t seed = 0;
  bool haveSeed = false;

  for (int i = 1; i < argc - 1; ++i) {
    const std::string a = argv[i];
    if (a == "--geometry") geomFile = argv[++i];
    else if (a == "--view") view = std::atoi(argv[++i]);
    else if (a == "--events") events = std::atoll(argv[++i]);
    else if (a == "--days") days = std::atof(argv[++i]);
    else if (a == "--seed") { seed = std::strtoull(argv[++i], nullptr, 10); haveSeed = true; }
    else if (a == "--out") outDir = argv[++i];
    else if (a == "--tables") tablesDir = argv[++i];
    else if (a == "--threads") threads = std::atoi(argv[++i]);
  }
  if (geomFile.empty() || view < 0 || view > 5 || tablesDir.empty() ||
      !haveSeed || (events <= 0 && days <= 0)) {
    usage(argv[0]);
    return 2;
  }

  FluxTables tables(tablesDir);
  if (events <= 0) {
    // Exposure-calibrated event count: Poisson around the protocol rate.
    std::mt19937_64 rng(seed);
    const double mean = tables.TotalCounts30d() * days / 30.0;
    events = std::poisson_distribution<long long>(mean)(rng);
    std::cout << "exposure " << days << " d -> events=" << events
              << " (mean " << mean << ")\n";
  }

  GeometryConfig cfg;
  try {
    cfg = GeometryConfig::FromJsonFile(geomFile);
  }
  catch (const std::exception& e) {
    std::cerr << "geometry error: " << e.what() << "\n";
    return 1;
  }

  CLHEP::HepRandom::setTheSeed(static_cast<long>(seed % 2147483647ULL));

  auto* runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetNumberOfThreads(threads);

  auto* detector = new DetectorConstruction(cfg, view);
  runManager->SetUserInitialization(detector);

  auto* physics = new FTFP_BERT(0);  // verbosity 0
  physics->SetDefaultCutValue(1.0 * m);  // coarse cuts: primary-only counting
  runManager->SetUserInitialization(physics);

  runManager->SetUserInitialization(new ActionInitialization(
      detector, cfg, view, seed, outDir, threads, &tables,
      days > 0 ? days : -1.0));

  runManager->Initialize();
  runManager->BeamOn(static_cast<G4int>(events));

  delete runManager;
  return 0;
}
