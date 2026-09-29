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
//
// Diagnostic mode (no BeamOn; runs on the master thread after
// runManager->Initialize() and exits): --check-geometry, --dump-geometry,
// --probe X,Y,Z, --slice-y Y:XMIN:XMAX:ZMIN:ZMAX:STEP,
// --slice-z Z:XMIN:XMAX:YMIN:YMAX:STEP, --volume-stat N. Any one of them
// switches the run into diagnostic mode.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "FTFP_BERT.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "FluxTables.hh"
#include "GeometryConfig.hh"
#include "GeometryDiagnostics.hh"

namespace {
void usage(const char* prog)
{
  std::cerr << "usage: " << prog
            << " --geometry FILE.json --view 0-5 --tables DIR --seed S"
               " --out DIR [--days D | --events N] [--threads T]\n";
}
// Same as usage(), plus the diagnostic-flag summary. Only shown for a
// diagnostic-context error, so the plain validation-failure message stays
// byte-identical to the base (no-diagnostics) build when no diagnostic
// flag was given at all.
void usageDiag(const char* prog)
{
  usage(prog);
  std::cerr << "   diagnostics (no BeamOn): [--check-geometry] [--dump-geometry]"
               " [--probe X,Y,Z]... [--slice-y Y:XMIN:XMAX:ZMIN:ZMAX:STEP]..."
               " [--slice-z Z:XMIN:XMAX:YMIN:YMAX:STEP]... [--volume-stat N]\n";
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

  // consumedAsValue[i] marks argv[i] as already claimed by a normal flag's
  // value (e.g. "--out foo"), so the diagnostic-flag pass below can never
  // reinterpret that same token as a diagnostic flag.
  std::vector<bool> consumedAsValue(static_cast<std::size_t>(argc), false);
  for (int i = 1; i < argc - 1; ++i) {
    const std::string a = argv[i];
    if (a == "--geometry") { geomFile = argv[++i]; consumedAsValue[i] = true; }
    else if (a == "--view") { view = std::atoi(argv[++i]); consumedAsValue[i] = true; }
    else if (a == "--events") { events = std::atoll(argv[++i]); consumedAsValue[i] = true; }
    else if (a == "--days") { days = std::atof(argv[++i]); consumedAsValue[i] = true; }
    else if (a == "--seed") {
      seed = std::strtoull(argv[++i], nullptr, 10);
      haveSeed = true;
      consumedAsValue[i] = true;
    }
    else if (a == "--out") { outDir = argv[++i]; consumedAsValue[i] = true; }
    else if (a == "--tables") { tablesDir = argv[++i]; consumedAsValue[i] = true; }
    else if (a == "--threads") { threads = std::atoi(argv[++i]); consumedAsValue[i] = true; }
  }

  // Separate pass for diagnostic flags: boolean flags (--check-geometry,
  // --dump-geometry) must work in any position, including last, so they
  // cannot share the argc-1-bounded loop above. Tokens already consumed as
  // a normal flag's value are skipped so a malformed normal flag (missing
  // its value) can never be silently reinterpreted as a diagnostic flag.
  bool diag = false;
  bool checkGeometry = false;
  bool dumpGeometry = false;
  std::vector<G4ThreeVector> probes;
  std::vector<geomdiag::SliceSpec> slicesY, slicesZ;
  long volumeStatN = 10000000;
  bool diagParseError = false;
  for (int i = 1; i < argc; ++i) {
    if (consumedAsValue[i]) continue;
    const std::string a = argv[i];
    if (a == "--check-geometry") { checkGeometry = true; diag = true; }
    else if (a == "--dump-geometry") { dumpGeometry = true; diag = true; }
    else if (a == "--probe") {
      diag = true;
      G4ThreeVector p;
      if (i + 1 >= argc || !geomdiag::ParseProbe(argv[++i], p)) { diagParseError = true; break; }
      probes.push_back(p);
    }
    else if (a == "--slice-y") {
      diag = true;
      geomdiag::SliceSpec spec;
      if (i + 1 >= argc || !geomdiag::ParseSlice(argv[++i], spec)) { diagParseError = true; break; }
      slicesY.push_back(spec);
    }
    else if (a == "--slice-z") {
      diag = true;
      geomdiag::SliceSpec spec;
      if (i + 1 >= argc || !geomdiag::ParseSlice(argv[++i], spec)) { diagParseError = true; break; }
      slicesZ.push_back(spec);
    }
    else if (a == "--volume-stat") {
      diag = true;
      if (i + 1 >= argc) { diagParseError = true; break; }
      volumeStatN = std::atol(argv[++i]);
      // EstimateCubicVolume takes a G4int and the SE divides by N.
      if (volumeStatN < 1 || volumeStatN > 2000000000L) { diagParseError = true; break; }
    }
  }
  if (diagParseError) {
    usageDiag(argv[0]);
    return 2;
  }

  if (geomFile.empty() || view < 0 || view > 5 || tablesDir.empty() ||
      !haveSeed || (!diag && (events <= 0 && days <= 0))) {
    if (diag) usageDiag(argv[0]); else usage(argv[0]);
    return 2;
  }

  FluxTables tables(tablesDir);
  if (!diag && events <= 0) {
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

  if (diag) {
    int exitCode = 0;
    if (checkGeometry) {
      exitCode = geomdiag::RunCheckGeometry(
          cfg.geometry_id, view, static_cast<long>(seed % 2147483647ULL),
          100000, 0.0, 100);
    }
    if (dumpGeometry) geomdiag::RunDumpGeometry(volumeStatN);
    for (const auto& p : probes) geomdiag::RunProbe(p);
    for (const auto& s : slicesY) geomdiag::RunSliceY(cfg.geometry_id, view, outDir, s);
    for (const auto& s : slicesZ) geomdiag::RunSliceZ(cfg.geometry_id, view, outDir, s);
    delete runManager;
    return exitCode;
  }

  runManager->BeamOn(static_cast<G4int>(events));

  delete runManager;
  return 0;
}
