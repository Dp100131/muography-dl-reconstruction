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
//
// OE1 stage 1 (P2) modes, selected by new flags (see include/Oe1Cli.hh);
// with any of them the whole command line is parsed strictly (exit 2):
//   --primaries LIST ...   list mode: transports exactly the frozen list
//                          (N = rows), per-muon reseed, <stem>_muones.csv,
//                          meta key "oe1_etapa1"; exit 4 if a check fails
//   --write-primaries OUT  generation only (no run manager, no transport)
//   --check-primaries LIST validation only (no Geant4 objects)
// List mode requires --region-pair I,J (conduit, dense_body|cavity) whenever
// the geometry has two or more anomalies; --region-pair none waives it on
// purpose (exit 2 otherwise). --muon-log min keeps only the O(1) status
// bookkeeping of EventAction::OnPrimaryStep (see include/EventAction.hh).
// Exit codes of list mode, in the order in which main() can produce them:
//   2  the strict parser rejects the command line (unknown or repeated flag,
//      missing or malformed value, a flag not allowed in the mode)
//   1  the list, the ROI mask or the geometry cannot be read or does not match
//      its --*-sha256; the list disagrees with the geometry, view, case code
//      or replica
//   2  the --region-pair rule of the geometry (two or more anomalies and no
//      pair, an invalid pair)
//   1  the provenance hashes cannot be taken, or --out is missing, not
//      writable or already holds a result of this run
//   4  right after runManager->Initialize(), BEFORE BeamOn: the run manager is
//      not G4TaskRunManager, or --variant does not match the geometry that was
//      built (oe1::CheckBeforeTransport). No transport runs and no output file
//      is written.
//   -- BeamOn --
//   4  a run check failed at the end of the run (ids, histogram
//      reconstruction, worker threads or engines, aborted events, output not
//      written, and the two checks above once more: second line of defence)
//   0  everything holds
// Before BeamOn the process also prints one OE1-PRERUN-CHECK line (status=OK
// or FAIL) so the log shows that the checks ran.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <typeinfo>
#include <vector>

#include "CLHEP/Random/MixMaxRng.h"
#include "FTFP_BERT.hh"
#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "FluxTables.hh"
#include "GeometryConfig.hh"
#include "GeometryDiagnostics.hh"
#include "Oe1Cli.hh"
#include "Oe1Hash.hh"
#include "Oe1ListMode.hh"
#include "PrimaryGeneratorAction.hh"
#include "PrimaryList.hh"
#include "RoiMask.hh"
#include "RunAction.hh"
#include "RunData.hh"
#include "Sha256.hh"

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

// OE1 stage 1 (P2): demangled dynamic class name (e.g. G4TaskRunManager).
std::string DemangledClassName(const std::type_info& type)
{
  int status = 0;
  char* name = abi::__cxa_demangle(type.name(), nullptr, nullptr, &status);
  const std::string out = (status == 0 && name != nullptr) ? name : type.name();
  std::free(name);
  return out;
}

// OE1 stage 1 (P2), --write-primaries: generation only; returns before any
// run manager exists. A private MixMaxRng seeded with the key
// {s_src, case_code, 0, TAG_SRC} is installed as the CLHEP engine and the
// UNCHANGED legacy sampler (PrimaryGeneratorAction::GeneratePrimaries, 7
// uniforms per muon) runs once per G4Event(id); the primary is read back
// from the event, so the pilot source model is reused, not re-implemented.
int RunWritePrimaries(const oe1::CliOptions& opt, const GeometryConfig& cfg,
                      const FluxTables& tables)
{
  const std::string listPath = opt.writeOut;
  const std::string sidecarPath = oe1::SidecarPath(listPath);
  std::error_code ec;
  if (std::filesystem::exists(listPath, ec) || std::filesystem::exists(sidecarPath, ec)) {
    std::cerr << "refusing to overwrite " << listPath << " or its sidecar\n";
    return 1;
  }

  const long key[4] = {static_cast<long>(opt.seed), static_cast<long>(opt.caseCode), 0L,
                       static_cast<long>(oe1::kTagSource)};
  CLHEP::MixMaxRng engine;
  engine.setSeeds(key, 4);
  CLHEP::HepRandomEngine* previous = G4Random::getTheEngine();
  G4Random::setTheEngine(&engine);

  // The constructor of G4VUserPrimaryGeneratorAction (base of the legacy
  // sampler) aborts with G4Exception Run0061 unless the particle table is
  // flagged ready, which a run-manager kernel normally does. No run manager
  // exists in this mode, so set the flag directly; nothing else is needed to
  // sample (the muon definitions are created on demand).
  G4ParticleTable::GetParticleTable()->SetReadiness();

  std::vector<oe1::PrimaryRecord> recs;
  recs.reserve(static_cast<std::size_t>(opt.events));
  bool single = true;
  {
    PrimaryGeneratorAction generator(opt.view, cfg, &tables);
    for (long long id = 0; id < opt.events; ++id) {
      G4Event event(static_cast<G4int>(id));
      generator.GeneratePrimaries(&event);
      const G4PrimaryVertex* vertex = event.GetPrimaryVertex(0);
      const G4PrimaryParticle* primary = vertex != nullptr ? vertex->GetPrimary(0) : nullptr;
      if (event.GetNumberOfPrimaryVertex() != 1 || primary == nullptr ||
          vertex->GetNumberOfParticle() != 1) {
        single = false;
        break;
      }
      oe1::PrimaryRecord r;
      r.id = id;
      r.x_m = vertex->GetX0() / m;
      r.y_m = vertex->GetY0() / m;
      r.z_m = vertex->GetZ0() / m;
      const G4ThreeVector& dir = primary->GetMomentumDirection();
      r.dx = dir.x();
      r.dy = dir.y();
      r.dz = dir.z();
      r.ekin_gev = primary->GetKineticEnergy() / GeV;
      r.pdg = primary->GetPDGcode();
      r.charge_e = (r.pdg == 13) ? -1 : ((r.pdg == -13) ? 1 : 0);
      recs.push_back(r);
    }
  }
  // The sidecar documents 7 uniform draws per muon: prove it. A fresh engine
  // with the same key advanced 7 N times must be in the state the sampler left.
  const bool drawsOk = [&]() {
    CLHEP::MixMaxRng fresh;
    fresh.setSeeds(key, 4);
    for (long long i = 0; i < 7 * opt.events; ++i) (void)fresh.flat();
    return fresh.put() == engine.put();
  }();
  G4Random::setTheEngine(previous);
  if (!single) {
    std::cerr << "generation failed: an event did not hold exactly one primary\n";
    return 1;
  }
  if (!drawsOk) {
    std::cerr << "generation failed: the sampler did not consume exactly 7 uniforms per muon\n";
    return 1;
  }

  oe1::ListDefects defects;
  oe1::ValidatePrimaryRecords(recs, defects);
  if (!defects.Clean()) {
    std::cerr << "generated list failed validation\n";
    return 1;
  }
  std::string err;
  if (!oe1::WritePrimaryCsv(listPath, recs, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  // Round trip: the text must parse back to exactly the same values.
  std::vector<oe1::PrimaryRecord> back;
  oe1::ListDefects backDefects;
  std::string fileSha;
  const uint64_t canon = oe1::CanonicalFnv(recs);
  if (!oe1::ReadPrimaryFile(listPath, back, backDefects, fileSha)) {
    std::cerr << "cannot re-read " << listPath << "\n";
    return 1;
  }
  oe1::ValidatePrimaryRecords(back, backDefects);
  if (!backDefects.Clean() || oe1::CanonicalFnv(back) != canon) {
    std::cerr << "round-trip check failed for " << listPath << "\n";
    return 1;
  }

  oe1::ListProvenance prov;
  prov.geometryId = cfg.geometry_id;
  prov.view = opt.view;
  prov.baseRadiusM = cfg.base_radius_m;
  // Same expression as PrimaryGeneratorAction's fStartRadiusM.
  prov.startRadiusM = viewpar::kDetDistanceM + cfg.base_radius_m + 300.0;
  prov.caseCode = opt.caseCode;
  prov.replica = opt.replica;
  prov.sourceSeed = opt.seed;
  prov.sourceTag = oe1::kTagSource;
  prov.drawsPerPrimary = 7;
  prov.canonFnv = canon;
  prov.fileSha256 = fileSha;
  if (!oe1::Sha256File("/proc/self/exe", prov.binarySha256)) prov.binarySha256 = "unavailable";
  if (!oe1::WriteSidecar(sidecarPath, prov, recs.size(), err)) {
    std::cerr << err << "\n";
    return 1;
  }
  std::cout << "OE1-WRITE-PRIMARIES file="
            << std::filesystem::path(listPath).filename().string()
            << " geometry=" << cfg.geometry_id << " view=" << opt.view
            << " case_code=" << opt.caseCode << " replica=" << opt.replica
            << " source_seed=" << opt.seed << " n=" << recs.size()
            << " n_mu_plus=" << defects.nMuPlus << " canon_fnv1a64=" << oe1::Hex16(canon)
            << " file_sha256=" << fileSha << " status=OK" << std::endl;
  return 0;
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
  // --- OE1 stage 1 (P2): inert unless an OE1 flag is present; then the
  //     whole command line is parsed strictly (errors exit 2).
  oe1::CliOptions oe1Opt;
  if (oe1::AnyStage1Flag(argc, argv)) {
    std::string oe1Err;
    if (!oe1::ParseStage1(argc, argv, oe1Opt, oe1Err)) {
      std::cerr << "error: " << oe1Err << "\n";
      oe1::PrintStage1Usage(std::cerr, argv[0]);
      return 2;
    }
    if (oe1Opt.mode == oe1::CliMode::kCheck) {
      return oe1::RunCheckPrimaries(oe1Opt.checkFile, std::cout, std::cerr);
    }
  }
  const bool listMode = oe1Opt.mode == oe1::CliMode::kList;
  oe1::PrimaryList primaryList;
  oe1::RoiMask roiMask;
  if (listMode) {
    std::string oe1Err;
    if (!primaryList.Load(oe1Opt.primaries, oe1Opt.primariesSha256, oe1Err) ||
        !roiMask.Load(oe1Opt.roiMask, oe1Opt.roiMaskSha256, oe1Err)) {
      std::cerr << "input error: " << oe1Err << "\n";
      return 1;
    }
    events = static_cast<long long>(primaryList.Size());  // N = list rows
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

  // OE1 stage 1 (P2): generation mode ends here, before any seeding of the
  // global engine or run manager; list mode checks the list against the
  // geometry and builds the region.
  if (oe1Opt.mode == oe1::CliMode::kWrite) return RunWritePrimaries(oe1Opt, cfg, tables);
  oe1::ListModeConfig listConfig;
  if (listMode) {
    std::string oe1Err;
    const int rc =
        oe1::ConfigureListMode(oe1Opt, cfg, primaryList, roiMask, listConfig, oe1Err);
    if (rc != 0) {
      std::cerr << (rc == 2 ? "error: " : "input error: ") << oe1Err << "\n";
      return rc;
    }
    // SHA-256 of this executable, the geometry JSON and the three table files:
    // recorded in the meta so the result can be tied to its exact inputs.
    if (!oe1::RecordProvenance(oe1Opt, listConfig, oe1Err)) {
      std::cerr << "input error: " << oe1Err << "\n";
      return 1;
    }
    // Never start a transport that would overwrite a result or cannot write its own.
    if (!oe1::PreflightOutputs(outDir, cfg.geometry_id + "_view" + std::to_string(view), oe1Err)) {
      std::cerr << "input error: " << oe1Err << "\n";
      return 1;
    }
    RunData::Instance().ReserveMuons(primaryList.Size());
  }

  CLHEP::HepRandom::setTheSeed(static_cast<long>(seed % 2147483647ULL));

  auto* runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetNumberOfThreads(threads);
  if (listMode) {
    listConfig.runManagerClass = DemangledClassName(typeid(*runManager));
    listConfig.masterEngine = G4Random::getTheEngine()->name();
    std::cout << "OE1-LIST-MODE list=" << listConfig.listFile
              << " n=" << primaryList.Size() << " sha256=" << primaryList.FileSha256()
              << " canon_fnv1a64=" << oe1::Hex16(primaryList.CanonFnv())
              << " mask=" << listConfig.maskFile << " mask_ones=" << roiMask.Ones()
              << " geometry=" << cfg.geometry_id << " view=" << view
              << " variant=" << listConfig.variant << " case_code=" << listConfig.caseCode
              << " replica=" << listConfig.replica
              << " transport_seed=" << listConfig.transportSeed << " master_seed=" << seed
              << " master_seed_applied=" << listConfig.masterSeedApplied
              << " run_manager=" << listConfig.runManagerClass
              << " muon_log="
              << (listConfig.logLevel == oe1::MuonLogLevel::kFull ? "full" : "min")
              << " region="
              << (listConfig.regionApplicable
                      ? std::to_string(listConfig.regionFirst) + "," +
                            std::to_string(listConfig.regionSecond)
                      : std::string("none"))
              << " binary_sha256=" << listConfig.binarySha256
              << " geometry_sha256=" << listConfig.geometrySha256 << std::endl;
  }

  auto* detector = new DetectorConstruction(cfg, view);
  runManager->SetUserInitialization(detector);

  auto* physics = new FTFP_BERT(0);  // verbosity 0
  physics->SetDefaultCutValue(1.0 * m);  // coarse cuts: primary-only counting
  runManager->SetUserInitialization(physics);

  runManager->SetUserInitialization(new ActionInitialization(
      detector, cfg, view, seed, outDir, threads, &tables,
      days > 0 ? days : -1.0, listMode ? &listConfig : nullptr));

  runManager->Initialize();

  // OE1 stage 1 (P2, plan S3): the run manager class and --variant against the
  // geometry just built need no transport, so a mismatch ends the process here,
  // before BeamOn, with no output file. RunAction repeats both checks at the end
  // of the run as a second line of defence.
  if (listMode) {
    const oe1::PreRunReport pre =
        oe1::CheckBeforeTransport(listConfig, oe1::BuiltGeometrySignature());
    const char* variantText = pre.variantCheck == oe1::VariantCheck::kOk
                                  ? "ok"
                                  : (pre.variantCheck == oe1::VariantCheck::kMismatch ? "mismatch"
                                                                                      : "not_checked");
    std::string codes;
    for (const std::string& e : pre.errors) codes += (codes.empty() ? "" : ",") + e;
    std::cout << "OE1-PRERUN-CHECK run_manager=" << listConfig.runManagerClass
              << " variant=" << listConfig.variant << " variant_check=" << variantText
              << " errors=" << (codes.empty() ? "none" : codes)
              << " status=" << (pre.Ok() ? "OK" : "FAIL") << std::endl;
    if (!pre.Ok()) {
      for (const std::string& msg : pre.messages) std::cerr << "list mode: " << msg << "\n";
      std::cerr << "list mode: a check failed before BeamOn: no transport was run and no output "
                   "was written (exit "
                << oe1::kExitRunCheckFailed << ")\n";
      delete runManager;
      return oe1::kExitRunCheckFailed;
    }
  }

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
  if (listMode) {
    if (!RunData::Instance().Finalized()) {
      std::cerr << "list mode: the master EndOfRunAction did not finalise the run "
                   "(no per-muon log was written)\n";
      return oe1::kExitRunCheckFailed;
    }
    if (RunData::Instance().HasErrors()) return oe1::kExitRunCheckFailed;  // OE1 checks failed
  }
  return 0;
}
