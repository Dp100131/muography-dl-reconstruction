// OE1 stage 1 (P2): synthetic harness of the Geant4-linked list-mode code.
// It feeds hand-made G4Event / G4Step objects to EventAction and finishes the
// run with RunAction on the master, exactly as a list-mode run would, but
// WITHOUT a run manager: no BeamOn, no geometry, no physics, no transport.
// Purpose: exercise the code paths that a real transport run would exercise
// first (reseed, primary hash, per-step crossing bookkeeping, final status,
// record push, master finalisation, reconstruction checks, output files,
// meta key, error flags) before any transport is allowed.
//
// Usage: test_list_mode_g4 SCENARIO OUTDIR
//   ok-full          all hooks on; every output and value is verified
//   ok-min           --muon-log min; same tracks, reduced columns
//   ok-noregion      --region-pair none (the geo-train-01 way): no crossing fields
//   aborted          one event is aborted (must be reported as an error)
//   hist-mismatch    one detected muon is not recorded in the histogram
//   missing-event    one event never finishes
//   dup-id           one event is finished twice
//   two-workers      records from two worker threads
//   bad-engine       a worker engine that is not MixMaxRng
//   bad-outdir       the output directory does not exist
//   bad-variant      --variant A on a geometry whose anomaly_0 is a G4SubtractionSolid
//   bad-run-manager  a run manager that is not G4TaskRunManager
//   bad-both         both of the above at once
// Plan S3: the two checks of the last three scenarios need no transport, so main()
// evaluates them right after runManager->Initialize() and exits 4 BEFORE BeamOn. The
// harness has no run manager, so it evaluates the very same functions
// (oe1::BuiltGeometrySignature + oe1::CheckBeforeTransport) at the point where main()
// would and asserts what main() would do: every scenario but those three must pass them,
// those three must fail with exactly the expected codes. It then still finishes the run
// with RunAction to prove the second line of defence (the same codes in the meta and in
// the error flags). The exit itself (code 4, no output written) is proven on the real
// binary by tests/g4/early_exit.sh.
// OUTDIR must exist and be empty; the scenario writes only below it.
// Exit status 0 = every check of the scenario passed.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CLHEP/Random/MixMaxRng.h"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4NavigationHistory.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleTable.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SubtractionSolid.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

#include "EventAction.hh"
#include "GeometryConfig.hh"
#include "MuonRecord.hh"
#include "Oe1Cli.hh"
#include "Oe1Hash.hh"
#include "Oe1ListMode.hh"
#include "PrimaryGeneratorAction.hh"
#include "PrimaryList.hh"
#include "RegionCrossing.hh"
#include "RoiMask.hh"
#include "RunAction.hh"
#include "RunData.hh"
#include "Sha256.hh"
#include "check.hh"
#include "json.hpp"
#include "list_support.hh"

namespace fs = std::filesystem;
using oe1test::g_checks;
using oe1test::g_fail;

namespace {

constexpr int kN = 19;  // number of synthetic muons (event ids 0..18)

// Process that only carries a name (GetProcessDefinedStep()).
class NamedProcess : public G4VProcess
{
 public:
  explicit NamedProcess(const G4String& name) : G4VProcess(name, fElectromagnetic) {}
  G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override { return nullptr; }
  G4VParticleChange* AlongStepDoIt(const G4Track&, const G4Step&) override { return nullptr; }
  G4VParticleChange* AtRestDoIt(const G4Track&, const G4Step&) override { return nullptr; }
  G4double AlongStepGetPhysicalInteractionLength(const G4Track&, G4double, G4double, G4double&,
                                                 G4GPILSelection*) override
  {
    return 0.0;
  }
  G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition*) override
  {
    return 0.0;
  }
  G4double PostStepGetPhysicalInteractionLength(const G4Track&, G4double, G4ForceCondition*) override
  {
    return 0.0;
  }
};

struct World {
  G4Material* rock = nullptr;
  G4Material* a0 = nullptr;
  G4Material* a1 = nullptr;
  G4Material* air = nullptr;
  G4VPhysicalVolume* pvWorld = nullptr;
  G4VPhysicalVolume* pvConduit = nullptr;
  G4VPhysicalVolume* pvBody = nullptr;
  G4VPhysicalVolume* pvDetector = nullptr;
  std::unique_ptr<NamedProcess> transport, msc, decay, capture, ioni;
};

World MakeWorld()
{
  World w;
  w.rock = new G4Material("standard_rock", 14., 28.0855 * g / mole, 2.65 * g / cm3);
  w.a0 = new G4Material("anomaly_0_mat", 14., 28.0855 * g / mole, 1.832 * g / cm3);
  w.a1 = new G4Material("anomaly_1_mat", 14., 28.0855 * g / mole, 3.285 * g / cm3);
  w.air = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  auto* box = new G4Box("box", 1.0 * m, 1.0 * m, 1.0 * m);
  auto vol = [&](const char* name, G4Material* mat) -> G4VPhysicalVolume* {
    auto* lv = new G4LogicalVolume(box, mat, std::string(name) + "_log");
    return new G4PVPlacement(nullptr, G4ThreeVector(), lv, name, nullptr, false, 0, false);
  };
  w.pvWorld = vol("world", w.air);
  // anomaly_0 as variant B builds it: a Boolean solid (the conduit minus the later
  // anomaly), so its solid type is G4SubtractionSolid (a G4Tubs in variant A). The
  // geometry signature the run records must agree with --variant.
  auto* cut = new G4Box("cut", 0.5 * m, 0.5 * m, 0.5 * m);
  auto* conduit = new G4SubtractionSolid("anomaly_0_solid", box, cut);
  auto* conduitLv = new G4LogicalVolume(conduit, w.a0, "anomaly_0_log");
  w.pvConduit = new G4PVPlacement(nullptr, G4ThreeVector(), conduitLv, "anomaly_0", nullptr, false,
                                  0, false);
  w.pvBody = vol("anomaly_1", w.a1);
  w.pvDetector = vol("detector", w.air);
  w.transport = std::make_unique<NamedProcess>("Transportation");
  w.msc = std::make_unique<NamedProcess>("msc");
  w.decay = std::make_unique<NamedProcess>("Decay");
  w.capture = std::make_unique<NamedProcess>("muMinusCaptureAtRest");
  w.ioni = std::make_unique<NamedProcess>("muIoni");
  return w;
}

G4TouchableHandle Touchable(G4VPhysicalVolume* pv)
{
  G4NavigationHistory history;
  if (pv != nullptr) history.SetFirstEntry(pv);
  return G4TouchableHandle(new G4TouchableHistory(history));
}

struct StepSpec {
  G4ThreeVector pre, post;  // metres
  double ePre = 10.0, ePost = 10.0;  // GeV
  G4Material* mat = nullptr;
  const G4VProcess* process = nullptr;
  G4StepStatus status = fPostStepDoItProc;
  G4VPhysicalVolume* postVolume = nullptr;
  double trueLengthFactor = 1.0;  // G4Step length / chord length
  G4ThreeVector dirPre;           // zero: use the chord direction
};

void Feed(EventAction& ea, const StepSpec& spec)
{
  G4Step step;
  G4StepPoint* pre = step.GetPreStepPoint();
  G4StepPoint* post = step.GetPostStepPoint();
  const G4ThreeVector p0 = spec.pre * m, p1 = spec.post * m;
  pre->SetPosition(p0);
  pre->SetKineticEnergy(spec.ePre * GeV);
  pre->SetMaterial(spec.mat);
  pre->SetMomentumDirection(spec.dirPre.mag2() > 0.0 ? spec.dirPre.unit() : (p1 - p0).unit());
  pre->SetTouchableHandle(Touchable(spec.postVolume));
  post->SetPosition(p1);
  post->SetKineticEnergy(spec.ePost * GeV);
  post->SetStepStatus(spec.status);
  post->SetProcessDefinedStep(spec.process);
  post->SetTouchableHandle(Touchable(spec.postVolume));
  step.SetStepLength((p1 - p0).mag() * spec.trueLengthFactor);
  ea.OnPrimaryStep(&step);
}

// ------------------------------------------------------------------ CSV
struct Csv {
  std::vector<std::string> header;
  std::map<long, std::vector<std::string>> rows;
  std::size_t nRows = 0;
  bool sortedById = true;
  std::string Get(long id, const std::string& col) const
  {
    const auto it = std::find(header.begin(), header.end(), col);
    if (it == header.end()) return "<no such column " + col + ">";
    const auto r = rows.find(id);
    if (r == rows.end()) return "<no such row>";
    return r->second[static_cast<std::size_t>(it - header.begin())];
  }
  double Num(long id, const std::string& col) const { return std::strtod(Get(id, col).c_str(), nullptr); }
};

std::vector<std::string> SplitCsv(const std::string& line)
{
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      out.push_back(line.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

bool ReadCsv(const fs::path& path, Csv& csv)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  csv.header = SplitCsv(line);
  long previous = -1;
  while (std::getline(in, line)) {
    auto f = SplitCsv(line);
    if (f.size() != csv.header.size()) return false;
    const long id = std::strtol(f[6].c_str(), nullptr, 10);
    if (id <= previous) csv.sortedById = false;
    previous = id;
    csv.rows[id] = std::move(f);
    ++csv.nRows;
  }
  return true;
}

// ------------------------------------------------------------------ scenario
struct Expect {
  bool errorsExpected = false;
  std::vector<std::string> errorCodes;  // must all appear
};

// Vertical line through the centroid: heights where it enters and leaves Z.
struct ZVertical {
  double zin, zout;
};

ZVertical VerticalZ(double x, double y)
{
  const double ex = 1.0 - std::pow((x - 32.2) / 72.9, 2);
  const double half = 64.6 * std::sqrt(ex - std::pow((y + 74.1) / 61.6, 2));
  return {137.8 - half, 137.8 + half};
}

std::string MaskText()
{
  std::string s;
  for (int r = 0; r < 256; ++r) {
    for (int c = 0; c < 256; ++c) s.push_back((r >= 17 && r <= 50 && c >= 131 && c <= 138) ? '1' : '0');
    s.push_back('\n');
  }
  return s;
}

int Run(const std::string& scenario, const fs::path& outDir)
{
  const bool min = scenario == "ok-min";
  const bool noRegion = scenario == "ok-noregion";  // --region-pair none
  const bool zOn = !min && !noRegion;               // crossing fields are logged
  G4ParticleTable::GetParticleTable()->SetReadiness();

  // ---- inputs: list + sidecar + mask, loaded exactly as main() does
  const fs::path inDir = outDir / "inputs";
  fs::create_directories(inDir);
  const auto recs = oe1test::MakeRecords(kN, 77);
  const std::string listPath = oe1test::WriteListAndSidecar(inDir, "geo-train-06_v0_r1_s2026092901.primarios.csv", recs);
  oe1::PrimaryList list;
  std::string err;
  CHECK(list.Load(listPath, "", err));
  const std::string maskText = MaskText();
  oe1test::WriteAll(inDir / "mask.txt", maskText);
  oe1::RoiMask mask;
  CHECK(mask.Load((inDir / "mask.txt").string(), oe1::Sha256Hex(maskText), err));

  GeometryConfig cfg;
  cfg.geometry_id = "geo-train-06";
  cfg.base_radius_m = 509.7;
  cfg.height_m = 416.3;
  AnomalyConfig conduit;
  conduit.kind = "conduit";
  conduit.density_g_cm3 = 1.832;
  conduit.cx = 20.3; conduit.cy = -24.5; conduit.radius = 21.0; conduit.z0 = 0.0; conduit.z1 = 340.0;
  AnomalyConfig body;
  body.kind = "dense_body";
  body.density_g_cm3 = 3.285;
  body.cx = 32.2; body.cy = -74.1; body.cz = 137.8; body.ax = 72.9; body.ay = 61.6; body.az = 64.6;
  cfg.anomalies = {conduit, body};

  // Fixed inputs whose SHA-256 the meta records (plan S2): the geometry file and
  // the three generator tables.
  const std::string geometryText = "{\"geometry_id\": \"geo-train-06\", \"note\": \"harness\"}\n";
  oe1test::WriteAll(inDir / "geo-train-06.json", geometryText);
  fs::create_directories(inDir / "tables");
  const char* tableNames[3] = {"row_weights_30d.txt", "p_cdf.txt", "meta.json"};
  const std::string tableTexts[3] = {"1 1 1\n", "0.5 0.5\n", "{\"harness\": true}\n"};
  for (int t = 0; t < 3; ++t) oe1test::WriteAll(inDir / "tables" / tableNames[t], tableTexts[t]);

  oe1::CliOptions opt;
  opt.mode = oe1::CliMode::kList;
  opt.geometry = (inDir / "geo-train-06.json").string();
  opt.tables = (inDir / "tables").string();
  opt.view = 0;
  opt.seed = 3026092901ULL;
  opt.primaries = listPath;
  opt.roiMask = (inDir / "mask.txt").string();
  opt.transportSeed = 1026092901ULL;
  // The synthetic world is variant B (anomaly_0 is a G4SubtractionSolid); the
  // bad-variant scenarios label the same run A.
  opt.variant = (scenario == "bad-variant" || scenario == "bad-both") ? "A" : "B";
  opt.caseCode = 6;
  opt.replica = 1;
  opt.haveRegionPair = !noRegion;
  opt.regionPairNone = noRegion;
  opt.regionFirst = 0;
  opt.regionSecond = 1;
  opt.logLevel = min ? oe1::MuonLogLevel::kMin : oe1::MuonLogLevel::kFull;
  oe1::ListModeConfig lm;
  CHECK(oe1::ConfigureListMode(opt, cfg, list, mask, lm, err) == 0);
  CHECK(oe1::RecordProvenance(opt, lm, err));
  // A real run reports its run manager class; the harness declares the one a
  // list-mode run must have (bad-run-manager and bad-both declare another).
  lm.runManagerClass = scenario == "bad-run-manager" ? "G4MTRunManager"
                       : scenario == "bad-both"      ? "G4SerialRunManager"
                                                     : oe1::kExpectedRunManager;
  lm.masterEngine = G4Random::getTheEngine()->name();
  CHECK(lm.masterEngine == "MixMaxRng");

  World w = MakeWorld();

  // Plan S3: what main() does right after runManager->Initialize(). A failure here ends the
  // real process with exit code 4 before BeamOn, so nothing below would run.
  {
    const std::vector<std::string> built = oe1::BuiltGeometrySignature();
    CHECK(built.size() == 4u);
    const oe1::PreRunReport pre = oe1::CheckBeforeTransport(lm, built);
    const oe1::VariantCheck wantVariant =
        noRegion ? oe1::VariantCheck::kNotChecked
                 : ((scenario == "bad-variant" || scenario == "bad-both") ? oe1::VariantCheck::kMismatch
                                                                           : oe1::VariantCheck::kOk);
    CHECK(pre.variantCheck == wantVariant);
    if (scenario == "bad-variant") {
      CHECK(!pre.Ok() && pre.errors == std::vector<std::string>{"variant_geometry_mismatch"});
      CHECK(pre.messages.size() == 1u && pre.messages[0].find("--variant A") != std::string::npos);
    }
    else if (scenario == "bad-run-manager") {
      CHECK(!pre.Ok() && pre.errors == std::vector<std::string>{"run_manager_not_task"});
      CHECK(pre.messages.size() == 1u && pre.messages[0].find("G4MTRunManager") != std::string::npos);
    }
    else if (scenario == "bad-both") {
      CHECK(!pre.Ok() &&
            pre.errors == (std::vector<std::string>{"run_manager_not_task", "variant_geometry_mismatch"}));
      CHECK(pre.messages.size() == 2u);
    }
    else {
      CHECK(pre.Ok() && pre.messages.empty());
    }
    std::string codes;
    for (const std::string& e : pre.errors) codes += (codes.empty() ? "" : ",") + e;
    std::printf("pre-BeamOn checks: %s errors=%s\n", pre.Ok() ? "OK" : "FAIL (main would exit 4 here)",
                codes.empty() ? "none" : codes.c_str());
  }
  PrimaryGeneratorAction pga(0, cfg, nullptr);
  pga.SetPrimaryList(&list);
  EventAction ea(lm);
  // G4Run's destructor needs a run manager: allocate it and never delete it.
  auto* runPtr = new G4Run();
  G4Run& run = *runPtr;

  const ZVertical zv = VerticalZ(21.36, -30.71);
  const double zin = zv.zin, zout = zv.zout;
  const double X = 21.36, Y = -30.71;

  // Muon 16: legs of a vertical track through the centroid. The body grown by 1 m
  // has its top at z = 184.10 and its bottom at z = 91.50 on this vertical (its
  // nominal top and bottom are zout = 182.64 and zin = 92.96).
  struct Leg {
    double zPost;   // m: end of the leg
    double offMm;   // sideways displacement of the post-step point (mm)
    G4Material* mat;
    bool nearZ;     // expected: the chord meets both primitives grown by 1 m
  };
  const Leg legs16[] = {
      {300.0, 6.0, w.a0, false},        // far above the dense body
      {zout + 3.0, 7.0, w.a0, false},   // ends 1.5 m above the top of the grown body
      {zout + 0.5, 1.0, w.a0, true},    // ends 0.5 m above the nominal top: inside the margin
      {zin - 3.0, 2.0, w.a1, true},     // through Z, ends 3 m past its bottom
      {zin - 5.0, 8.0, w.a0, false}};   // 2 m further down, below the grown bottom

  // Muon k -> synthetic track. Returns after feeding the steps.
  auto track = [&](int id, G4Event& event) {
    const auto P = [&](double z) { return G4ThreeVector(X, Y, z); };
    switch (id) {
      case 0: {  // detected inside R_int, crossing Z in two steps (pre-step A0 then A1)
        Feed(ea, {P(400), P(150), 100.0, 98.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.001, {}});
        Feed(ea, {P(150), P(-5), 98.0, 95.0, w.a1, w.transport.get(), fGeomBoundary, w.pvBody, 1.001, {}});
        Feed(ea, {P(-5), G4ThreeVector(1000.0, 0.0, 1.5), 95.0, 94.0, w.air, w.transport.get(),
                  fGeomBoundary, w.pvDetector, 1.0, {}});
        ea.OnPlane(135, 34, 94.0, 7.93, 1.8);
        RunData::Instance().RecordHit(135, 34, 94.0);
        break;
      }
      case 1: {  // detected outside R_int, Z crossed inside ONE step with both ends outside
        Feed(ea, {P(400), P(-10), 50.0, 40.0, w.a1, w.ioni.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        Feed(ea, {P(-10), G4ThreeVector(1000.0, 0.0, 1.5), 40.0, 39.0, w.air, w.transport.get(),
                  fGeomBoundary, w.pvDetector, 1.0, {}});
        ea.OnPlane(50, 100, 39.0, 20.0, -15.0);
        RunData::Instance().RecordHit(50, 100, 39.0);
        break;
      }
      case 2: {  // reaches the plane outside the angular grid
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(1000, 0, 1.5), 20.0, 19.0, w.air,
                  w.transport.get(), fGeomBoundary, w.pvDetector, 1.0, {}});
        ea.OnPlane(-1, 100, 19.0, 60.0, 31.0);
        break;
      }
      case 3: {  // leaves the world
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-9000, 0, 300), 20.0, 20.0, w.air,
                  w.transport.get(), fWorldBoundary, nullptr, 1.0, {}});
        break;
      }
      case 4: {  // stops by ionisation, then decays at rest (an at-rest step: pre == post)
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-400, 0, 290), 0.5, 0.0, w.rock,
                  w.ioni.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        Feed(ea, {G4ThreeVector(-400, 0, 290), G4ThreeVector(-400, 0, 290), 0.0, 0.0, w.rock,
                  w.decay.get(), fAtRestDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 5: {  // stops by ionisation, then mu- capture at rest
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-499, 0, 299), 0.5, 0.0, w.rock,
                  w.ioni.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        Feed(ea, {G4ThreeVector(-499, 0, 299), G4ThreeVector(-499, 0, 299), 0.0, 0.0, w.rock,
                  w.capture.get(), fAtRestDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 6: {  // stops by ionisation
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-499, 0, 299), 0.5, 0.0, w.rock,
                  w.ioni.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 7: {  // ends with energy left, not at the plane, not in the world border
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-499, 0, 299), 5.0, 4.0, w.rock,
                  w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 8: {  // ends like muon 7; the "aborted" scenario aborts the event
        Feed(ea, {G4ThreeVector(-500, 0, 300), G4ThreeVector(-499, 0, 299), 5.0, 4.0, w.rock,
                  w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        if (scenario == "aborted") event.SetEventAborted();
        break;
      }
      case 9: {  // two separate entries into Z (bounce): both steps have both ends outside
        Feed(ea, {P(300), P(50), 30.0, 25.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        Feed(ea, {P(50), P(300), 25.0, 20.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 10: {  // tangent step: 1e-7 m inside the conduit wall (a graze, not deep)
        Feed(ea, {G4ThreeVector(41.3 - 1e-7, -100.0, 137.8), G4ThreeVector(41.3 - 1e-7, 50.0, 137.8),
                  10.0, 9.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 11: {  // lateral displacement of 0.3 mm inside one step
        StepSpec s{P(400), P(300) + G4ThreeVector(0.3e-3, 0.0, 0.0), 10.0, 9.9, w.rock,
                   w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, G4ThreeVector(0, 0, -1)};
        Feed(ea, s);
        break;
      }
      case 13: {  // 1e-13 m sliver at the entry (endpoint on the Z surface, pre-step A0)
        Feed(ea, {P(300), P(zout - 1e-13), 20.0, 20.0, w.a0, w.transport.get(), fGeomBoundary,
                  w.pvBody, 1.0, {}});
        Feed(ea, {P(zout - 1e-13), P(zin), 20.0, 19.0, w.a1, w.transport.get(), fGeomBoundary,
                  w.pvWorld, 1.0, {}});
        Feed(ea, {P(zin), P(-5), 19.0, 18.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 14: {  // 1e-13 m sliver at the exit (next step starts on the Z surface, pre-step A0)
        Feed(ea, {P(300), P(zout), 20.0, 20.0, w.a0, w.transport.get(), fGeomBoundary, w.pvBody, 1.0, {}});
        Feed(ea, {P(zout), P(zin + 1e-13), 20.0, 19.0, w.a1, w.transport.get(), fGeomBoundary,
                  w.pvWorld, 1.0, {}});
        Feed(ea, {P(zin + 1e-13), P(-5), 19.0, 18.0, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 15: {  // a zero-length step in the middle of the crossing must not split it
        Feed(ea, {P(300), P(150), 20.0, 19.5, w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        Feed(ea, {P(150), P(150), 19.5, 19.5, w.a1, w.msc.get(), fPostStepDoItProc, w.pvBody, 1.0, {}});
        Feed(ea, {P(150), P(-5), 19.5, 19.0, w.a1, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0, {}});
        break;
      }
      case 16: {  // lateral displacement of the steps near Z (plan correction C1):
                  // a vertical track through the centroid whose post-step points are
                  // displaced sideways (x) by the given mm, as WentzelVI does
        double x = X, z = 400.0;
        int k = 0;
        for (const Leg& leg : legs16) {
          const double xPost = x + leg.offMm * 1e-3;
          Feed(ea, {G4ThreeVector(x, Y, z), G4ThreeVector(xPost, Y, leg.zPost), 50.0 - k, 49.5 - k,
                    leg.mat, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.0,
                    G4ThreeVector(0, 0, -1)});
          x = xPost;
          z = leg.zPost;
          ++k;
        }
        break;
      }
      case 17: {  // steps beside the conduit wall (x = 41.3 is its +x face), crossing the dense body
        const double wall = 20.3 + 21.0;
        // 0.7 m outside the wall: inside the 1 m margin, no length in Z
        Feed(ea, {G4ThreeVector(wall + 0.7, -100.0, 137.8),
                  G4ThreeVector(wall + 0.7 + 4e-3, 50.0, 137.8), 10.0, 9.9, w.a1, w.msc.get(),
                  fPostStepDoItProc, w.pvWorld, 1.0, G4ThreeVector(0, 1, 0)});
        // 1.6 m outside the wall: beyond the margin
        Feed(ea, {G4ThreeVector(wall + 1.6, -100.0, 137.8),
                  G4ThreeVector(wall + 1.6 + 9e-3, 50.0, 137.8), 9.9, 9.8, w.a1, w.msc.get(),
                  fPostStepDoItProc, w.pvWorld, 1.0, G4ThreeVector(0, 1, 0)});
        break;
      }
      case 18: {  // plan S4: a 1 um step deep inside Z is "deep" but not longer than 2 eps
        Feed(ea, {P(150.0), P(150.0 - 1e-6), 20.0, 20.0, w.a1, w.msc.get(), fPostStepDoItProc,
                  w.pvBody, 1.0, {}});
        break;
      }
      default:  // 12: no step at all
        break;
    }
  };

  // ---- events
  const bool skipHit0 = scenario == "hist-mismatch";
  for (int id = 0; id < kN; ++id) {
    if (scenario == "missing-event" && id == 5) continue;
    for (int pass = 0; pass < (scenario == "dup-id" && id == 4 ? 2 : 1); ++pass) {
      G4Event event(id);
      pga.GeneratePrimaries(&event);
      ea.BeginOfEventAction(&event);
      if (skipHit0 && id == 0) {
        // Same track, but the histogram bookkeeping of SteppingAction is "lost".
        Feed(ea, {G4ThreeVector(21.36, -30.71, 400), G4ThreeVector(21.36, -30.71, 150), 100.0, 98.0,
                  w.a0, w.msc.get(), fPostStepDoItProc, w.pvWorld, 1.001, {}});
        ea.OnPlane(135, 34, 94.0, 7.93, 1.8);
      }
      else {
        track(id, event);
      }
      ea.EndOfEventAction(&event);
      run.RecordEvent(&event);
    }
  }
  if (scenario == "two-workers") RunData::Instance().NoteWorker(7, "MixMaxRng");
  if (scenario == "bad-engine") RunData::Instance().NoteWorker(0, "RanecuEngine");

  // ---- master finalisation, as in the real run
  const fs::path runDir = scenario == "bad-outdir" ? outDir / "does-not-exist" : outDir / "run";
  if (scenario != "bad-outdir") fs::create_directories(runDir);
  RunAction ra(cfg, 0, opt.seed, runDir.string(), 1, -1.0);
  ra.SetListMode(&lm);
  ra.BeginOfRunAction(&run);
  ra.EndOfRunAction(&run);

  const RunData& data = RunData::Instance();
  CHECK(data.Finalized());  // main() returns 4 when a list-mode run ends without this
  std::printf("errors after finalisation:");
  for (const auto& e : data.Errors()) std::printf(" %s", e.c_str());
  std::printf("%s\n", data.Errors().empty() ? " none" : "");

  const std::string stem = (runDir / "geo-train-06_view0").string();
  const fs::path muonCsv = stem + "_muones.csv";

  // ------------------------------------------------------------ failure scenarios
  if (scenario == "aborted") {
    CHECK(data.Errors().size() == 1u && data.Errors().count("aborted_events") == 1);
    Csv aborted;
    CHECK(ReadCsv(muonCsv, aborted));
    CHECK_EQ_STR(aborted.Get(8, "status"), "ABORTED");
    CHECK_EQ_STR(aborted.Get(7, "status"), "OTHER");
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "hist-mismatch") {
    CHECK(data.HasErrors());
    CHECK(data.Errors().count("hist_reconstruction_mismatch") == 1);
    CHECK(data.Errors().count("detected_count_mismatch") == 1);
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "missing-event") {
    CHECK(data.Errors().count("ids_incomplete_or_duplicate") == 1);
    CHECK(data.Errors().count("events_transported_mismatch") == 1);
    CHECK(data.Errors().count("generated_count_mismatch") == 1);
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "dup-id") {
    CHECK(data.Errors().count("ids_incomplete_or_duplicate") == 1);
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "two-workers") {
    CHECK(data.Errors().count("worker_threads_not_one") == 1);
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "bad-engine") {
    CHECK(data.Errors().count("worker_engine_not_mixmax") == 1);
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "bad-outdir") {
    CHECK(data.Errors().count("hist_write_failed") == 1);
    CHECK(data.Errors().count("muon_log_write_failed") == 1);
    CHECK(data.Errors().count("meta_write_failed") == 1);
    CHECK(!fs::exists(runDir));
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "bad-variant") {
    // --variant A on a geometry whose anomaly_0 is a G4SubtractionSolid (plan S2)
    CHECK(data.Errors().size() == 1u && data.Errors().count("variant_geometry_mismatch") == 1);
    std::ifstream in(stem + ".meta.json");
    const nlohmann::json meta = nlohmann::json::parse(in);
    const nlohmann::json& oe = meta.at("oe1_etapa1");
    CHECK(oe.at("variant") == "A");
    CHECK(oe.at("checks").at("variant_matches_geometry") == false);
    CHECK(oe.at("errors") == nlohmann::json::array({"variant_geometry_mismatch"}));
    const std::string note = oe.at("geometry_signature").at("variant_check_note").get<std::string>();
    CHECK(note.find("--variant A") != std::string::npos);
    CHECK(note.find("G4Tubs") != std::string::npos && note.find("G4SubtractionSolid") != std::string::npos);
    CHECK(fs::exists(muonCsv));  // the run still writes its evidence
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "bad-run-manager") {
    // a run manager other than G4TaskRunManager is an error (plan S7), like a worker count != 1
    CHECK(data.Errors().size() == 1u && data.Errors().count("run_manager_not_task") == 1);
    std::ifstream in(stem + ".meta.json");
    const nlohmann::json meta = nlohmann::json::parse(in);
    const nlohmann::json& oe = meta.at("oe1_etapa1");
    CHECK(oe.at("runtime").at("run_manager_class") == "G4MTRunManager");
    CHECK(oe.at("errors") == nlohmann::json::array({"run_manager_not_task"}));
    CHECK(oe.at("checks").at("variant_matches_geometry") == true);  // nothing else is wrong
    return g_fail == 0 ? 0 : 1;
  }
  if (scenario == "bad-both") {
    // both pre-BeamOn checks failing at once (main would have exited before the run); finishing
    // the run anyway shows that the second line of defence reports the same two codes
    CHECK(data.Errors() == (std::set<std::string>{"run_manager_not_task", "variant_geometry_mismatch"}));
    std::ifstream in(stem + ".meta.json");
    const nlohmann::json meta = nlohmann::json::parse(in);
    const nlohmann::json& oe = meta.at("oe1_etapa1");
    CHECK(oe.at("runtime").at("run_manager_class") == "G4SerialRunManager");
    CHECK(oe.at("checks").at("variant_matches_geometry") == false);
    CHECK(oe.at("errors") == nlohmann::json::array({"run_manager_not_task", "variant_geometry_mismatch"}));
    return g_fail == 0 ? 0 : 1;
  }

  // ------------------------------------------------------------ ok scenarios
  CHECK(!data.HasErrors());
  CHECK(fs::exists(muonCsv));
  CHECK(!fs::exists(fs::path(muonCsv.string() + ".part")));  // renamed, none left over
  CHECK(fs::exists(stem + "_hist.csv") && fs::exists(stem + ".meta.json"));

  Csv csv;
  CHECK(ReadCsv(muonCsv, csv));
  CHECK(csv.header.size() == static_cast<std::size_t>(oe1::kMuonCsvColumns));
  CHECK(csv.nRows == static_cast<std::size_t>(kN));
  CHECK(csv.sortedById);
  auto gs = [&](int id, const char* col) { return csv.Get(id, col); };
  auto gn = [&](int id, const char* col) { return csv.Num(id, col); };

  // Identity and random-stream evidence, every row.
  for (int id = 0; id < kN; ++id) {
    CHECK_EQ_STR(gs(id, "caso"), "6");
    CHECK_EQ_STR(gs(id, "geometry_id"), "geo-train-06");
    CHECK_EQ_STR(gs(id, "variant"), "B");
    CHECK_EQ_STR(gs(id, "replica"), "1");
    CHECK_EQ_STR(gs(id, "list_sha256"), list.FileSha256());
    // rng0: state of a MixMaxRng seeded with the documented key.
    long key[4] = {id, 1026092901L, 6L, 0x4F455431L};
    CLHEP::MixMaxRng ref;
    ref.setSeeds(key, 4);
    CHECK_EQ_STR(gs(id, "rng0"), oe1::Hex16(oe1::HashEngineWords(ref.put())));
    // No hook consumed a random number: the end state equals the start state.
    CHECK_EQ_STR(gs(id, "rng1"), gs(id, "rng0"));
    // prim_fnv from the list row through the gun.
    const oe1::PrimaryRecord& r = list.At(static_cast<std::size_t>(id));
    oe1::Fnv1a64 h;
    h.F64(r.x_m * m);
    h.F64(r.y_m * m);
    h.F64(r.z_m * m);
    const G4ThreeVector d = G4ThreeVector(r.dx, r.dy, r.dz).unit();
    h.F64(d.x());
    h.F64(d.y());
    h.F64(d.z());
    h.F64(r.ekin_gev * GeV);
    h.I32(r.pdg);
    CHECK_EQ_STR(gs(id, "prim_fnv"), oe1::Hex16(h.Value()));
  }
  // Distinct events start from distinct streams.
  {
    std::map<std::string, int> seen;
    for (int id = 0; id < kN; ++id) ++seen[gs(id, "rng0")];
    CHECK(seen.size() == static_cast<std::size_t>(kN));
  }

  // Final states.
  const char* status[kN] = {"DET",        "DET",     "PLANE_OUT", "WORLD_EXIT", "DECAY",
                            "MU_CAPTURE", "STOPPED", "OTHER",     "OTHER",      "OTHER",
                            "OTHER",      "OTHER",   "OTHER",     "OTHER",      "OTHER",
                            "OTHER",      "OTHER",   "OTHER",     "OTHER"};
  for (int id = 0; id < kN; ++id) CHECK_EQ_STR(gs(id, "status"), status[id]);
  CHECK_EQ_STR(gs(0, "detected"), "1");
  CHECK_EQ_STR(gs(0, "iel"), "34");
  CHECK_EQ_STR(gs(0, "iaz"), "135");
  CHECK_EQ_STR(gs(0, "in_rint"), "1");        // (34,135) is inside the mask
  CHECK_EQ_STR(gs(1, "in_rint"), "0");        // (100,50) is outside
  CHECK_EQ_STR(gs(2, "detected"), "0");
  CHECK_EQ_STR(gs(2, "iel"), "-1");
  CHECK_EQ_STR(gs(2, "iaz"), "-1");
  CHECK_EQ_STR(gs(2, "in_rint"), "-1");
  CHECK_NEAR(gn(2, "ekin_plane_gev"), 19.0, 1e-12);  // PLANE_OUT keeps the plane state
  CHECK_NEAR(gn(2, "el_plane_deg"), 60.0, 1e-12);
  CHECK_EQ_STR(gs(3, "ekin_plane_gev"), "");
  CHECK_EQ_STR(gs(3, "in_rint"), "-1");
  CHECK_EQ_STR(gs(0, "end_process"), "Transportation");
  CHECK_EQ_STR(gs(4, "end_process"), "Decay");
  CHECK_EQ_STR(gs(5, "end_process"), "muMinusCaptureAtRest");
  CHECK_EQ_STR(gs(6, "end_process"), "muIoni");
  CHECK_EQ_STR(gs(12, "end_process"), "none");
  CHECK_EQ_STR(gs(0, "n_steps"), "3");
  CHECK_EQ_STR(gs(1, "n_steps"), "2");
  CHECK_EQ_STR(gs(9, "n_steps"), "2");
  CHECK_EQ_STR(gs(12, "n_steps"), "0");
  CHECK_EQ_STR(gs(4, "n_steps"), "2");   // ionisation stop + at-rest step
  CHECK_EQ_STR(gs(5, "n_steps"), "2");
  CHECK_EQ_STR(gs(15, "n_steps"), "3");
  CHECK_NEAR(gn(0, "ekin_plane_gev"), 94.0, 1e-12);

  if (min) {
    // Reduced log: same identity/state columns, none of the step-accumulated ones.
    for (int id = 0; id < kN; ++id) {
      for (const char* col : {"end_x_m", "end_y_m", "end_z_m", "end_vol", "crossed", "lz_m",
                              "lz_true_m", "lc_m", "le_m", "n_steps_z", "mat_in", "e_in_gev",
                              "max_lat_mm", "max_lat_z_mm", "sum_lat_z_mm"}) {
        CHECK_EQ_STR(gs(id, col), "");
      }
    }
  }
  else if (noRegion) {
    // --region-pair none (the geo-train-01 way): end points and max_lat_mm are still
    // logged, every crossing field and the lateral displacement near Z are "not applicable".
    CHECK_NEAR(gn(0, "end_x_m"), 1000.0, 1e-9);
    CHECK_NEAR(gn(0, "end_z_m"), 1.5, 1e-9);
    CHECK_EQ_STR(gs(0, "end_vol"), "detector");
    for (int id = 0; id < kN; ++id) {
      CHECK_EQ_STR(gs(id, "crossed"), "-1");
      for (const char* col : {"lz_m", "lz_true_m", "lz_rock_m", "lz_a0_m", "lz_a1_m", "lz_other_m",
                              "lc_m", "le_m", "rho_lz_gcm2", "n_steps_z", "n_steps_z_both_out",
                              "n_entries_z", "n_graze_z", "mat_in", "mat_out", "e_in_gev",
                              "e_in_lo_gev", "e_in_hi_gev", "e_out_gev", "e_out_lo_gev",
                              "e_out_hi_gev", "max_lat_z_mm", "sum_lat_z_mm"}) {
        CHECK_EQ_STR(gs(id, col), "");
      }
    }
    CHECK_NEAR(gn(11, "max_lat_mm"), 0.3, 1e-9);
    CHECK_NEAR(gn(16, "max_lat_mm"), 8.0, 1e-6);
    CHECK_EQ_STR(gs(12, "max_lat_mm"), "");  // no step, no displacement
  }
  else {
    // End points and volumes.
    CHECK_NEAR(gn(0, "end_x_m"), 1000.0, 1e-9);
    CHECK_NEAR(gn(0, "end_z_m"), 1.5, 1e-9);
    CHECK_EQ_STR(gs(0, "end_vol"), "detector");
    CHECK_EQ_STR(gs(3, "end_vol"), "OutOfWorld");
    CHECK_EQ_STR(gs(4, "end_vol"), "world");
    CHECK_EQ_STR(gs(12, "end_vol"), "");

    // Muon 0: two steps in Z, entry from A0, exit into A1, one entry.
    const double lz = zout - zin;
    CHECK_EQ_STR(gs(0, "crossed"), "1");
    CHECK_NEAR(gn(0, "lz_m"), lz, 1e-9);
    CHECK_NEAR(gn(0, "lz_a0_m"), zout - 150.0, 1e-9);
    CHECK_NEAR(gn(0, "lz_a1_m"), 150.0 - zin, 1e-9);
    CHECK_NEAR(gn(0, "lz_rock_m"), 0.0, 1e-12);
    CHECK_NEAR(gn(0, "lz_other_m"), 0.0, 1e-12);
    CHECK_NEAR(gn(0, "lz_true_m"), lz * 1.001, 1e-9);
    CHECK_NEAR(gn(0, "lc_m"), 340.0, 1e-9);
    CHECK_EQ_STR(gs(0, "n_steps_z"), "2");
    CHECK_EQ_STR(gs(0, "n_steps_z_both_out"), "0");
    CHECK_EQ_STR(gs(0, "n_entries_z"), "1");
    CHECK_EQ_STR(gs(0, "n_graze_z"), "0");
    CHECK_EQ_STR(gs(0, "mat_in"), "A0");
    CHECK_EQ_STR(gs(0, "mat_out"), "A1");
    CHECK_NEAR(gn(0, "e_in_gev"), 100.0 + (98.0 - 100.0) * (400.0 - zout) / 250.0, 1e-9);
    CHECK_NEAR(gn(0, "e_in_lo_gev"), 98.0, 1e-12);
    CHECK_NEAR(gn(0, "e_in_hi_gev"), 100.0, 1e-12);
    CHECK_NEAR(gn(0, "e_out_gev"), 98.0 + (95.0 - 98.0) * (150.0 - zin) / 155.0, 1e-9);
    CHECK_NEAR(gn(0, "e_out_lo_gev"), 95.0, 1e-12);
    CHECK_NEAR(gn(0, "e_out_hi_gev"), 98.0, 1e-12);
    // rho * L = sum(density [g/cm3] * length [cm]).
    CHECK_NEAR(gn(0, "rho_lz_gcm2"),
               (1.832 * (zout - 150.0) + 3.285 * (150.0 - zin)) * 100.0, 1e-6);
    CHECK_NEAR(gn(0, "max_lat_mm"), 0.0, 1e-9);

    // Muon 1: one step with both ends outside Z.
    CHECK_NEAR(gn(1, "lz_m"), lz, 1e-9);
    CHECK_EQ_STR(gs(1, "n_steps_z"), "1");
    CHECK_EQ_STR(gs(1, "n_steps_z_both_out"), "1");
    CHECK_EQ_STR(gs(1, "n_entries_z"), "1");
    CHECK_EQ_STR(gs(1, "mat_in"), "A1");
    CHECK_EQ_STR(gs(1, "mat_out"), "A1");
    CHECK_NEAR(gn(1, "lz_a1_m"), lz, 1e-9);
    CHECK_NEAR(gn(1, "e_in_gev"), 50.0 + (40.0 - 50.0) * (400.0 - zout) / 410.0, 1e-9);
    CHECK_NEAR(gn(1, "e_out_gev"), 50.0 + (40.0 - 50.0) * (400.0 - zin) / 410.0, 1e-9);

    // Muon 13: a sliver of ~1e-13 m at the entry (pre-step A0) is not a Z step: the
    // categorical fields describe the real step only; the exact sums keep the sliver.
    CHECK_EQ_STR(gs(13, "crossed"), "1");
    CHECK_NEAR(gn(13, "lz_m"), lz, 1e-9);
    CHECK_NEAR(gn(13, "lz_a1_m"), lz, 1e-9);
    CHECK_NEAR(gn(13, "lz_a0_m"), 0.0, 1e-9);
    CHECK_EQ_STR(gs(13, "n_steps_z"), "1");
    CHECK_EQ_STR(gs(13, "n_steps_z_both_out"), "0");
    CHECK_EQ_STR(gs(13, "n_entries_z"), "1");
    CHECK_EQ_STR(gs(13, "mat_in"), "A1");
    CHECK_EQ_STR(gs(13, "mat_out"), "A1");
    CHECK_NEAR(gn(13, "e_in_gev"), 20.0, 1e-9);
    CHECK_NEAR(gn(13, "e_out_gev"), 19.0, 1e-6);
    // Muon 14: the same at the exit (the step after the real one starts inside Z by ~1e-13 m).
    CHECK_EQ_STR(gs(14, "crossed"), "1");
    CHECK_NEAR(gn(14, "lz_m"), lz, 1e-9);
    CHECK_NEAR(gn(14, "lz_a0_m"), 0.0, 1e-9);
    CHECK_EQ_STR(gs(14, "n_steps_z"), "1");
    CHECK_EQ_STR(gs(14, "n_entries_z"), "1");
    CHECK_EQ_STR(gs(14, "mat_in"), "A1");
    CHECK_EQ_STR(gs(14, "mat_out"), "A1");
    // Muon 15: a zero-length step inside Z between two real steps: one entry, two Z steps.
    CHECK_EQ_STR(gs(15, "crossed"), "1");
    CHECK_NEAR(gn(15, "lz_m"), lz, 1e-9);
    CHECK_NEAR(gn(15, "lz_a0_m"), zout - 150.0, 1e-9);
    CHECK_NEAR(gn(15, "lz_a1_m"), 150.0 - zin, 1e-9);
    CHECK_EQ_STR(gs(15, "n_steps_z"), "2");
    CHECK_EQ_STR(gs(15, "n_entries_z"), "1");
    CHECK_EQ_STR(gs(15, "mat_in"), "A0");
    CHECK_EQ_STR(gs(15, "mat_out"), "A1");
    // Muons that never touch Z: crossed = 0 and empty crossing fields.
    for (int id : {2, 3, 4, 5, 6, 7, 8, 11, 12}) {
      CHECK_EQ_STR(gs(id, "crossed"), "0");
      CHECK_NEAR(gn(id, "lz_m"), 0.0, 1e-12);
      CHECK_EQ_STR(gs(id, "n_steps_z"), "0");
      CHECK_EQ_STR(gs(id, "mat_in"), "");
      CHECK_EQ_STR(gs(id, "e_in_gev"), "");
    }
    // Muon 9: two separate entries.
    CHECK_EQ_STR(gs(9, "crossed"), "1");
    CHECK_EQ_STR(gs(9, "n_steps_z"), "2");
    CHECK_EQ_STR(gs(9, "n_steps_z_both_out"), "2");
    CHECK_EQ_STR(gs(9, "n_entries_z"), "2");
    CHECK_NEAR(gn(9, "lz_m"), 2.0 * lz, 1e-9);
    CHECK_NEAR(gn(9, "lz_a0_m"), 2.0 * lz, 1e-9);
    // Muon 10: a graze, not a crossing.
    CHECK_EQ_STR(gs(10, "crossed"), "0");
    CHECK_EQ_STR(gs(10, "n_graze_z"), "1");
    CHECK_NEAR(gn(10, "lz_m"), 0.004098780406609269, 1e-9);
    CHECK_EQ_STR(gs(10, "n_steps_z"), "1");
    // Muon 11: lateral displacement of 0.3 mm far from Z (the step is above the dense body
    // and the chord does not meet both grown primitives): counted in max_lat_mm only.
    CHECK_NEAR(gn(11, "max_lat_mm"), 0.3, 1e-9);
    CHECK_NEAR(gn(11, "max_lat_z_mm"), 0.0, 1e-12);
    CHECK_NEAR(gn(11, "sum_lat_z_mm"), 0.0, 1e-12);
    CHECK_NEAR(gn(0, "max_lat_mm"), 0.0, 1e-9);

    // Muon 16 (plan correction C1): post-step points displaced sideways. max_lat_mm is the
    // largest over ALL steps; max_lat_z_mm / sum_lat_z_mm only see the steps near Z.
    {
      double maxAll = 0.0, maxNear = 0.0, sumNear = 0.0;
      int nNear = 0;
      for (const Leg& leg : legs16) {
        maxAll = std::max(maxAll, leg.offMm);
        if (leg.nearZ) {
          maxNear = std::max(maxNear, leg.offMm);
          sumNear += leg.offMm;
          ++nNear;
        }
      }
      CHECK(nNear == 2 && maxAll == 8.0 && maxNear == 2.0 && sumNear == 3.0);  // the design
      CHECK_NEAR(gn(16, "max_lat_mm"), maxAll, 1e-6);
      CHECK_NEAR(gn(16, "max_lat_z_mm"), maxNear, 1e-6);
      CHECK_NEAR(gn(16, "sum_lat_z_mm"), sumNear, 1e-6);
      CHECK(gn(16, "sum_lat_z_mm") >= gn(16, "max_lat_z_mm"));
      CHECK(gn(16, "max_lat_z_mm") <= gn(16, "max_lat_mm"));
      CHECK_EQ_STR(gs(16, "crossed"), "1");  // the leg through Z
      CHECK_EQ_STR(gs(16, "n_steps_z"), "1");
      CHECK_EQ_STR(gs(16, "n_entries_z"), "1");
      // The track has drifted 14 mm sideways where it enters the body, whose height on the
      // vertical changes by 0.4 m per m of x there: the length differs by ~5 mm from the
      // undisplaced vertical.
      CHECK_NEAR(gn(16, "lz_m"), zout - zin, 2e-2);
      CHECK_EQ_STR(gs(16, "mat_in"), "A1");
    }
    // Muon 17: beside the conduit wall. The step 0.7 m outside is near (no length in Z), the
    // one 1.6 m outside is not.
    CHECK_NEAR(gn(17, "max_lat_mm"), 9.0, 1e-6);
    CHECK_NEAR(gn(17, "max_lat_z_mm"), 4.0, 1e-6);
    CHECK_NEAR(gn(17, "sum_lat_z_mm"), 4.0, 1e-6);
    CHECK_EQ_STR(gs(17, "crossed"), "0");
    CHECK_NEAR(gn(17, "lz_m"), 0.0, 1e-12);
    CHECK_EQ_STR(gs(17, "n_steps_z"), "0");
    // Every other muon: their steps run along the chord (a rounding-size lateral offset at most),
    // so the two columns are present (0, never "not applicable") and ~0.
    for (int id : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18}) {
      CHECK(!gs(id, "max_lat_z_mm").empty() && !gs(id, "sum_lat_z_mm").empty());
      CHECK_NEAR(gn(id, "max_lat_z_mm"), 0.0, 1e-8);
      CHECK_NEAR(gn(id, "sum_lat_z_mm"), 0.0, 1e-8);
    }

    // Muon 18 (plan S4): a 1 um step deep inside Z is "deep" for the detector but not longer
    // than 2 eps: it adds to the exact length and sets no categorical field, so crossed = 0
    // and there is no Z step.
    CHECK_EQ_STR(gs(18, "crossed"), "0");
    CHECK_NEAR(gn(18, "lz_m"), 1e-6, 1e-9);
    CHECK_NEAR(gn(18, "lz_a1_m"), 1e-6, 1e-9);
    CHECK_EQ_STR(gs(18, "n_steps_z"), "0");
    CHECK_EQ_STR(gs(18, "n_entries_z"), "0");
    CHECK_EQ_STR(gs(18, "mat_in"), "");
    CHECK_EQ_STR(gs(18, "e_in_gev"), "");
    // The invariant behind S4, on every row: crossed = 1 implies a Z step and its material.
    for (int id = 0; id < kN; ++id) {
      if (gs(id, "crossed") == "1") {
        CHECK(gn(id, "n_steps_z") >= 1.0);
        CHECK(!gs(id, "mat_in").empty() && !gs(id, "mat_out").empty());
        CHECK(!gs(id, "e_in_gev").empty() && !gs(id, "e_out_gev").empty());
      }
    }
  }

  // ---- hist.csv reproduces the two detected muons and only them
  {
    std::ifstream in(stem + "_hist.csv");
    std::string line;
    long total = 0, at34_135 = 0, at100_50 = 0;
    int row = 0;
    while (std::getline(in, line)) {
      const auto f = SplitCsv(line);
      CHECK(f.size() == 256u);
      for (int c = 0; c < static_cast<int>(f.size()); ++c) {
        const long v = std::strtol(f[static_cast<std::size_t>(c)].c_str(), nullptr, 10);
        total += v;
        if (row == 34 && c == 135) at34_135 = v;
        if (row == 100 && c == 50) at100_50 = v;
      }
      ++row;
    }
    CHECK(row == 256);
    CHECK(total == 2 && at34_135 == 1 && at100_50 == 1);
  }

  // ---- meta.json: legacy keys intact, oe1_etapa1 present, no timing, no paths
  {
    std::ifstream in(stem + ".meta.json");
    const nlohmann::json meta = nlohmann::json::parse(in);
    CHECK(meta.at("events_transported") == kN);
    CHECK(meta.at("detected_in_bins") == 2);
    CHECK(meta.at("ecomug_draws_total") == kN);
    const nlohmann::json& o = meta.at("oe1_etapa1");
    CHECK(o.at("format") == "oe1-etapa1/1");
    CHECK(o.at("variant") == "B");
    CHECK(o.at("case_code") == 6);
    CHECK(o.at("replica") == 1);
    CHECK(o.at("list").at("sha256") == list.FileSha256());
    CHECK(o.at("list").at("file") == "geo-train-06_v0_r1_s2026092901.primarios.csv");
    CHECK(o.at("list").at("canon_fnv1a64") == oe1::Hex16(list.CanonFnv()));
    CHECK(o.at("list").at("source_seed") == 2026092901LL);
    CHECK(o.at("roi_mask").at("ones") == 34 * 8);
    CHECK(o.at("roi_mask").at("file") == "mask.txt");
    CHECK(o.at("seeds").at("master_cli") == 3026092901ULL);
    CHECK(o.at("seeds").at("master_applied") == 878609254LL);
    CHECK(o.at("seeds").at("transport_seed") == 1026092901ULL);
    CHECK(o.at("runtime").at("worker_engines") == nlohmann::json::array({"MixMaxRng"}));
    CHECK(o.at("runtime").at("run_manager_class") == "G4TaskRunManager");
    CHECK(o.at("region").at("applicable") == !noRegion);
    CHECK(o.at("region").at("explicit_none") == noRegion);
    if (!noRegion) {
      CHECK(o.at("region").at("pair") == nlohmann::json::array({0, 1}));
      CHECK(o.at("region").at("kinds") == nlohmann::json::array({"conduit", "dense_body"}));
      CHECK(o.at("region").at("near_margin_m") == 1.0);
      // the definitions say that the chord is not the path (plan correction C1)
      const std::string defs = o.at("region").at("definitions").get<std::string>();
      CHECK(defs.find("CHORD") != std::string::npos && defs.find("not the path") != std::string::npos);
      CHECK(defs.find("sum_lat_z_mm") != std::string::npos);
    }
    CHECK(o.at("counts").at("status").at("DET") == 2);
    CHECK(o.at("counts").at("status").at("ABORTED") == 0);
    CHECK(o.at("counts").at("status").at("OTHER") == 12);
    CHECK(o.at("region").at("geometry_anomalies") == 2);
    {
      const nlohmann::json expected = nlohmann::json::array(
          {"anomaly_0:G4SubtractionSolid:anomaly_0_mat", "anomaly_1:G4Box:anomaly_1_mat",
           "detector:G4Box:G4_AIR", "world:G4Box:G4_AIR"});
      CHECK(o.at("geometry_signature").at("volumes") == expected);
      CHECK(o.at("geometry_signature").at("fnv1a64").get<std::string>().size() == 16u);
    }
    CHECK(o.at("counts").at("in_rint") == 1);
    // Provenance of the executable and of the fixed inputs (plan S2): file names and hashes.
    CHECK(o.at("binary_sha256").get<std::string>() == lm.binarySha256);
    CHECK(oe1::IsSha256Hex(o.at("binary_sha256").get<std::string>()));
    {
      std::string self;
      CHECK(oe1::Sha256File("/proc/self/exe", self));
      CHECK(o.at("binary_sha256") == self);
    }
    CHECK(o.at("inputs").at("geometry_json").at("file") == "geo-train-06.json");
    CHECK(o.at("inputs").at("geometry_json").at("sha256") == oe1::Sha256Hex(geometryText));
    for (int t = 0; t < 3; ++t) {
      CHECK(o.at("inputs").at("tables").at(tableNames[t]) == oe1::Sha256Hex(tableTexts[t]));
    }
    CHECK(o.at("inputs").at("tables").size() == 3u);
    // Not computed means null, never 0 (a zero would pass a check vacuously).
    if (!zOn) {
      for (const char* k : {"crossed", "z_muons", "graze_muons", "steps_z", "steps_z_both_out",
                            "entries_z", "graze_steps", "max_lat_z_mm", "max_sum_lat_z_mm",
                            "near_z_steps", "near_z_muons"}) {
        CHECK(o.at("counts").at(k).is_null());
      }
      CHECK(o.at("counts").at("max_lat_mm").is_null() == min);
      CHECK(o.at("counts").at("max_lat_mm").is_number() == noRegion);
    }
    else {
      CHECK(o.at("counts").at("crossed") == 7);  // muons 0, 1, 9, 13, 14, 15, 16
      CHECK(o.at("counts").at("max_lat_mm").is_number());
      // Largest displacement over all steps: the 9 mm step of muon 17 (muon 16 has 8 mm). Near Z
      // the largest is the 4 mm step of muon 17 (muon 16: 2 mm) and the largest per-muon sum is
      // also muon 17's 4 mm (muon 16: 1 + 2 = 3 mm); the other near steps follow their chord.
      CHECK_NEAR(o.at("counts").at("max_lat_mm").get<double>(), 9.0, 1e-6);
      CHECK_NEAR(o.at("counts").at("max_lat_z_mm").get<double>(), 4.0, 1e-6);
      CHECK_NEAR(o.at("counts").at("max_sum_lat_z_mm").get<double>(), 4.0, 1e-6);
      // Steps near Z: muons 0 (2), 1 (1), 9 (2), 10 (1), 13 (3), 14 (3), 15 (3), 16 (2), 17 (1),
      // 18 (1) = 19 in 10 muons; muon 11 is above the body and the others are far away.
      CHECK(o.at("counts").at("near_z_steps").get<long>() == 19);
      CHECK(o.at("counts").at("near_z_muons").get<long>() == 10);
    }
    for (const char* k : {"ids_complete_unique", "hist_reconstructed_equal", "detected_equal",
                          "events_equal", "generated_equal", "ekin_sum_equal", "writes_ok"}) {
      CHECK(o.at("checks").at(k) == true);
    }
    // The variant agrees with the built geometry when a region pair names the anomalies to
    // look at; without a pair it is not checked (null).
    if (noRegion) CHECK(o.at("checks").at("variant_matches_geometry").is_null());
    else CHECK(o.at("checks").at("variant_matches_geometry") == true);
    CHECK(o.at("muon_log").at("rows") == kN);
    CHECK(o.at("muon_log").at("level") == (min ? "min" : "full"));
    CHECK(o.at("muon_log").at("columns") == oe1::kMuonCsvColumns);
    CHECK(o.at("errors").empty());
    const std::string dump = o.dump();
    // No timing fields (they would break A/A byte identity of the meta).
    for (const char* forbidden : {"wall", "seconds", "rusage", "elapsed", "cpu_"}) {
      CHECK(dump.find(forbidden) == std::string::npos);
    }
    CHECK(dump.find(outDir.string()) == std::string::npos);  // no absolute output path
    CHECK(dump.find(inDir.string()) == std::string::npos);

    // getrusage of the process (plan W3): top-level keys next to wall_seconds, outside
    // oe1_etapa1; their names contain "rusage" so the A/A comparison of the analysis ignores them.
    CHECK(meta.at("wall_seconds").is_number());
    CHECK(meta.at("rusage_user_cpu_seconds").is_number_float());
    CHECK(meta.at("rusage_system_cpu_seconds").is_number_float());
    CHECK(meta.at("rusage_user_cpu_seconds").get<double>() >= 0.0);
    CHECK(meta.at("rusage_system_cpu_seconds").get<double>() >= 0.0);
    CHECK(meta.at("rusage_max_rss_kb").is_number_integer());
    CHECK(meta.at("rusage_max_rss_kb").get<long long>() > 1000);  // kB: this process is far above 1 MB
  }
  return g_fail == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s SCENARIO OUTDIR\n", argv[0]);
    return 2;
  }
  const std::string scenario = argv[1];
  const fs::path outDir = argv[2];
  const std::vector<std::string> known = {"ok-full", "ok-min", "ok-noregion", "aborted",
                                          "hist-mismatch", "missing-event", "dup-id",
                                          "two-workers", "bad-engine", "bad-outdir", "bad-variant",
                                          "bad-run-manager", "bad-both"};
  if (std::find(known.begin(), known.end(), scenario) == known.end() || !fs::is_directory(outDir)) {
    std::fprintf(stderr, "unknown scenario or OUTDIR is not a directory\n");
    return 2;
  }
  const int rc = Run(scenario, outDir);
  std::printf("%s [%s]: %d checks, %d failed\n", rc == 0 ? "PASS" : "FAIL", scenario.c_str(),
              g_checks, g_fail);
  std::fflush(stdout);
  return rc;
}
