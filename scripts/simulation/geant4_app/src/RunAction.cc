#include "RunAction.hh"
#include "AzimuthGrid.hh"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>
#include <vector>

#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Run.hh"
#include "G4Threading.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Version.hh"

#include "MuonRecord.hh"
#include "Oe1Hash.hh"
#include "Oe1ListMode.hh"
#include "RunData.hh"
#include "json.hpp"

namespace {
double nowSeconds()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
}  // namespace

// ---------------------------------------------------------------------------
// OE1 stage 1 (P2): list-mode finalisation on the master. Only reached when
// SetListMode() was given a configuration; legacy runs never enter here.
namespace {

struct Oe1Summary {
  std::size_t expected = 0;
  std::size_t rows = 0;
  bool idsCompleteUnique = false;
  bool histEqual = false;
  bool detectedEqual = false;
  bool eventsEqual = false;
  bool generatedEqual = false;
  bool ekinSumEqual = false;
  bool histWriteOk = false;
  bool muonLogOk = false;
  double ekinSumRebuiltGeV = 0.0;
  double ekinSumRunGeV = 0.0;
  std::array<uint64_t, oe1::kNumMuonStatus> statusCounts{};
  uint64_t nInRint = 0;
  uint64_t nCrossed = 0;
  uint64_t nZMuons = 0;
  uint64_t nGrazeMuons = 0;
  uint64_t nStepsZ = 0;
  uint64_t nStepsZBothOut = 0;
  uint64_t nEntriesZ = 0;
  uint64_t nGrazeSteps = 0;
  double maxLatMm = 0.0;       // diagnostic, all primary steps
  double maxLatZMm = 0.0;      // steps near Z (plan correction C1)
  double maxSumLatZMm = 0.0;   // largest per-muon sum over the steps near Z
  uint64_t nStepsNearZ = 0;
  uint64_t nMuonsNearZ = 0;
  std::string muonLogFile;
  uint64_t muonLogFnv = 0;
  uint64_t muonLogBytes = 0;
  std::vector<std::string> signature;  // geometry signature (sorted)
  oe1::VariantCheck variantCheck = oe1::VariantCheck::kNotChecked;
  std::string variantWhy;
};

std::string FileNameOf(const std::string& path)
{
  return std::filesystem::path(path).filename().string();
}

// Writes the per-muon log to <final>.part, then renames it; never overwrites.
bool WriteMuonLog(const oe1::ListModeConfig& lm, const std::vector<oe1::MuonRecord>& recs,
                  const std::string& finalPath, Oe1Summary& s)
{
  const std::string partPath = finalPath + ".part";
  std::error_code ec;
  if (std::filesystem::exists(finalPath, ec) || std::filesystem::exists(partPath, ec)) {
    G4cerr << "OE1: refusing to overwrite " << finalPath << G4endl;
    return false;
  }
  oe1::MuonRowConstants k;
  k.caseCode = lm.caseCode;
  k.geometryId = lm.geometryId;
  k.view = lm.view;
  k.variant = lm.variant;
  k.replica = lm.replica;
  k.listSha256 = lm.list->FileSha256();
  oe1::MuonLogShape shape;
  shape.fullLog = lm.logLevel == oe1::MuonLogLevel::kFull;
  shape.regionApplicable = lm.regionApplicable;

  std::ofstream out(partPath, std::ios::binary);
  if (!out) return false;
  oe1::Fnv1a64 fnv;
  std::string buf = std::string(oe1::MuonCsvHeader()) + "\n";
  buf.reserve(1U << 20);
  for (const auto& r : recs) {
    oe1::AppendMuonCsvRow(buf, k, shape, r);
    if (buf.size() >= (1U << 20)) {
      fnv.Bytes(buf.data(), buf.size());
      s.muonLogBytes += buf.size();
      out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
      buf.clear();
    }
  }
  fnv.Bytes(buf.data(), buf.size());
  s.muonLogBytes += buf.size();
  out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  out.close();
  if (out.fail()) return false;
  std::filesystem::rename(partPath, finalPath, ec);
  if (ec) return false;
  s.muonLogFnv = fnv.Value();
  return true;
}

Oe1Summary FinalizeListMode(const oe1::ListModeConfig& lm, const G4Run* run,
                            const std::string& stem, bool histWriteOk)
{
  Oe1Summary s;
  RunData& data = RunData::Instance();
  std::vector<oe1::MuonRecord>& recs = data.Muons();
  s.expected = lm.list->Size();
  s.rows = recs.size();
  s.idsCompleteUnique = oe1::SortAndCheckIds(recs, s.expected);
  s.eventsEqual = run->GetNumberOfEvent() >= 0 &&
                  static_cast<std::size_t>(run->GetNumberOfEvent()) == s.expected;
  s.generatedEqual = data.Generated() == s.expected;

  std::size_t badBins = 0;
  const std::vector<uint64_t> rebuilt =
      oe1::RebuildHistogram(recs, viewpar::kNEl, viewpar::kNAz, badBins);
  s.histEqual = badBins == 0 && rebuilt == data.Histogram();

  uint64_t nDet = 0;
  for (const auto& r : recs) {  // id order = 1-thread RecordHit summation order
    s.statusCounts[static_cast<std::size_t>(r.status)] += 1;
    if (r.status == oe1::MuonStatus::kDet) {
      ++nDet;
      s.ekinSumRebuiltGeV += r.ekinPlaneGeV;
    }
    if (r.inRint == 1) ++s.nInRint;
    if (r.deep) ++s.nCrossed;
    if (r.lz > 0.0) ++s.nZMuons;
    if (r.nGrazeZ > 0) ++s.nGrazeMuons;
    s.nStepsZ += static_cast<uint64_t>(r.nStepsZ);
    s.nStepsZBothOut += static_cast<uint64_t>(r.nStepsZBothOut);
    s.nEntriesZ += static_cast<uint64_t>(r.nEntriesZ);
    s.nGrazeSteps += static_cast<uint64_t>(r.nGrazeZ);
    s.maxLatMm = std::max(s.maxLatMm, r.maxLatMm);
    s.maxLatZMm = std::max(s.maxLatZMm, r.maxLatZMm);
    s.maxSumLatZMm = std::max(s.maxSumLatZMm, r.sumLatZMm);
    s.nStepsNearZ += static_cast<uint64_t>(r.nStepsNearZ);
    if (r.nStepsNearZ > 0) ++s.nMuonsNearZ;
  }
  s.detectedEqual = nDet == data.Detected();
  s.ekinSumRunGeV = data.EkinSumGeV();
  s.ekinSumEqual = s.ekinSumRebuiltGeV == s.ekinSumRunGeV;  // diagnostic only

  s.histWriteOk = histWriteOk;
  const std::string logPath = stem + "_muones.csv";
  s.muonLogFile = FileNameOf(logPath);
  s.muonLogOk = WriteMuonLog(lm, recs, logPath, s);

  if (!s.idsCompleteUnique) data.SetError("ids_incomplete_or_duplicate");
  if (!s.eventsEqual) data.SetError("events_transported_mismatch");
  if (!s.generatedEqual) data.SetError("generated_count_mismatch");
  if (!s.histEqual) data.SetError("hist_reconstruction_mismatch");
  if (!s.detectedEqual) data.SetError("detected_count_mismatch");
  if (!s.histWriteOk) data.SetError("hist_write_failed");
  if (!s.muonLogOk) data.SetError("muon_log_write_failed");
  if (s.statusCounts[static_cast<std::size_t>(oe1::MuonStatus::kAborted)] > 0)
    data.SetError("aborted_events");
  if (data.WorkerIds().size() != 1) data.SetError("worker_threads_not_one");
  if (data.EngineNames() != std::set<std::string>{"MixMaxRng"})
    data.SetError("worker_engine_not_mixmax");

  // The run manager (one worker of the task-based one, as in the pilot; plan S7)
  // and --variant against the geometry that was built (plan S2). main() runs the
  // same two checks right after runManager->Initialize() and never starts the
  // transport when one fails (plan S3); repeating them here is the second line
  // of defence, and records the outcome in the meta.
  s.signature = oe1::BuiltGeometrySignature();
  const oe1::PreRunReport pre = oe1::CheckBeforeTransport(lm, s.signature);
  s.variantCheck = pre.variantCheck;
  s.variantWhy = pre.variantWhy;
  for (std::size_t i = 0; i < pre.errors.size(); ++i) {
    G4cerr << "OE1: " << pre.messages[i] << G4endl;
    data.SetError(pre.errors[i]);
  }
  return s;
}

nlohmann::json Oe1Meta(const oe1::ListModeConfig& lm, const Oe1Summary& s)
{
  using nlohmann::json;
  const RunData& data = RunData::Instance();
  const oe1::ListSidecar& sc = lm.list->Sidecar();
  char tag[16];
  std::snprintf(tag, sizeof tag, "0x%08X", static_cast<unsigned>(oe1::kTagTransport));
  const json sourceSeed = sc.sourceSeed >= 0 ? json(sc.sourceSeed) : json(nullptr);

  json m;
  m["format"] = "oe1-etapa1/1";
  m["mode"] = "list";
  m["variant"] = lm.variant;
  m["case_code"] = lm.caseCode;
  m["replica"] = lm.replica;
  // Provenance of the executable and of the fixed inputs (plan S2), file names
  // only. binary_sha256 differs between the A and B binaries by construction.
  m["binary_sha256"] = lm.binarySha256;
  json geometryJson;
  geometryJson["file"] = lm.geometryFile;
  geometryJson["sha256"] = lm.geometrySha256;
  json tableHashes = json::object();
  for (const auto& t : lm.tableSha256) tableHashes[t.first] = t.second;
  json inputs;
  inputs["geometry_json"] = geometryJson;
  inputs["tables"] = tableHashes;
  m["inputs"] = inputs;
  m["list"] = {{"file", lm.listFile},
               {"sha256", lm.list->FileSha256()},
               {"sha256_verified", true},
               {"canon_fnv1a64", oe1::Hex16(lm.list->CanonFnv())},
               {"n", lm.list->Size()},
               {"sidecar_kind", sc.kind},
               {"sidecar_case_code", sc.caseCode},
               {"sidecar_replica", sc.replica},
               {"source_seed", sourceSeed}};
  m["roi_mask"] = {{"file", lm.maskFile},
                   {"sha256", lm.mask->FileSha256()},
                   {"sha256_verified", true},
                   {"content_fnv1a64", oe1::Hex16(lm.mask->ContentFnv())},
                   {"ones", lm.mask->Ones()},
                   {"rows", {lm.mask->RowMin(), lm.mask->RowMax()}},
                   {"cols", {lm.mask->ColMin(), lm.mask->ColMax()}},
                   {"indexing", "line = iEl (hist.csv row), character = iAz (hist.csv column)"},
                   {"in_rint", "1/0 = detected inside/outside the mask; -1 = not detected"}};
  json transportKey = json::array();
  transportKey.push_back("event_id");
  transportKey.push_back(lm.transportSeed);
  transportKey.push_back(lm.caseCode);
  transportKey.push_back(oe1::kTagTransport);
  m["seeds"] = {
      {"master_cli", lm.masterSeedCli},
      {"master_applied", lm.masterSeedApplied},
      {"master_rule", "CLHEP::HepRandom::setTheSeed(seed % 2147483647) before the run "
                      "manager (unchanged)"},
      {"transport_seed", lm.transportSeed},
      {"transport_key", transportKey},
      {"transport_key_text",
       std::string("G4Random::setTheSeeds({event_id, s_tr, case_code, TAG_TR=") + tag +
           "}, 4) in EventAction::BeginOfEventAction"},
      {"source_seed", sourceSeed},
      {"source_key_text", "{s_src, case_code, 0, TAG_SRC=0x4F455331} (list generation)"}};
  json workers = json::array();
  for (const int id : data.WorkerIds()) workers.push_back(id);
  json engines = json::array();
  for (const auto& e : data.EngineNames()) engines.push_back(e);
  m["runtime"] = {{"run_manager_class", lm.runManagerClass},
                  {"threads_cli", lm.threadsCli},
                  {"worker_thread_ids", workers},
                  {"worker_engines", engines},
                  {"master_engine", lm.masterEngine}};
  json region;
  region["applicable"] = lm.regionApplicable;
  region["explicit_none"] = lm.regionExplicitNone;  // --region-pair none
  region["geometry_anomalies"] = lm.geometryAnomalies;
  if (lm.regionApplicable) {
    const oe1::CylinderPrim& c = lm.region.Cylinder();
    const oe1::EllipsoidPrim& e = lm.region.Ellipsoid();
    oe1::Vec3 lo, hi;
    lm.region.ZBox(lo, hi);
    region["pair"] = {lm.regionFirst, lm.regionSecond};
    region["kinds"] = {lm.regionFirstKind, lm.regionSecondKind};
    region["cylinder_m"] = {{"cx", c.cx}, {"cy", c.cy}, {"r", c.r}, {"z0", c.z0}, {"z1", c.z1}};
    region["ellipsoid_m"] = {{"cx", e.cx}, {"cy", e.cy}, {"cz", e.cz},
                             {"ax", e.ax}, {"ay", e.ay}, {"az", e.az}};
    region["zbox_m"] = {{"lo", {lo.x, lo.y, lo.z}}, {"hi", {hi.x, hi.y, hi.z}}};
    region["eps_m"] = lm.region.Eps();
    region["near_margin_m"] = lm.region.NearMargin();
    region["frame"] = "world, metres (cone axis at the origin, ground at z = 0)";
    region["materials"] = {{"R", lm.rockMaterial}, {"A0", lm.firstMaterial},
                           {"A1", lm.secondMaterial}, {"AIR", "G4_AIR"}};
    region["definitions"] =
        "per primary step, exact interval of the pre->post CHORD with cylinder intersect "
        "ellipsoid. The chord is not the path followed: multiple scattering displaces the "
        "post-step point sideways, so lengths in Z are chord lengths and their error is "
        "reported, not zero: max_lat_mm = largest lateral displacement over all primary "
        "steps (diagnostic); max_lat_z_mm / sum_lat_z_mm = largest / summed lateral "
        "displacement over the steps near Z (chord meets both primitives grown by "
        "near_margin_m); sum_lat_z_mm is a scale of the error of the chord lengths in Z, "
        "not a bound. crossed = some step longer than 2 eps inside Z with positive "
        "length inside Z shrunk by eps (so crossed = 1 implies n_steps_z >= 1); graze = "
        "touches Z grown by eps away from both step ends; lz_true = chord length x "
        "StepLength/|chord|; length split by the pre-step material; e_in/e_out "
        "interpolated along the chord with the step's [E_post, E_pre] bracket";
  }
  m["region"] = region;
  json status;
  for (int i = 0; i < oe1::kNumMuonStatus; ++i) {
    status[oe1::StatusName(static_cast<oe1::MuonStatus>(i))] =
        s.statusCounts[static_cast<std::size_t>(i)];
  }
  // Fields that were not computed (--muon-log min, or no --region-pair for the
  // crossing counts) are null, never 0: a zero would pass a check vacuously.
  const bool full = lm.logLevel == oe1::MuonLogLevel::kFull;
  const json na = nullptr;
  const bool zOn = full && lm.regionApplicable;
  m["counts"] = {{"status", status},
                 {"in_rint", s.nInRint},
                 {"crossed", zOn ? json(s.nCrossed) : na},
                 {"z_muons", zOn ? json(s.nZMuons) : na},
                 {"graze_muons", zOn ? json(s.nGrazeMuons) : na},
                 {"steps_z", zOn ? json(s.nStepsZ) : na},
                 {"steps_z_both_out", zOn ? json(s.nStepsZBothOut) : na},
                 {"entries_z", zOn ? json(s.nEntriesZ) : na},
                 {"graze_steps", zOn ? json(s.nGrazeSteps) : na},
                 {"max_lat_mm", full ? json(s.maxLatMm) : na},
                 // Lateral displacement of the steps near Z (plan correction C1).
                 {"max_lat_z_mm", zOn ? json(s.maxLatZMm) : na},
                 {"max_sum_lat_z_mm", zOn ? json(s.maxSumLatZMm) : na},
                 {"near_z_steps", zOn ? json(s.nStepsNearZ) : na},
                 {"near_z_muons", zOn ? json(s.nMuonsNearZ) : na}};
  std::string joined;
  for (const auto& e : s.signature) joined += e + "\n";
  m["geometry_signature"] = {{"volumes", s.signature},
                             {"fnv1a64", oe1::Hex16(oe1::Fnv1a64Of(joined))},
                             {"variant_check_note", s.variantWhy}};
  // true / false = checked; null = not checked (no region pair, or a region
  // whose primitives do not overlap, where variant B cannot be told from A).
  json variantOk = nullptr;
  if (s.variantCheck == oe1::VariantCheck::kOk) variantOk = true;
  if (s.variantCheck == oe1::VariantCheck::kMismatch) variantOk = false;
  m["checks"] = {{"ids_complete_unique", s.idsCompleteUnique},
                 {"hist_reconstructed_equal", s.histEqual},
                 {"detected_equal", s.detectedEqual},
                 {"events_equal", s.eventsEqual},
                 {"generated_equal", s.generatedEqual},
                 {"ekin_sum_equal", s.ekinSumEqual},
                 {"writes_ok", s.histWriteOk && s.muonLogOk},
                 {"variant_matches_geometry", variantOk}};
  m["muon_log"] = {{"file", s.muonLogFile},
                   {"level", lm.logLevel == oe1::MuonLogLevel::kFull ? "full" : "min"},
                   {"rows", s.rows},
                   {"columns", oe1::kMuonCsvColumns},
                   {"bytes", s.muonLogBytes},
                   {"fnv1a64", oe1::Hex16(s.muonLogFnv)}};
  json errors = json::array();
  for (const auto& e : data.Errors()) errors.push_back(e);
  m["errors"] = errors;
  return m;
}

std::string Oe1SummaryLine(const oe1::ListModeConfig& lm, const Oe1Summary& s)
{
  std::ostringstream o;
  o << "OE1-ETAPA1 geometry=" << lm.geometryId << " view=" << lm.view
    << " variant=" << lm.variant << " case_code=" << lm.caseCode
    << " replica=" << lm.replica
    << " muon_log=" << (lm.logLevel == oe1::MuonLogLevel::kFull ? "full" : "min")
    << " n=" << s.expected << " rows=" << s.rows;
  for (int i = 0; i < oe1::kNumMuonStatus; ++i) {
    o << " " << oe1::StatusName(static_cast<oe1::MuonStatus>(i)) << "="
      << s.statusCounts[static_cast<std::size_t>(i)];
  }
  const bool zOn = lm.logLevel == oe1::MuonLogLevel::kFull && lm.regionApplicable;
  o << " in_rint=" << s.nInRint;
  if (zOn) o << " crossed=" << s.nCrossed << " graze_muons=" << s.nGrazeMuons;
  else o << " crossed=NA graze_muons=NA";
  o << " ids_ok=" << s.idsCompleteUnique
    << " hist_ok=" << s.histEqual << " det_ok=" << s.detectedEqual
    << " writes_ok=" << (s.histWriteOk && s.muonLogOk)
    << " ekin_sum_equal=" << s.ekinSumEqual << " errors=";
  const auto& errors = RunData::Instance().Errors();
  if (errors.empty()) o << "none";
  bool first = true;
  for (const auto& e : errors) {
    o << (first ? "" : ",") << e;
    first = false;
  }
  o << " status=" << (errors.empty() ? "OK" : "FAIL");
  return o.str();
}

}  // namespace

std::vector<std::string> oe1::BuiltGeometrySignature()
{
  std::vector<std::string> sig;
  const G4PhysicalVolumeStore* store = G4PhysicalVolumeStore::GetInstance();
  for (const G4VPhysicalVolume* pv : *store) {
    if (pv == nullptr) continue;
    const G4LogicalVolume* lv = pv->GetLogicalVolume();
    const std::string solid = (lv != nullptr && lv->GetSolid() != nullptr)
                                  ? std::string(lv->GetSolid()->GetEntityType())
                                  : std::string("none");
    const std::string mat = (lv != nullptr && lv->GetMaterial() != nullptr)
                                ? std::string(lv->GetMaterial()->GetName())
                                : std::string("none");
    sig.push_back(std::string(pv->GetName()) + ":" + solid + ":" + mat);
  }
  std::sort(sig.begin(), sig.end());
  return sig;
}

RunAction::RunAction(const GeometryConfig& cfg, int viewIndex, uint64_t seed,
                     std::string outDir, int nThreads, double exposureDays)
    : fCfg(cfg), fViewIndex(viewIndex), fSeed(seed),
      fOutDir(std::move(outDir)), fNThreads(nThreads),
      fExposureDays(exposureDays)
{}

void RunAction::BeginOfRunAction(const G4Run*)
{
  if (IsMaster()) fWallStart = nowSeconds();
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  if (!IsMaster()) return;

  const double wall = nowSeconds() - fWallStart;
  const auto& data = RunData::Instance();
  const std::string stem =
      fOutDir + "/" + fCfg.geometry_id + "_view" + std::to_string(fViewIndex);

  // Histogram CSV: one row per elevation bin (matches surrogate row order).
  bool histWriteOk = true;  // OE1 list mode: checked below
  {
    std::ofstream out(stem + "_hist.csv");
    const auto& h = data.Histogram();
    for (int iEl = 0; iEl < viewpar::kNEl; ++iEl) {
      for (int iAz = 0; iAz < viewpar::kNAz; ++iAz) {
        out << h[static_cast<size_t>(iEl) * viewpar::kNAz + iAz];
        out << (iAz + 1 < viewpar::kNAz ? ',' : '\n');
      }
    }
    if (fLm != nullptr) {  // OE1 list mode: stream state is part of the checks
      out.close();
      histWriteOk = !out.fail();
    }
  }

  // OE1 list mode: per-muon log, reconstruction and id checks.
  Oe1Summary oe1Summary;
  if (fLm != nullptr) oe1Summary = FinalizeListMode(*fLm, run, stem, histWriteOk);

  // Metadata sidecar (schema aligned with the surrogate .meta.json).
  {
    nlohmann::json meta;
    meta["engine"] = std::string("geant4-") + G4Version;
    meta["geometry_id"] = fCfg.geometry_id;
    meta["view_index"] = fViewIndex;
    meta["seed"] = fSeed;
    meta["threads"] = fNThreads;
    meta["events_transported"] = run->GetNumberOfEvent();
    meta["ecomug_draws_total"] = data.Generated();
    meta["detected_in_bins"] = data.Detected();
    meta["mean_ekin_detected_gev"] = data.MeanEkinGeV();
    meta["wall_seconds"] = wall;
    meta["seconds_per_event"] =
        run->GetNumberOfEvent() > 0 ? wall / run->GetNumberOfEvent() : 0.0;
    if (fLm != nullptr) {
      // OE1 list mode (plan W3): the CPU time and peak memory of THIS process,
      // because sacct cannot separate the 13 processes that xargs starts inside
      // one job. getrusage(RUSAGE_SELF) covers every thread of the process from
      // its start (geometry construction included), unlike wall_seconds, which
      // starts at BeginOfRunAction. Like wall_seconds these keys differ between
      // two identical runs; the A/A and A/B comparisons of the analysis ignore
      // exactly these top-level keys (oe1_stage1_common.TIMING_META_KEYS, an
      // explicit list, no substring rule). They are never inside oe1_etapa1.
      struct rusage usage;
      if (getrusage(RUSAGE_SELF, &usage) == 0) {
        meta["rusage_user_cpu_seconds"] =
            static_cast<double>(usage.ru_utime.tv_sec) + 1.0e-6 * static_cast<double>(usage.ru_utime.tv_usec);
        meta["rusage_system_cpu_seconds"] =
            static_cast<double>(usage.ru_stime.tv_sec) + 1.0e-6 * static_cast<double>(usage.ru_stime.tv_usec);
        meta["rusage_max_rss_kb"] = static_cast<long long>(usage.ru_maxrss);  // kilobytes on Linux
      }
    }
    meta["grid"] = {{"azimuth_convention", azgrid::kVersion},
                    {"az_edges_deg", {-viewpar::kAzHalfDeg, viewpar::kAzHalfDeg}},
                    {"az_step_deg", azgrid::kStepDeg},
                    {"az_first_center_deg", azgrid::Center(0)},
                    {"elevation_convention", "legacy-centers-2-47"},
                    {"n_az", viewpar::kNAz},
                    {"n_el", viewpar::kNEl},
                    {"az_half_deg", viewpar::kAzHalfDeg},
                    {"el_min_deg", viewpar::kElMinDeg},
                    {"el_max_deg", viewpar::kElMaxDeg}};
    meta["detector"] = {{"side_m", viewpar::kDetSideM},
                        {"distance_m", viewpar::kDetDistanceM},
                        {"height_m", viewpar::kDetHeightM}};
    meta["exposure_days"] = fExposureDays;
    meta["sampling"] = "table-driven: protocol lambda_open row weights "
                       "(Guan-normalized, surrogate convention) + real-EcoMug "
                       "momentum spectra; backtracked line of sight, uniform "
                       "panel impact";
    meta["physics_note"] = "secondaries killed at creation; primary-only "
                           "transport for transmission counting";
    if (fLm != nullptr) meta["oe1_etapa1"] = Oe1Meta(*fLm, oe1Summary);

    std::ofstream out(stem + ".meta.json");
    out << meta.dump(1) << "\n";
    if (fLm != nullptr) {
      out.close();
      if (out.fail()) RunData::Instance().SetError("meta_write_failed");
    }
  }

  G4cout << "RUN DONE " << fCfg.geometry_id << " view " << fViewIndex
         << ": events=" << run->GetNumberOfEvent()
         << " detected=" << data.Detected()
         << " wall_s=" << wall << G4endl;
  if (fLm != nullptr) {
    RunData::Instance().MarkFinalized();  // main() returns 4 if this never happens
    G4cout << Oe1SummaryLine(*fLm, oe1Summary) << G4endl;
  }
}
