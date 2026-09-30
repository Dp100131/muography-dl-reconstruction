#ifndef OE1_LIST_MODE_HH
#define OE1_LIST_MODE_HH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "GeometryConfig.hh"
#include "PrimaryList.hh"
#include "RegionCrossing.hh"
#include "RoiMask.hh"

// OE1 stage 1 (P2): configuration of a list-mode run (--primaries), built
// once in main() and shared read-only by the actions. No Geant4 dependency.
namespace oe1 {

// Fourth word of the MixMax seed keys. Geant4's own per-event reseed uses
// (0, 0, s2, s1), so a non-zero tag keeps these streams disjoint from it.
constexpr uint32_t kTagTransport = 0x4F455431U;  // transport, per muon
constexpr uint32_t kTagSource = 0x4F455331U;     // source list generation

constexpr double kRegionEpsM = 1.0e-6;  // tangency band (deep vs graze)

// Run manager class a list-mode run must report (one worker thread of the
// task-based run manager, as in the pilot). Anything else is an error flag.
constexpr const char* kExpectedRunManager = "G4TaskRunManager";

enum class MuonLogLevel { kFull, kMin };

struct ListModeConfig {
  // Inputs (owned by main; outlive the run manager).
  const PrimaryList* list = nullptr;
  const RoiMask* mask = nullptr;
  std::string listFile;   // file name only (outputs carry no paths)
  std::string maskFile;

  // Identity of the run.
  std::string geometryId;
  int view = 0;
  std::string variant;  // "A" or "B"
  uint32_t caseCode = 0;
  int replica = 0;

  // Seeds: per-muon transport key {event_id, transportSeed, caseCode,
  // kTagTransport}; the master keeps setTheSeed(masterSeedCli % 2147483647).
  uint64_t transportSeed = 0;
  uint64_t masterSeedCli = 0;
  long masterSeedApplied = 0;

  // Runtime provenance, filled in main() once the run manager exists.
  std::string runManagerClass;
  std::string masterEngine;
  int threadsCli = 1;

  // Input provenance recorded in the meta (RecordProvenance, Oe1Cli.cc): the
  // SHA-256 of this executable (/proc/self/exe), of the geometry JSON and of
  // the three generator table files, so a result can be tied to the exact
  // binary and inputs that produced it. File names only, never paths.
  std::string binarySha256;
  std::string geometryFile;
  std::string geometrySha256;
  std::vector<std::pair<std::string, std::string>> tableSha256;  // (file name, sha256)

  MuonLogLevel logLevel = MuonLogLevel::kFull;

  // 3D region Z from --region-pair (not applicable when absent or "none").
  bool regionApplicable = false;
  bool regionExplicitNone = false;  // --region-pair none: no region on purpose
  int geometryAnomalies = 0;  // anomalies in the catalog geometry
  int regionFirst = -1;   // conduit
  int regionSecond = -1;  // dense_body or cavity
  std::string regionFirstKind;
  std::string regionSecondKind;
  RegionZ region;
  // Material names as built by DetectorConstruction (same in A and B).
  std::string rockMaterial = "standard_rock";
  std::string firstMaterial;   // "anomaly_<first>_mat"
  std::string secondMaterial;  // "anomaly_<second>_mat"
};

// Builds Z = cylinder(anomaly i) AND ellipsoid/sphere(anomaly j) from the
// catalog primitives (world frame, metres). Returns false with a message if
// the indices or kinds are invalid.
inline bool BuildRegionFromConfig(const GeometryConfig& cfg, int i, int j, double eps,
                                  RegionZ& out, std::string& err)
{
  const int n = static_cast<int>(cfg.anomalies.size());
  if (i < 0 || j < 0 || i >= n || j >= n || i == j) {
    err = "--region-pair " + std::to_string(i) + "," + std::to_string(j) +
          " does not name two distinct anomalies (geometry has " + std::to_string(n) + ")";
    return false;
  }
  const AnomalyConfig& c = cfg.anomalies[static_cast<std::size_t>(i)];
  const AnomalyConfig& e = cfg.anomalies[static_cast<std::size_t>(j)];
  if (c.kind != "conduit") {
    err = "--region-pair: anomaly " + std::to_string(i) + " is '" + c.kind +
          "', expected 'conduit'";
    return false;
  }
  EllipsoidPrim ell;
  if (e.kind == "dense_body") {
    ell = {e.cx, e.cy, e.cz, e.ax, e.ay, e.az};
  }
  else if (e.kind == "cavity") {
    ell = {e.cx, e.cy, e.cz, e.radius, e.radius, e.radius};
  }
  else {
    err = "--region-pair: anomaly " + std::to_string(j) + " is '" + e.kind +
          "', expected 'dense_body' or 'cavity'";
    return false;
  }
  const CylinderPrim cyl{c.cx, c.cy, c.radius, c.z0, c.z1};
  out = RegionZ(cyl, ell, eps);
  return true;
}

// ---------------------------------------------------------------------------
// --variant against the geometry that was actually built (plan S2).
//
// The master RunAction records a geometry signature: one "name:SolidType:Material"
// entry per physical volume (SolidType = G4VSolid::GetEntityType()). DetectorConstruction
// builds each anomaly as its plain primitive in variant A (conduit G4Tubs, dense body
// G4Ellipsoid, cavity G4Orb) and, in variant B, as (anomaly intersect edifice) minus every
// LATER anomaly that overlaps it. For a region pair (conduit i, ellipsoid j) whose
// primitives overlap, B therefore builds the EARLIER of the two as a G4SubtractionSolid
// (geo-train-06: anomaly_0 is a G4Tubs in A and a G4SubtractionSolid in B).
enum class VariantCheck { kNotChecked, kOk, kMismatch };

inline const char* PrimitiveSolidType(const std::string& kind)
{
  if (kind == "conduit") return "G4Tubs";
  if (kind == "dense_body") return "G4Ellipsoid";
  if (kind == "cavity") return "G4Orb";
  return "";
}

// Solid type of "anomaly_<index>" in the signature entries; empty if absent.
inline std::string SignatureSolid(const std::vector<std::string>& volumes, int index)
{
  const std::string prefix = "anomaly_" + std::to_string(index) + ":";
  for (const std::string& entry : volumes) {
    if (entry.compare(0, prefix.size(), prefix) != 0) continue;
    const std::size_t end = entry.find(':', prefix.size());
    return entry.substr(prefix.size(),
                        end == std::string::npos ? std::string::npos : end - prefix.size());
  }
  return std::string();
}

// `zBoxNonEmpty` = the bounding boxes of the two primitives overlap (variant B
// only subtracts overlapping anomalies; without an overlap it cannot be told
// from A and is reported as not checked).
inline VariantCheck CheckVariantAgainstSignature(const std::string& variant, int first, int second,
                                                 const std::string& firstKind,
                                                 const std::string& secondKind, bool zBoxNonEmpty,
                                                 const std::vector<std::string>& volumes,
                                                 std::string& why)
{
  why.clear();
  if (variant == "A") {
    const int idx[2] = {first, second};
    const std::string* kind[2] = {&firstKind, &secondKind};
    for (int k = 0; k < 2; ++k) {
      const std::string want = PrimitiveSolidType(*kind[k]);
      const std::string got = SignatureSolid(volumes, idx[k]);
      if (got != want) {
        why = "--variant A expects anomaly_" + std::to_string(idx[k]) + " to be a " + want +
              " (plain primitive) but the geometry built '" + (got.empty() ? "no such volume" : got) +
              "'";
        return VariantCheck::kMismatch;
      }
    }
    return VariantCheck::kOk;
  }
  if (variant == "B") {
    if (!zBoxNonEmpty) {
      why = "the two primitives do not overlap: variant B cannot be told from A";
      return VariantCheck::kNotChecked;
    }
    const int earlier = std::min(first, second);
    const std::string got = SignatureSolid(volumes, earlier);
    if (got != "G4SubtractionSolid") {
      why = "--variant B expects anomaly_" + std::to_string(earlier) +
            " to be a G4SubtractionSolid but the geometry built '" +
            (got.empty() ? "no such volume" : got) + "'";
      return VariantCheck::kMismatch;
    }
    return VariantCheck::kOk;
  }
  why = "--variant '" + variant + "' is neither A nor B";
  return VariantCheck::kMismatch;
}

// ---------------------------------------------------------------------------
// The run checks that need no transport (plan S3).
//
// The run manager class is known as soon as the manager exists and the geometry
// signature as soon as runManager->Initialize() has built the geometry, so
// main() evaluates both right after Initialize() and, when one fails, exits
// with kExitRunCheckFailed BEFORE BeamOn: no transport runs and no evidence
// file is written for a run that cannot be valid. RunAction evaluates the very
// same function again at the end of the run (second line of defence: it puts
// the codes into oe1_etapa1.errors and the process still exits 4).
//
// Exit codes of a list-mode run (the full order is in the header of
// muography_sim.cc): 2 strict command line and --region-pair rule, 1 unusable
// input, then 4 for a failure of this function after runManager->Initialize()
// (nothing ran), then BeamOn, then 4 for a run check that failed at the end of
// the run (the two checks of this function once more among them), else 0.
constexpr int kExitRunCheckFailed = 4;
constexpr const char* kErrRunManagerNotTask = "run_manager_not_task";
constexpr const char* kErrVariantMismatch = "variant_geometry_mismatch";

struct PreRunReport {
  std::vector<std::string> errors;    // error codes, as in RunData and oe1_etapa1.errors
  std::vector<std::string> messages;  // one readable line per error
  VariantCheck variantCheck = VariantCheck::kNotChecked;
  std::string variantWhy;             // also the meta's geometry_signature.variant_check_note
  bool Ok() const { return errors.empty(); }
};

// `lm.runManagerClass` must already hold the class of the run manager in use;
// `signature` is the geometry signature of the geometry that was built (one
// "name:SolidType:Material" entry per physical volume, see above). The variant
// is only checked when a region pair names the anomalies whose solids tell A
// from B (lm.regionApplicable).
inline PreRunReport CheckBeforeTransport(const ListModeConfig& lm,
                                         const std::vector<std::string>& signature)
{
  PreRunReport r;
  if (lm.runManagerClass != kExpectedRunManager) {
    r.errors.emplace_back(kErrRunManagerNotTask);
    r.messages.push_back("the run manager is '" + lm.runManagerClass + "', expected '" +
                         std::string(kExpectedRunManager) + "'");
  }
  if (lm.regionApplicable) {
    r.variantCheck = CheckVariantAgainstSignature(lm.variant, lm.regionFirst, lm.regionSecond,
                                                  lm.regionFirstKind, lm.regionSecondKind,
                                                  lm.region.ZBoxNonEmpty(), signature, r.variantWhy);
    if (r.variantCheck == VariantCheck::kMismatch) {
      r.errors.emplace_back(kErrVariantMismatch);
      r.messages.push_back(r.variantWhy);
    }
  }
  return r;
}

}  // namespace oe1

#endif
