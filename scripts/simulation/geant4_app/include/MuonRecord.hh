#ifndef OE1_MUON_RECORD_HH
#define OE1_MUON_RECORD_HH

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

#include "Oe1Hash.hh"

// OE1 stage 1 (P2): per-primary record of a list-mode run and the
// <stem>_muones.csv row format. No Geant4 dependency.
//
// Every field has a defined default (value initialisation), so two runs
// with identical physics produce byte-identical rows (A/A identity).
namespace oe1 {

// Final state of a primary. DET: entered the scoring plane inside the
// 256x256 grid (counted in hist.csv). PLANE_OUT: entered the plane outside
// the grid (not counted). WORLD_EXIT: last step reached the world boundary.
// DECAY / MU_CAPTURE: ended by a decay process / by the mu- capture-at-rest
// process (which also covers decay in orbit). STOPPED: kinetic energy zero
// without one of those. OTHER: anything else. ABORTED: event aborted.
enum class MuonStatus : int {
  kDet = 0,
  kPlaneOut,
  kWorldExit,
  kDecay,
  kMuCapture,
  kStopped,
  kOther,
  kAborted
};
constexpr int kNumMuonStatus = 8;

inline const char* StatusName(MuonStatus s)
{
  switch (s) {
    case MuonStatus::kDet: return "DET";
    case MuonStatus::kPlaneOut: return "PLANE_OUT";
    case MuonStatus::kWorldExit: return "WORLD_EXIT";
    case MuonStatus::kDecay: return "DECAY";
    case MuonStatus::kMuCapture: return "MU_CAPTURE";
    case MuonStatus::kStopped: return "STOPPED";
    case MuonStatus::kOther: return "OTHER";
    case MuonStatus::kAborted: return "ABORTED";
  }
  return "OTHER";
}

// Material class of a step's pre-step volume: R = edifice rock,
// A0 / A1 = material of the first / second anomaly of --region-pair,
// AIR = G4_AIR, X = anything else. kNone = no step inside Z.
enum class MatCode : int { kNone = 0, kRock, kA0, kA1, kAir, kOther };

inline const char* MatCodeName(MatCode m)
{
  switch (m) {
    case MatCode::kNone: return "";
    case MatCode::kRock: return "R";
    case MatCode::kA0: return "A0";
    case MatCode::kA1: return "A1";
    case MatCode::kAir: return "AIR";
    case MatCode::kOther: return "X";
  }
  return "X";
}

struct MuonRecord {
  // identity and random-stream evidence
  int64_t id = -1;
  uint64_t primFnv = 0;  // FNV-1a-64 of the Geant4 primary at BeginOfEventAction
  uint64_t rng0 = 0;     // engine state hash right after the per-muon reseed
  uint64_t rng1 = 0;     // engine state hash at EndOfEventAction

  // detection (same condition and integers as SteppingAction / RunData)
  int detected = 0;
  int iel = -1;
  int iaz = -1;
  int inRint = -1;  // 1/0 = detected inside/outside R_int; -1 = not detected
  bool planeReached = false;
  double ekinPlaneGeV = 0.0;
  double elPlaneDeg = 0.0;
  double relAzPlaneDeg = 0.0;

  // completion
  MuonStatus status = MuonStatus::kOther;
  std::string endProcess;  // process that defined the last primary step
  int64_t nSteps = 0;
  bool haveEnd = false;  // full log only
  double endXm = 0.0, endYm = 0.0, endZm = 0.0;
  std::string endVolume;  // volume of the last post-step point

  // 3D region Z (full log and region applicable only)
  bool deep = false;
  double lz = 0.0, lzTrue = 0.0;
  double lzRock = 0.0, lzA0 = 0.0, lzA1 = 0.0, lzOther = 0.0;
  double lc = 0.0, le = 0.0;
  double rhoLzGcm2 = 0.0;
  int64_t nStepsZ = 0, nStepsZBothOut = 0, nEntriesZ = 0, nGrazeZ = 0;
  MatCode matIn = MatCode::kNone, matOut = MatCode::kNone;
  bool haveEIn = false, haveEOut = false;
  double eInGeV = 0.0, eInLoGeV = 0.0, eInHiGeV = 0.0;
  double eOutGeV = 0.0, eOutLoGeV = 0.0, eOutHiGeV = 0.0;

  // Lateral displacement (plan correction C1). lat = |(post - pre) x dir_pre|
  // of a primary step is the sideways offset of the post-step point from the
  // line pre + s dir_pre; multiple scattering (WentzelVI) makes it non-zero
  // in nearly every muon, so the pre->post chord is NOT the path that was
  // followed. maxLatMm is a pure diagnostic over all primary steps;
  // maxLatZMm / sumLatZMm are the largest and the summed lat over the steps
  // "near Z" (chord meets both primitives grown by oe1::kNearZMarginM). The
  // sum is the accumulated transverse displacement of those steps: a scale of
  // the uncertainty of the length in Z, NOT a bound. At an incidence angle
  // theta the length error is ~ lat / sin(theta); for a chord skimming a
  // surface it grows like sqrt(2 R lat) (R <= 86.3 m here), so report it as
  // <~ 2 sqrt(2 R_max lat). A chord that penetrates Z by less than
  // max_lat_z_mm may be classified differently from the real path.
  // nStepsNearZ counts those steps (meta only, not a column of the log).
  double maxLatMm = 0.0;
  double maxLatZMm = 0.0;
  double sumLatZMm = 0.0;
  int64_t nStepsNearZ = 0;
};

// Columns that are constant within a run (PROT:48: case, replica, list
// fingerprint in every row).
struct MuonRowConstants {
  uint32_t caseCode = 0;
  std::string geometryId;
  int view = 0;
  std::string variant;
  int replica = 0;
  std::string listSha256;
};

// What the run logged: fullLog = crossing/step accumulation on
// (--muon-log full); regionApplicable = --region-pair given.
struct MuonLogShape {
  bool fullLog = true;
  bool regionApplicable = false;
};

// 49 columns: the 47 of the first delivery followed by max_lat_z_mm and
// sum_lat_z_mm (plan correction C1). The Python readers take the two new names
// as optional columns (oe1_stage1_common.MUON_OPTIONAL_COLUMNS).
inline const char* MuonCsvHeader()
{
  return "caso,geometry_id,view,variant,replica,list_sha256,id,prim_fnv,rng0,rng1,"
         "detected,iel,iaz,in_rint,ekin_plane_gev,el_plane_deg,relaz_plane_deg,"
         "status,end_process,n_steps,end_x_m,end_y_m,end_z_m,end_vol,"
         "crossed,lz_m,lz_true_m,lz_rock_m,lz_a0_m,lz_a1_m,lz_other_m,lc_m,le_m,"
         "rho_lz_gcm2,n_steps_z,n_steps_z_both_out,n_entries_z,n_graze_z,mat_in,"
         "mat_out,e_in_gev,e_in_lo_gev,e_in_hi_gev,e_out_gev,e_out_lo_gev,"
         "e_out_hi_gev,max_lat_mm,max_lat_z_mm,sum_lat_z_mm";
}
constexpr int kMuonCsvColumns = 49;

namespace muon_csv_detail {

inline void Num(std::string& out, double v)
{
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);
  out.append(buf, res.ptr);
}

inline void Int(std::string& out, int64_t v)
{
  char buf[32];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);
  out.append(buf, res.ptr);
}

inline void Sep(std::string& out) { out.push_back(','); }

}  // namespace muon_csv_detail

// Appends one CSV row (with trailing LF). Empty field = not applicable or
// not logged; see MuonCsvHeader() for the column order.
inline void AppendMuonCsvRow(std::string& out, const MuonRowConstants& k,
                             const MuonLogShape& shape, const MuonRecord& r)
{
  using namespace muon_csv_detail;
  Int(out, static_cast<int64_t>(k.caseCode)); Sep(out);
  out += k.geometryId; Sep(out);
  Int(out, k.view); Sep(out);
  out += k.variant; Sep(out);
  Int(out, k.replica); Sep(out);
  out += k.listSha256; Sep(out);
  Int(out, r.id); Sep(out);
  out += Hex16(r.primFnv); Sep(out);
  out += Hex16(r.rng0); Sep(out);
  out += Hex16(r.rng1); Sep(out);

  Int(out, r.detected); Sep(out);
  Int(out, r.iel); Sep(out);
  Int(out, r.iaz); Sep(out);
  Int(out, r.inRint); Sep(out);
  if (r.planeReached) Num(out, r.ekinPlaneGeV);
  Sep(out);
  if (r.planeReached) Num(out, r.elPlaneDeg);
  Sep(out);
  if (r.planeReached) Num(out, r.relAzPlaneDeg);
  Sep(out);

  out += StatusName(r.status); Sep(out);
  out += r.endProcess; Sep(out);
  Int(out, r.nSteps); Sep(out);
  if (shape.fullLog && r.haveEnd) Num(out, r.endXm);
  Sep(out);
  if (shape.fullLog && r.haveEnd) Num(out, r.endYm);
  Sep(out);
  if (shape.fullLog && r.haveEnd) Num(out, r.endZm);
  Sep(out);
  if (shape.fullLog && r.haveEnd) out += r.endVolume;
  Sep(out);

  const bool z = shape.fullLog && shape.regionApplicable;
  if (shape.fullLog) Int(out, shape.regionApplicable ? (r.deep ? 1 : 0) : -1);
  Sep(out);
  const double zvals[] = {r.lz, r.lzTrue, r.lzRock, r.lzA0, r.lzA1,
                          r.lzOther, r.lc, r.le, r.rhoLzGcm2};
  for (const double v : zvals) {
    if (z) Num(out, v);
    Sep(out);
  }
  const int64_t zcounts[] = {r.nStepsZ, r.nStepsZBothOut, r.nEntriesZ, r.nGrazeZ};
  for (const int64_t v : zcounts) {
    if (z) Int(out, v);
    Sep(out);
  }
  if (z) out += MatCodeName(r.matIn);
  Sep(out);
  if (z) out += MatCodeName(r.matOut);
  Sep(out);
  const double ein[] = {r.eInGeV, r.eInLoGeV, r.eInHiGeV};
  for (const double v : ein) {
    if (z && r.haveEIn) Num(out, v);
    Sep(out);
  }
  const double eout[] = {r.eOutGeV, r.eOutLoGeV, r.eOutHiGeV};
  for (const double v : eout) {
    if (z && r.haveEOut) Num(out, v);
    Sep(out);
  }
  if (shape.fullLog && r.nSteps > 0) Num(out, r.maxLatMm);
  Sep(out);
  // Lateral displacement of the steps near Z: like the crossing fields, only
  // with the full log and a region (a muon without a near step logs 0).
  if (z) Num(out, r.maxLatZMm);
  Sep(out);
  if (z) Num(out, r.sumLatZMm);
  out.push_back('\n');
}

// Sorts by id and checks that the ids are exactly 0..n-1, each once.
inline bool SortAndCheckIds(std::vector<MuonRecord>& recs, std::size_t n)
{
  std::sort(recs.begin(), recs.end(),
            [](const MuonRecord& a, const MuonRecord& b) { return a.id < b.id; });
  if (recs.size() != n) return false;
  for (std::size_t i = 0; i < recs.size(); ++i) {
    if (recs[i].id != static_cast<int64_t>(i)) return false;
  }
  return true;
}

// Histogram [iel * nAz + iaz] rebuilt from DET rows only. Rows with an
// out-of-range bin are counted in badBins and skipped.
inline std::vector<uint64_t> RebuildHistogram(const std::vector<MuonRecord>& recs,
                                              int nEl, int nAz, std::size_t& badBins)
{
  std::vector<uint64_t> h(static_cast<std::size_t>(nEl) * static_cast<std::size_t>(nAz), 0);
  badBins = 0;
  for (const auto& r : recs) {
    if (r.status != MuonStatus::kDet) continue;
    if (r.iel < 0 || r.iel >= nEl || r.iaz < 0 || r.iaz >= nAz) {
      ++badBins;
      continue;
    }
    h[static_cast<std::size_t>(r.iel) * static_cast<std::size_t>(nAz) +
      static_cast<std::size_t>(r.iaz)] += 1;
  }
  return h;
}

}  // namespace oe1

#endif
