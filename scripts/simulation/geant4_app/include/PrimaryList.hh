#ifndef OE1_PRIMARY_LIST_HH
#define OE1_PRIMARY_LIST_HH

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

// OE1 stage 1 (P2): frozen list of initial muons (PROT:22-26). No Geant4
// dependency.
//
// List file <case>_v<view>_r<k>_s<s_src>.primarios.csv (LF only):
//   id,x_m,y_m,z_m,dx,dy,dz,ekin_gev,pdg,charge_e
//   id       int64, row k has id k (0..N-1) = Geant4 event ID
//   x_m..z_m start point, world frame (cone axis at origin, ground z = 0), m
//   dx..dz   unit travel direction
//   ekin_gev kinetic energy, GeV
//   pdg      13 (mu-) or -13 (mu+); charge_e -1 / +1 (redundant, checked)
// Doubles are written with std::to_chars (shortest round-trip) and read with
// std::from_chars.
//
// Canonical content hash (canon_fnv1a64): FNV-1a-64 over every record, in
// file order, of the little-endian bytes int64 id, float64 x_m, y_m, z_m,
// dx, dy, dz, ekin_gev, int32 pdg (Python: struct.pack('<q7di', ...)).
//
// Sidecar <list>.meta.json: see WriteSidecar() for the fields; list mode
// requires format, geometry_id, view_index, base_radius_m, case_code,
// replica, n and canon_fnv1a64.
namespace oe1 {

constexpr const char* kPrimaryListFormat = "oe1-primarios/1";
constexpr const char* kPrimaryListHeader = "id,x_m,y_m,z_m,dx,dy,dz,ekin_gev,pdg,charge_e";

struct PrimaryRecord {
  int64_t id = 0;
  double x_m = 0.0, y_m = 0.0, z_m = 0.0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double ekin_gev = 0.0;
  int32_t pdg = 0;
  int32_t charge_e = 0;
};

// Defect counters shared by --check-primaries and the list-mode loader.
struct ListDefects {
  bool readable = true;
  bool lfOnly = true;          // no CR byte anywhere
  bool headerOk = true;
  bool endsWithNewline = true;
  std::size_t badRows = 0;     // wrong field count or unparsable field
  int64_t idMin = 0, idMax = -1;
  bool idsContiguous = true;   // row k has id k
  std::size_t dupIds = 0;      // ids seen more than once
  std::size_t nonFinite = 0;
  std::size_t normFail = 0;    // | |d| - 1 | > 1e-12
  double maxNormDev = 0.0;
  std::size_t dzNonNeg = 0;    // dz >= 0 (must travel downwards)
  std::size_t ekinNonPos = 0;
  std::size_t pdgBad = 0;      // pdg not +-13
  std::size_t chargeMismatch = 0;
  std::size_t dupStates = 0;   // identical (x,y,z,dx,dy,dz,ekin,pdg) rows
  std::size_t nMuPlus = 0, nMuMinus = 0;

  bool Clean() const;
};

// Parsed sidecar fields (missing optional values keep their defaults).
struct ListSidecar {
  std::string format;
  std::string geometryId;
  int64_t view = -1;
  double baseRadiusM = 0.0;
  bool haveBaseRadius = false;
  int64_t caseCode = -1;
  int64_t replica = -1;
  int64_t sourceSeed = -1;      // -1 when absent (e.g. synthetic lists)
  int64_t n = -1;
  uint64_t canonFnv = 0;
  bool haveCanonFnv = false;
  std::string fileSha256;       // optional
  std::string kind;             // optional free label (e.g. "sintetico")
};

// Provenance written into a new sidecar by --write-primaries.
struct ListProvenance {
  std::string geometryId;
  int view = 0;
  double baseRadiusM = 0.0;
  double startRadiusM = 0.0;
  uint32_t caseCode = 0;
  int replica = 0;
  uint64_t sourceSeed = 0;
  uint32_t sourceTag = 0;
  int drawsPerPrimary = 7;
  uint64_t canonFnv = 0;
  std::string fileSha256;
  std::string binarySha256;
};

uint64_t CanonicalFnv(const std::vector<PrimaryRecord>& recs);

// One CSV data row, LF-terminated.
std::string FormatPrimaryRow(const PrimaryRecord& r);

// Parses a list file. Returns false only if the file cannot be read; all
// content problems are reported through `defects`. Rows that parse are
// appended to `out` even when other rows are defective.
bool ReadPrimaryFile(const std::string& path, std::vector<PrimaryRecord>& out,
                     ListDefects& defects, std::string& fileSha256);

// Parses list text already in memory (used by ReadPrimaryFile and tests).
void ParsePrimaryText(const std::string& text, std::vector<PrimaryRecord>& out,
                      ListDefects& defects);

// Id and value checks on parsed records (accumulated into `defects`).
void ValidatePrimaryRecords(const std::vector<PrimaryRecord>& recs, ListDefects& defects);

std::string SidecarPath(const std::string& listPath);
bool ReadSidecar(const std::string& path, ListSidecar& sc, std::string& err);

// Writers refuse to overwrite an existing file (exclusive create).
bool WritePrimaryCsv(const std::string& path, const std::vector<PrimaryRecord>& recs,
                     std::string& err);
bool WriteSidecar(const std::string& path, const ListProvenance& prov, std::size_t n,
                  std::string& err);

// --check-primaries: one summary line on `out`; returns 0 if the list and
// its sidecar are defect-free, 1 otherwise.
int RunCheckPrimaries(const std::string& path, std::ostream& out, std::ostream& errOut);

// Validated list for list mode.
class PrimaryList
{
 public:
  // Reads the list and its sidecar; every ListDefects counter must be zero,
  // the file SHA-256 must equal expectedSha256, and the sidecar n /
  // canon_fnv1a64 (and file_sha256 if present) must match the content.
  bool Load(const std::string& path, const std::string& expectedSha256, std::string& err);

  std::size_t Size() const { return fRecords.size(); }
  const PrimaryRecord& At(std::size_t id) const { return fRecords[id]; }
  uint64_t CanonFnv() const { return fCanonFnv; }
  const std::string& FileSha256() const { return fFileSha256; }
  const ListSidecar& Sidecar() const { return fSidecar; }
  const ListDefects& Defects() const { return fDefects; }

 private:
  std::vector<PrimaryRecord> fRecords;
  ListDefects fDefects;
  ListSidecar fSidecar;
  uint64_t fCanonFnv = 0;
  std::string fFileSha256;
};

}  // namespace oe1

#endif
