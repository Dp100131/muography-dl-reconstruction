#include "PrimaryList.hh"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>
#include <system_error>
#include <tuple>

#include "Oe1Hash.hh"
#include "Sha256.hh"
#include "json.hpp"

namespace oe1 {

namespace {

constexpr double kNormTolerance = 1.0e-12;
constexpr std::size_t kListFields = 10;

bool ParseI64(std::string_view s, int64_t& v)
{
  if (s.empty()) return false;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

bool ParseI32(std::string_view s, int32_t& v)
{
  if (s.empty()) return false;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

bool ParseF64(std::string_view s, double& v)
{
  if (s.empty()) return false;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

void AppendDouble(std::string& out, double v)
{
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);
  out.append(buf, res.ptr);
}

void AppendInt(std::string& out, int64_t v)
{
  char buf[32];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);
  out.append(buf, res.ptr);
}

uint64_t Bits(double v)
{
  uint64_t b = 0;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

std::string Basename(const std::string& path)
{
  return std::filesystem::path(path).filename().string();
}

std::string FormatDouble(double v)
{
  std::string s;
  AppendDouble(s, v);
  return s;
}

// Exclusive create ("wx": fails if the file exists) and full write.
bool WriteNewFile(const std::string& path, const std::string& content, std::string& err)
{
  std::FILE* f = std::fopen(path.c_str(), "wx");
  if (f == nullptr) {
    err = "cannot create (exists or not writable): " + path;
    return false;
  }
  const std::size_t written = std::fwrite(content.data(), 1, content.size(), f);
  const bool flushOk = std::fflush(f) == 0;
  const bool closeOk = std::fclose(f) == 0;
  if (written != content.size() || !flushOk || !closeOk) {
    err = "write error on " + path;
    return false;
  }
  return true;
}

std::string DefectSummary(const ListDefects& d, std::size_t n)
{
  std::ostringstream s;
  s << "n=" << n << " lf_only=" << (d.lfOnly ? 1 : 0)
    << " header_ok=" << (d.headerOk ? 1 : 0)
    << " ends_with_newline=" << (d.endsWithNewline ? 1 : 0)
    << " bad_rows=" << d.badRows << " id_min=" << d.idMin << " id_max=" << d.idMax
    << " ids_contiguous=" << (d.idsContiguous ? 1 : 0) << " dup_ids=" << d.dupIds
    << " dup_states=" << d.dupStates << " nonfinite=" << d.nonFinite
    << " norm_fail=" << d.normFail << " max_norm_dev=" << FormatDouble(d.maxNormDev)
    << " dz_nonneg=" << d.dzNonNeg << " ekin_nonpos=" << d.ekinNonPos
    << " pdg_bad=" << d.pdgBad << " charge_mismatch=" << d.chargeMismatch
    << " n_mu_plus=" << d.nMuPlus << " n_mu_minus=" << d.nMuMinus;
  return s.str();
}

}  // namespace

bool ListDefects::Clean() const
{
  return readable && lfOnly && headerOk && endsWithNewline && badRows == 0 &&
         idsContiguous && dupIds == 0 && nonFinite == 0 && normFail == 0 &&
         dzNonNeg == 0 && ekinNonPos == 0 && pdgBad == 0 && chargeMismatch == 0 &&
         dupStates == 0;
}

uint64_t CanonicalFnv(const std::vector<PrimaryRecord>& recs)
{
  Fnv1a64 h;
  for (const auto& r : recs) {
    h.I64(r.id);
    h.F64(r.x_m);
    h.F64(r.y_m);
    h.F64(r.z_m);
    h.F64(r.dx);
    h.F64(r.dy);
    h.F64(r.dz);
    h.F64(r.ekin_gev);
    h.I32(r.pdg);
  }
  return h.Value();
}

std::string FormatPrimaryRow(const PrimaryRecord& r)
{
  std::string s;
  s.reserve(200);
  AppendInt(s, r.id);
  const double v[] = {r.x_m, r.y_m, r.z_m, r.dx, r.dy, r.dz, r.ekin_gev};
  for (const double x : v) {
    s.push_back(',');
    AppendDouble(s, x);
  }
  s.push_back(',');
  AppendInt(s, r.pdg);
  s.push_back(',');
  AppendInt(s, r.charge_e);
  s.push_back('\n');
  return s;
}

void ParsePrimaryText(const std::string& text, std::vector<PrimaryRecord>& out,
                      ListDefects& d)
{
  d.lfOnly = text.find('\r') == std::string::npos;
  d.endsWithNewline = !text.empty() && text.back() == '\n';
  std::size_t pos = 0;
  bool first = true;
  while (pos < text.size()) {
    std::size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string_view line(text.data() + pos, eol - pos);
    pos = eol + 1;
    if (first) {
      first = false;
      d.headerOk = (line == kPrimaryListHeader);
      continue;
    }
    std::string_view fields[kListFields];
    std::size_t nf = 0;
    std::size_t start = 0;
    bool tooMany = false;
    for (std::size_t i = 0; i <= line.size(); ++i) {
      if (i == line.size() || line[i] == ',') {
        if (nf == kListFields) {
          tooMany = true;
          break;
        }
        fields[nf++] = line.substr(start, i - start);
        start = i + 1;
      }
    }
    PrimaryRecord r;
    const bool ok = !tooMany && nf == kListFields && ParseI64(fields[0], r.id) &&
                    ParseF64(fields[1], r.x_m) && ParseF64(fields[2], r.y_m) &&
                    ParseF64(fields[3], r.z_m) && ParseF64(fields[4], r.dx) &&
                    ParseF64(fields[5], r.dy) && ParseF64(fields[6], r.dz) &&
                    ParseF64(fields[7], r.ekin_gev) && ParseI32(fields[8], r.pdg) &&
                    ParseI32(fields[9], r.charge_e);
    if (!ok) {
      ++d.badRows;
      continue;
    }
    out.push_back(r);
  }
  if (first) d.headerOk = false;  // empty file
}

void ValidatePrimaryRecords(const std::vector<PrimaryRecord>& recs, ListDefects& d)
{
  const std::size_t n = recs.size();
  if (n > 0) {
    d.idMin = std::numeric_limits<int64_t>::max();
    d.idMax = std::numeric_limits<int64_t>::min();
  }
  std::vector<int64_t> ids;
  ids.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    const auto& r = recs[k];
    d.idMin = std::min(d.idMin, r.id);
    d.idMax = std::max(d.idMax, r.id);
    if (r.id != static_cast<int64_t>(k)) d.idsContiguous = false;
    ids.push_back(r.id);

    const double v[] = {r.x_m, r.y_m, r.z_m, r.dx, r.dy, r.dz, r.ekin_gev};
    bool finite = true;
    for (const double x : v) finite = finite && std::isfinite(x);
    if (!finite) {
      ++d.nonFinite;
    }
    else {
      const double dev = std::fabs(std::sqrt(r.dx * r.dx + r.dy * r.dy + r.dz * r.dz) - 1.0);
      d.maxNormDev = std::max(d.maxNormDev, dev);
      if (dev > kNormTolerance) ++d.normFail;
      if (!(r.dz < 0.0)) ++d.dzNonNeg;
      if (!(r.ekin_gev > 0.0)) ++d.ekinNonPos;
    }
    if (r.pdg == 13) {
      ++d.nMuMinus;
      if (r.charge_e != -1) ++d.chargeMismatch;
    }
    else if (r.pdg == -13) {
      ++d.nMuPlus;
      if (r.charge_e != 1) ++d.chargeMismatch;
    }
    else {
      ++d.pdgBad;
    }
  }
  std::sort(ids.begin(), ids.end());
  for (std::size_t k = 1; k < ids.size(); ++k) {
    if (ids[k] == ids[k - 1]) ++d.dupIds;
  }

  // Exact duplicate initial states (bit patterns), independent of the id.
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  auto key = [&recs](std::size_t i) {
    const auto& r = recs[i];
    return std::make_tuple(Bits(r.x_m), Bits(r.y_m), Bits(r.z_m), Bits(r.dx), Bits(r.dy),
                           Bits(r.dz), Bits(r.ekin_gev), r.pdg);
  };
  std::sort(idx.begin(), idx.end(),
            [&key](std::size_t a, std::size_t b) { return key(a) < key(b); });
  for (std::size_t k = 1; k < idx.size(); ++k) {
    if (key(idx[k]) == key(idx[k - 1])) ++d.dupStates;
  }
}

bool ReadPrimaryFile(const std::string& path, std::vector<PrimaryRecord>& out,
                     ListDefects& d, std::string& fileSha256)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    d.readable = false;
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) {
    d.readable = false;
    return false;
  }
  const std::string text = ss.str();
  fileSha256 = Sha256Hex(text);
  ParsePrimaryText(text, out, d);
  return true;
}

std::string SidecarPath(const std::string& listPath) { return listPath + ".meta.json"; }

bool ReadSidecar(const std::string& path, ListSidecar& sc, std::string& err)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    err = "missing sidecar " + path;
    return false;
  }
  try {
    const nlohmann::json j = nlohmann::json::parse(in);
    sc.format = j.at("format").get<std::string>();
    sc.n = j.at("n").get<int64_t>();
    if (!ParseHex16(j.at("canon_fnv1a64").get<std::string>(), sc.canonFnv)) {
      err = "sidecar canon_fnv1a64 is not 16 hex digits: " + path;
      return false;
    }
    sc.haveCanonFnv = true;
    if (j.contains("geometry_id")) sc.geometryId = j.at("geometry_id").get<std::string>();
    if (j.contains("view_index")) sc.view = j.at("view_index").get<int64_t>();
    if (j.contains("base_radius_m")) {
      sc.baseRadiusM = j.at("base_radius_m").get<double>();
      sc.haveBaseRadius = true;
    }
    if (j.contains("case_code")) sc.caseCode = j.at("case_code").get<int64_t>();
    if (j.contains("replica")) sc.replica = j.at("replica").get<int64_t>();
    if (j.contains("source_seed") && !j.at("source_seed").is_null())
      sc.sourceSeed = j.at("source_seed").get<int64_t>();
    if (j.contains("file_sha256")) sc.fileSha256 = j.at("file_sha256").get<std::string>();
    if (j.contains("kind")) sc.kind = j.at("kind").get<std::string>();
  }
  catch (const std::exception& e) {
    err = "invalid sidecar " + path + ": " + e.what();
    return false;
  }
  return true;
}

bool WritePrimaryCsv(const std::string& path, const std::vector<PrimaryRecord>& recs,
                     std::string& err)
{
  std::string content = std::string(kPrimaryListHeader) + "\n";
  content.reserve(content.size() + recs.size() * 180);
  for (const auto& r : recs) content += FormatPrimaryRow(r);
  return WriteNewFile(path, content, err);
}

bool WriteSidecar(const std::string& path, const ListProvenance& p, std::size_t n,
                  std::string& err)
{
  nlohmann::json j;
  j["format"] = kPrimaryListFormat;
  j["kind"] = "source-model";
  j["geometry_id"] = p.geometryId;
  j["view_index"] = p.view;
  j["base_radius_m"] = p.baseRadiusM;
  j["start_radius_m"] = p.startRadiusM;
  j["case_code"] = p.caseCode;
  j["replica"] = p.replica;
  j["source_seed"] = p.sourceSeed;
  char tag[16];
  std::snprintf(tag, sizeof tag, "0x%08X", static_cast<unsigned>(p.sourceTag));
  j["rng"] = {
      {"engine", "CLHEP::MixMaxRng (private instance, installed with G4Random::setTheEngine "
                 "for generation only)"},
      {"key", {p.sourceSeed, p.caseCode, 0, p.sourceTag}},
      {"key_text", std::string("setSeeds({s_src, case_code, 0, TAG_SRC=") + tag + "}, 4)"},
      {"draws_per_primary", p.drawsPerPrimary}};
  j["generator"] = "legacy PrimaryGeneratorAction::GeneratePrimaries (unchanged) called for "
                   "each G4Event(id); the primary is read back from the event";
  j["n"] = n;
  j["columns"] = {"id", "x_m", "y_m", "z_m", "dx", "dy", "dz", "ekin_gev", "pdg", "charge_e"};
  j["units"] = {{"id", "Geant4 event ID, rows 0..n-1 in order"},
                {"position", "m, world frame (cone axis at the origin, ground at z = 0)"},
                {"direction", "unit travel direction"},
                {"ekin_gev", "kinetic energy, GeV"},
                {"pdg", "13 = mu-, -13 = mu+"},
                {"charge_e", "elementary charges"}};
  j["float_format"] = "std::to_chars shortest round-trip; LF line endings";
  j["canon_fnv1a64"] = Hex16(p.canonFnv);
  j["canon_hash_layout"] =
      "FNV-1a-64 over each record in file order: little-endian int64 id, float64 x_m, y_m, "
      "z_m, dx, dy, dz, ekin_gev, int32 pdg";
  j["file_sha256"] = p.fileSha256;
  j["binary_sha256"] = p.binarySha256;
  return WriteNewFile(path, j.dump(1) + "\n", err);
}

int RunCheckPrimaries(const std::string& path, std::ostream& out, std::ostream& errOut)
{
  std::vector<PrimaryRecord> recs;
  ListDefects d;
  std::string fileSha;
  if (!ReadPrimaryFile(path, recs, d, fileSha)) {
    out << "OE1-CHECK-PRIMARIES file=" << Basename(path) << " status=UNREADABLE\n";
    errOut << "cannot read primary list: " << path << "\n";
    return 1;
  }
  ValidatePrimaryRecords(recs, d);
  const uint64_t canon = CanonicalFnv(recs);

  ListSidecar sc;
  std::string scErr;
  const bool haveSidecar = ReadSidecar(SidecarPath(path), sc, scErr);
  const bool formatOk = haveSidecar && sc.format == kPrimaryListFormat;
  const bool nMatch = haveSidecar && sc.n == static_cast<int64_t>(recs.size());
  const bool fnvMatch = haveSidecar && sc.haveCanonFnv && sc.canonFnv == canon;
  const bool shaChecked = haveSidecar && !sc.fileSha256.empty();
  const bool shaMatch = shaChecked && sc.fileSha256 == fileSha;

  const bool ok = d.Clean() && !recs.empty() && formatOk && nMatch && fnvMatch &&
                  (!shaChecked || shaMatch);
  out << "OE1-CHECK-PRIMARIES file=" << Basename(path) << " " << DefectSummary(d, recs.size())
      << " canon_fnv1a64=" << Hex16(canon) << " file_sha256=" << fileSha
      << " sidecar=" << (haveSidecar ? (formatOk ? "ok" : "bad-format") : "missing")
      << " sidecar_n=" << (haveSidecar ? std::to_string(sc.n) : std::string("na"))
      << " sidecar_fnv_match=" << (fnvMatch ? 1 : 0)
      << " sidecar_sha_match=" << (shaChecked ? (shaMatch ? "1" : "0") : "na")
      << " status=" << (ok ? "OK" : "DEFECT") << "\n";
  if (!haveSidecar) errOut << scErr << "\n";
  return ok ? 0 : 1;
}

bool PrimaryList::Load(const std::string& path, const std::string& expectedSha256,
                       std::string& err)
{
  std::vector<PrimaryRecord> recs;
  ListDefects d;
  std::string fileSha;
  if (!ReadPrimaryFile(path, recs, d, fileSha)) {
    err = "cannot read primary list " + path;
    return false;
  }
  if (!expectedSha256.empty() && fileSha != expectedSha256) {
    err = "primary list SHA-256 mismatch: file " + fileSha + " != --primaries-sha256 " +
          expectedSha256;
    return false;
  }
  ValidatePrimaryRecords(recs, d);
  if (!d.Clean() || recs.empty()) {
    err = "primary list has defects: " + DefectSummary(d, recs.size());
    return false;
  }
  const uint64_t canon = CanonicalFnv(recs);
  ListSidecar sc;
  if (!ReadSidecar(SidecarPath(path), sc, err)) return false;
  if (sc.format != kPrimaryListFormat) {
    err = "sidecar format is '" + sc.format + "', expected '" + kPrimaryListFormat + "'";
    return false;
  }
  if (sc.geometryId.empty() || sc.view < 0 || !sc.haveBaseRadius || sc.caseCode < 0 ||
      sc.replica < 0) {
    err = "sidecar lacks geometry_id / view_index / base_radius_m / case_code / replica";
    return false;
  }
  if (sc.n != static_cast<int64_t>(recs.size())) {
    err = "sidecar n=" + std::to_string(sc.n) + " but the list has " +
          std::to_string(recs.size()) + " rows";
    return false;
  }
  if (!sc.haveCanonFnv || sc.canonFnv != canon) {
    err = "sidecar canon_fnv1a64 " + Hex16(sc.canonFnv) + " != content " + Hex16(canon);
    return false;
  }
  if (!sc.fileSha256.empty() && sc.fileSha256 != fileSha) {
    err = "sidecar file_sha256 " + sc.fileSha256 + " != file " + fileSha;
    return false;
  }
  fRecords.swap(recs);
  fDefects = d;
  fSidecar = sc;
  fCanonFnv = canon;
  fFileSha256 = fileSha;
  return true;
}

}  // namespace oe1
