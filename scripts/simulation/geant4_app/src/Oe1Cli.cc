#include "Oe1Cli.hh"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string_view>
#include <system_error>
#include <vector>

#include "Sha256.hh"

namespace oe1 {

namespace {

const std::vector<std::string>& LegacyValueFlags()
{
  static const std::vector<std::string> f = {"--geometry", "--view",   "--events",
                                             "--days",     "--seed",   "--out",
                                             "--tables",   "--threads"};
  return f;
}

const std::vector<std::string>& Stage1Flags()
{
  static const std::vector<std::string> f = {
      "--primaries",      "--primaries-sha256", "--roi-mask",       "--roi-mask-sha256",
      "--transport-seed", "--variant",          "--case-code",      "--replica",
      "--region-pair",    "--muon-log",         "--write-primaries", "--check-primaries"};
  return f;
}

const std::vector<std::string>& DiagnosticFlags()
{
  static const std::vector<std::string> f = {"--check-geometry", "--dump-geometry",
                                             "--probe",          "--slice-y",
                                             "--slice-z",        "--volume-stat"};
  return f;
}

bool Contains(const std::vector<std::string>& v, const std::string& s)
{
  return std::find(v.begin(), v.end(), s) != v.end();
}

bool ParseU64(const std::string& s, uint64_t lo, uint64_t hi, uint64_t& out)
{
  uint64_t v = 0;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || res.ec != std::errc() || res.ptr != s.data() + s.size()) return false;
  if (v < lo || v > hi) return false;
  out = v;
  return true;
}

bool ParseInt(const std::string& s, long long lo, long long hi, long long& out)
{
  long long v = 0;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || res.ec != std::errc() || res.ptr != s.data() + s.size()) return false;
  if (v < lo || v > hi) return false;
  out = v;
  return true;
}

bool ParsePair(const std::string& s, int& i, int& j)
{
  const std::size_t comma = s.find(',');
  if (comma == std::string::npos) return false;
  long long a = 0, b = 0;
  if (!ParseInt(s.substr(0, comma), 0, 1000000, a)) return false;
  if (!ParseInt(s.substr(comma + 1), 0, 1000000, b)) return false;
  if (a == b) return false;
  i = static_cast<int>(a);
  j = static_cast<int>(b);
  return true;
}

std::string FileName(const std::string& path)
{
  return std::filesystem::path(path).filename().string();
}

}  // namespace

bool AnyStage1Flag(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {
    if (Contains(Stage1Flags(), argv[i])) return true;
  }
  return false;
}

bool ParseStage1(int argc, char** argv, CliOptions& o, std::string& err)
{
  constexpr uint64_t kU32Max = 4294967295ULL;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (Contains(DiagnosticFlags(), a)) {
      err = a + " (P0 geometry diagnostics) cannot be combined with OE1 stage-1 flags";
      return false;
    }
    if (!Contains(LegacyValueFlags(), a) && !Contains(Stage1Flags(), a)) {
      err = "unknown argument '" + a + "'";
      return false;
    }
    if (!seen.insert(a).second) {
      err = a + " given more than once";
      return false;
    }
    if (i + 1 >= argc) {
      err = a + " needs a value";
      return false;
    }
    const std::string v = argv[++i];
    if (v.empty() || v.rfind("--", 0) == 0) {
      err = a + " needs a value (got '" + v + "')";
      return false;
    }
    long long n = 0;
    uint64_t u = 0;
    bool ok = true;
    if (a == "--geometry") o.geometry = v;
    else if (a == "--tables") o.tables = v;
    else if (a == "--out") o.out = v;
    else if (a == "--view") { ok = ParseInt(v, 0, 5, n); o.view = static_cast<int>(n); }
    else if (a == "--seed") ok = ParseU64(v, 0, UINT64_MAX, o.seed);
    else if (a == "--events") { ok = ParseInt(v, 1, 2147483647LL, n); o.events = n; }
    else if (a == "--threads") { ok = ParseInt(v, 1, 4096, n); o.threads = static_cast<int>(n); }
    else if (a == "--days") ok = true;  // rejected below: never valid in an OE1 mode
    else if (a == "--primaries") o.primaries = v;
    else if (a == "--roi-mask") o.roiMask = v;
    else if (a == "--write-primaries") o.writeOut = v;
    else if (a == "--check-primaries") o.checkFile = v;
    else if (a == "--primaries-sha256") { ok = IsSha256Hex(v); o.primariesSha256 = v; }
    else if (a == "--roi-mask-sha256") { ok = IsSha256Hex(v); o.roiMaskSha256 = v; }
    else if (a == "--transport-seed") ok = ParseU64(v, 1, kU32Max, o.transportSeed);
    else if (a == "--case-code") { ok = ParseU64(v, 0, kU32Max, u); o.caseCode = static_cast<uint32_t>(u); }
    else if (a == "--replica") { ok = ParseInt(v, 0, 2147483647LL, n); o.replica = static_cast<int>(n); }
    else if (a == "--variant") { ok = (v == "A" || v == "B"); o.variant = v; }
    else if (a == "--region-pair") {
      if (v == "none") {
        o.regionPairNone = true;  // deliberate: no region, every crossing field is "not applicable"
      }
      else {
        ok = ParsePair(v, o.regionFirst, o.regionSecond);
        o.haveRegionPair = ok;
      }
    }
    else if (a == "--muon-log") {
      ok = (v == "full" || v == "min");
      o.logLevel = (v == "min") ? MuonLogLevel::kMin : MuonLogLevel::kFull;
    }
    if (!ok) {
      err = "invalid value for " + a + ": '" + v + "'";
      return false;
    }
  }

  const int nModes = static_cast<int>(seen.count("--primaries") + seen.count("--write-primaries") +
                                      seen.count("--check-primaries"));
  if (nModes != 1) {
    err = "exactly one of --primaries, --write-primaries, --check-primaries is required "
          "with OE1 flags";
    return false;
  }

  std::vector<std::string> required, optional;
  std::string modeName;
  if (seen.count("--check-primaries")) {
    o.mode = CliMode::kCheck;
    modeName = "--check-primaries";
    required = {"--check-primaries"};
  }
  else if (seen.count("--write-primaries")) {
    o.mode = CliMode::kWrite;
    modeName = "--write-primaries";
    required = {"--write-primaries", "--geometry", "--view", "--tables",
                "--seed", "--events", "--case-code", "--replica"};
  }
  else {
    o.mode = CliMode::kList;
    modeName = "--primaries";
    required = {"--primaries", "--primaries-sha256", "--roi-mask", "--roi-mask-sha256",
                "--transport-seed", "--variant", "--case-code", "--replica",
                "--geometry", "--view", "--tables", "--seed"};
    optional = {"--region-pair", "--muon-log", "--out", "--threads"};
  }
  for (const auto& f : seen) {
    if (!Contains(required, f) && !Contains(optional, f)) {
      err = f + " is not allowed with " + modeName;
      return false;
    }
  }
  for (const auto& f : required) {
    if (!seen.count(f)) {
      err = modeName + " requires " + f;
      return false;
    }
  }
  if (o.mode == CliMode::kWrite && (o.seed < 1 || o.seed > kU32Max)) {
    err = "--write-primaries: the source seed --seed must be in [1, 4294967295] "
          "(MixMax keeps 32 bits per seed word)";
    return false;
  }
  if (o.mode == CliMode::kList && o.threads != 1) {
    err = "--primaries requires --threads 1 (one thread per process, PROT:26)";
    return false;
  }
  if (o.mode == CliMode::kList && o.seed % 2147483647ULL == 0) {
    err = "--primaries: the master --seed is reduced modulo 2147483647 and must not be a "
          "multiple of it (seed 0 is invalid for MixMaxRng)";
    return false;
  }
  return true;
}

void PrintStage1Usage(std::ostream& os, const char* prog)
{
  os << "OE1 stage-1 modes (strict parsing: any unknown or repeated argument exits 2):\n"
     << "  " << prog << " --primaries LIST.csv --primaries-sha256 HEX --roi-mask MASK.txt\n"
     << "      --roi-mask-sha256 HEX --transport-seed S --variant A|B --case-code K\n"
     << "      --replica R [--region-pair I,J|none] [--muon-log full|min] --geometry G.json\n"
     << "      --view V --tables DIR --seed MASTER [--out DIR] [--threads 1]\n"
     << "      (a geometry with two or more anomalies REQUIRES --region-pair: I,J = conduit,\n"
     << "       dense_body|cavity, or none to run without a region on purpose)\n"
     << "  " << prog << " --write-primaries OUT.csv --geometry G.json --view V --tables DIR\n"
     << "      --seed S_SRC --events N --case-code K --replica R\n"
     << "  " << prog << " --check-primaries LIST.csv\n";
}

int ConfigureListMode(const CliOptions& opt, const GeometryConfig& cfg,
                      const PrimaryList& list, const RoiMask& mask,
                      ListModeConfig& out, std::string& err)
{
  const ListSidecar& sc = list.Sidecar();
  if (sc.geometryId != cfg.geometry_id) {
    err = "list sidecar geometry_id '" + sc.geometryId + "' != geometry '" +
          cfg.geometry_id + "'";
    return 1;
  }
  if (sc.view != opt.view) {
    err = "list sidecar view_index " + std::to_string(sc.view) + " != --view " +
          std::to_string(opt.view);
    return 1;
  }
  if (!sc.haveBaseRadius || sc.baseRadiusM != cfg.base_radius_m) {
    err = "list sidecar base_radius_m does not equal the geometry's base_radius_m";
    return 1;
  }
  if (sc.caseCode != static_cast<int64_t>(opt.caseCode)) {
    err = "list sidecar case_code " + std::to_string(sc.caseCode) + " != --case-code " +
          std::to_string(opt.caseCode);
    return 1;
  }
  if (sc.replica != static_cast<int64_t>(opt.replica)) {
    err = "list sidecar replica " + std::to_string(sc.replica) + " != --replica " +
          std::to_string(opt.replica);
    return 1;
  }

  out.list = &list;
  out.mask = &mask;
  out.listFile = FileName(opt.primaries);
  out.maskFile = FileName(opt.roiMask);
  out.geometryId = cfg.geometry_id;
  out.view = opt.view;
  out.variant = opt.variant;
  out.caseCode = opt.caseCode;
  out.replica = opt.replica;
  out.transportSeed = opt.transportSeed;
  out.masterSeedCli = opt.seed;
  out.masterSeedApplied = static_cast<long>(opt.seed % 2147483647ULL);
  out.threadsCli = opt.threads;
  out.logLevel = opt.logLevel;
  out.geometryAnomalies = static_cast<int>(cfg.anomalies.size());

  // A forgotten pair on a geometry that has one must not turn every crossing
  // field into "not applicable" without anybody noticing (plan W2).
  if (!opt.haveRegionPair && !opt.regionPairNone && out.geometryAnomalies >= 2) {
    err = "the geometry has " + std::to_string(out.geometryAnomalies) +
          " anomalies: --region-pair I,J (conduit, dense_body|cavity) is required; "
          "pass --region-pair none to run without a region on purpose";
    return 2;
  }
  out.regionExplicitNone = opt.regionPairNone;

  if (opt.haveRegionPair) {
    if (!BuildRegionFromConfig(cfg, opt.regionFirst, opt.regionSecond, kRegionEpsM,
                               out.region, err)) {
      return 2;
    }
    out.regionApplicable = true;
    out.regionFirst = opt.regionFirst;
    out.regionSecond = opt.regionSecond;
    out.regionFirstKind = cfg.anomalies[static_cast<std::size_t>(opt.regionFirst)].kind;
    out.regionSecondKind = cfg.anomalies[static_cast<std::size_t>(opt.regionSecond)].kind;
    out.firstMaterial = "anomaly_" + std::to_string(opt.regionFirst) + "_mat";
    out.secondMaterial = "anomaly_" + std::to_string(opt.regionSecond) + "_mat";
  }
  return 0;
}

bool RecordProvenance(const CliOptions& opt, ListModeConfig& out, std::string& err)
{
  std::string sha;
  if (!Sha256File("/proc/self/exe", sha)) {
    err = "cannot read /proc/self/exe to record the SHA-256 of this executable";
    return false;
  }
  out.binarySha256 = sha;
  if (!Sha256File(opt.geometry, sha)) {
    err = "cannot read the geometry file to record its SHA-256: " + opt.geometry;
    return false;
  }
  out.geometryFile = FileName(opt.geometry);
  out.geometrySha256 = sha;
  out.tableSha256.clear();
  for (const char* name : {"row_weights_30d.txt", "p_cdf.txt", "meta.json"}) {
    const std::string path = opt.tables + "/" + name;
    if (!Sha256File(path, sha)) {
      err = "cannot read the table file to record its SHA-256: " + path;
      return false;
    }
    out.tableSha256.emplace_back(name, sha);
  }
  return true;
}

bool PreflightOutputs(const std::string& outDir, const std::string& stem, std::string& err)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(outDir, ec)) {
    err = "--out is not an existing directory: " + outDir;
    return false;
  }
  for (const char* suffix : {"_hist.csv", ".meta.json", "_muones.csv", "_muones.csv.part"}) {
    const std::string path = outDir + "/" + stem + suffix;
    if (fs::exists(path, ec)) {
      err = "output already exists (runs never overwrite; use a new folder): " + path;
      return false;
    }
  }
  const std::string probe = outDir + "/.oe1_write_probe_" + stem;
  {
    std::FILE* f = std::fopen(probe.c_str(), "wx");
    if (f == nullptr) {
      err = "--out is not writable: " + outDir;
      return false;
    }
    std::fclose(f);
  }
  fs::remove(probe, ec);
  return true;
}

}  // namespace oe1
