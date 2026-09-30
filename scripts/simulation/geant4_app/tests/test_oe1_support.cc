// OE1 stage 1 (P2): tests of the Geant4-free support code of the list mode:
// the strict command line (src/Oe1Cli.cc), the ROI mask reader
// (include/RoiMask.hh), the per-muon CSV row format and reconstruction
// helpers (include/MuonRecord.hh), the list-mode configuration
// (ConfigureListMode) and the thread-safe RunData accumulator.
// Standalone: built and run by tests/build_tests.sh (needs -pthread).
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "GeometryConfig.hh"
#include "MuonRecord.hh"
#include "Oe1Cli.hh"
#include "Oe1Hash.hh"
#include "Oe1ListMode.hh"
#include "PrimaryList.hh"
#include "RoiMask.hh"
#include "RunData.hh"
#include "Sha256.hh"
#include "check.hh"
#include "list_support.hh"

using oe1test::g_checks;
using oe1test::g_fail;
using oe1test::TempDir;

namespace {

const std::string kSha(64, 'a');
const std::string kSha2(64, 'b');

// ------------------------------------------------------------------ CLI
struct Args {
  std::vector<std::string> store;
  std::vector<char*> ptrs;
  Args(std::initializer_list<const char*> list)
  {
    store.emplace_back("muography_sim");
    for (const char* s : list) store.emplace_back(s);
    for (auto& s : store) ptrs.push_back(s.data());
  }
  int argc() const { return static_cast<int>(ptrs.size()); }
  char** argv() { return ptrs.data(); }
};

// A complete list-mode command line without the optional flags.
std::vector<const char*> ListArgs()
{
  static const std::string s1 = kSha, s2 = kSha2;
  return {"--geometry", "g.json", "--view", "0", "--tables", "tab", "--seed", "3026092901",
          "--out", "out", "--threads", "1", "--primaries", "L.csv", "--primaries-sha256",
          s1.c_str(), "--roi-mask", "M.txt", "--roi-mask-sha256", s2.c_str(),
          "--transport-seed", "1026092901", "--variant", "A", "--case-code", "6", "--replica",
          "1"};
}

bool Parse(const std::vector<const char*>& v, oe1::CliOptions& o, std::string& err)
{
  std::vector<std::string> store{"muography_sim"};
  for (const char* s : v) store.emplace_back(s);
  std::vector<char*> ptrs;
  for (auto& s : store) ptrs.push_back(s.data());
  o = oe1::CliOptions{};
  return oe1::ParseStage1(static_cast<int>(ptrs.size()), ptrs.data(), o, err);
}

bool Rejects(std::vector<const char*> v, const std::string& expectedFragment = "")
{
  oe1::CliOptions o;
  std::string err;
  const bool ok = Parse(v, o, err);
  return !ok && err.find(expectedFragment) != std::string::npos;
}

void TestCli()
{
  // Legacy command lines never reach the strict parser.
  {
    Args legacy{"--geometry", "g.json", "--view", "0", "--tables", "t", "--seed", "5", "--out",
                "o", "--events", "100", "--threads", "8"};
    CHECK(!oe1::AnyStage1Flag(legacy.argc(), legacy.argv()));
    Args diag{"--geometry", "g.json", "--check-geometry"};
    CHECK(!oe1::AnyStage1Flag(diag.argc(), diag.argv()));
    Args one{"--geometry", "g.json", "--muon-log", "min"};
    CHECK(oe1::AnyStage1Flag(one.argc(), one.argv()));
    Args last{"--replica"};
    CHECK(oe1::AnyStage1Flag(last.argc(), last.argv()));
  }

  oe1::CliOptions o;
  std::string err;
  CHECK(Parse(ListArgs(), o, err));
  CHECK(o.mode == oe1::CliMode::kList);
  CHECK(o.geometry == "g.json" && o.tables == "tab" && o.out == "out" && o.view == 0);
  CHECK(o.seed == 3026092901ULL && o.transportSeed == 1026092901ULL);
  CHECK(o.primaries == "L.csv" && o.primariesSha256 == kSha);
  CHECK(o.roiMask == "M.txt" && o.roiMaskSha256 == kSha2);
  CHECK(o.variant == "A" && o.caseCode == 6 && o.replica == 1 && o.threads == 1);
  CHECK(!o.haveRegionPair && !o.regionPairNone && o.logLevel == oe1::MuonLogLevel::kFull);

  auto with = [](std::initializer_list<const char*> extra) {
    auto v = ListArgs();
    v.insert(v.end(), extra);
    return v;
  };
  CHECK(Parse(with({"--region-pair", "0,1", "--muon-log", "min"}), o, err));
  CHECK(o.haveRegionPair && !o.regionPairNone && o.regionFirst == 0 && o.regionSecond == 1);
  CHECK(o.logLevel == oe1::MuonLogLevel::kMin);
  CHECK(Parse(with({"--muon-log", "full"}), o, err) && o.logLevel == oe1::MuonLogLevel::kFull);
  // --region-pair none is the explicit way to run without a region (plan W2).
  CHECK(Parse(with({"--region-pair", "none"}), o, err));
  CHECK(!o.haveRegionPair && o.regionPairNone);
  CHECK(Rejects(with({"--region-pair", "None"}), "invalid value"));  // exact word only
  CHECK(Rejects(with({"--region-pair", "none", "--region-pair", "0,1"}), "more than once"));

  // Unknown token, missing value, repeated flag, bad values: exit-2 class.
  CHECK(Rejects(with({"--bogus", "1"}), "unknown argument"));
  CHECK(Rejects(with({"stray"}), "unknown argument"));
  CHECK(Rejects(with({"--muon-log"}), "needs a value"));
  CHECK(Rejects(with({"--muon-log", "--region-pair", "0,1"}), "needs a value"));
  CHECK(Rejects(with({"--replica", "2"}), "more than once"));
  CHECK(Rejects(with({"--muon-log", "none"}), "invalid value"));
  CHECK(Rejects(with({"--region-pair", "1,1"}), "invalid value"));
  CHECK(Rejects(with({"--region-pair", "0"}), "invalid value"));
  CHECK(Rejects(with({"--region-pair", "a,b"}), "invalid value"));
  // Modes are mutually exclusive, also with the P0 diagnostic modes.
  CHECK(Rejects(with({"--write-primaries", "o.csv"}), "exactly one"));
  CHECK(Rejects(with({"--check-primaries", "L.csv"}), "exactly one"));
  CHECK(Rejects(with({"--check-geometry"}), "diagnostics"));
  CHECK(Rejects(with({"--dump-geometry"}), "diagnostics"));
  CHECK(Rejects(with({"--probe", "1,2,3"}), "diagnostics"));
  CHECK(Rejects(with({"--slice-y", "0:1:2:3:4:1"}), "diagnostics"));
  CHECK(Rejects(with({"--volume-stat", "1000"}), "diagnostics"));
  // List mode: no --events / --days, exactly one thread.
  CHECK(Rejects(with({"--events", "100"}), "not allowed"));
  CHECK(Rejects(with({"--days", "1"}), "not allowed"));
  {
    auto v = ListArgs();
    v[11] = "2";  // --threads 2
    CHECK(Rejects(v, "--threads 1"));
  }
  {
    auto v = ListArgs();
    v[7] = "5";  // --seed value still fine
    CHECK(Parse(v, o, err) && o.seed == 5);
    v[7] = "-5";
    CHECK(Rejects(v, "invalid value"));
    v[7] = "12abc";
    CHECK(Rejects(v, "invalid value"));
  }
  // The master seed is reduced modulo 2147483647: multiples of it (0 included)
  // would reach MixMaxRng::setSeed(0), which throws.
  {
    auto v = ListArgs();
    v[7] = "2147483647";
    CHECK(Rejects(v, "multiple"));
    v[7] = "4294967294";
    CHECK(Rejects(v, "multiple"));
    v[7] = "0";
    CHECK(Rejects(v, "multiple"));
    v[7] = "2147483648";  // 1 modulo 2147483647: fine
    CHECK(Parse(v, o, err));
  }
  // Every required flag is required.
  const char* required[] = {"--geometry", "--view", "--tables", "--seed", "--primaries",
                            "--primaries-sha256", "--roi-mask", "--roi-mask-sha256",
                            "--transport-seed", "--variant", "--case-code", "--replica"};
  for (const char* flag : required) {
    auto v = ListArgs();
    for (std::size_t i = 0; i + 1 < v.size(); ++i) {
      if (std::string(v[i]) == flag) {
        v.erase(v.begin() + static_cast<long>(i), v.begin() + static_cast<long>(i) + 2);
        break;
      }
    }
    // Without --primaries there is no mode at all.
    CHECK(Rejects(v, std::string(flag) == "--primaries" ? std::string("exactly one")
                                                        : std::string("requires ") + flag));
  }
  // Value checks.
  {
    auto v = ListArgs();
    v[15] = "notasha";  // --primaries-sha256
    CHECK(Rejects(v, "invalid value"));
    v = ListArgs();
    v[21] = "0";  // --transport-seed 0 is not allowed (MixMax zero seed)
    CHECK(Rejects(v, "invalid value"));
    v = ListArgs();
    v[21] = "4294967296";  // above 2^32-1: MixMax would alias it
    CHECK(Rejects(v, "invalid value"));
    v = ListArgs();
    v[23] = "C";  // --variant
    CHECK(Rejects(v, "invalid value"));
    v = ListArgs();
    v[3] = "6";  // --view 6
    CHECK(Rejects(v, "invalid value"));
  }

  // --write-primaries
  const std::vector<const char*> write = {"--write-primaries", "out.csv", "--geometry", "g.json",
                                          "--view", "0", "--tables", "t", "--seed", "2026092901",
                                          "--events", "50", "--case-code", "6", "--replica", "1"};
  CHECK(Parse(write, o, err));
  CHECK(o.mode == oe1::CliMode::kWrite && o.writeOut == "out.csv" && o.events == 50);
  CHECK(o.seed == 2026092901ULL && o.caseCode == 6 && o.replica == 1);
  {
    auto v = write;
    v.insert(v.end(), {"--primaries-sha256", kSha.c_str()});
    CHECK(Rejects(v, "exactly one") || Rejects(v, "not allowed"));
    v = write;
    v.insert(v.end(), {"--out", "o"});
    CHECK(Rejects(v, "not allowed"));
    v = write;
    v[9] = "0";  // --seed 0
    CHECK(Rejects(v, "source seed"));
    v = write;
    v[9] = "4294967296";
    CHECK(Rejects(v, "source seed"));
    v = write;
    v.erase(v.begin() + 10, v.begin() + 12);  // no --events
    CHECK(Rejects(v, "requires --events"));
    v = write;
    v[11] = "0";  // --events 0
    CHECK(Rejects(v, "invalid value"));
  }
  // --check-primaries
  CHECK(Parse({"--check-primaries", "L.csv"}, o, err));
  CHECK(o.mode == oe1::CliMode::kCheck && o.checkFile == "L.csv");
  CHECK(Rejects({"--check-primaries", "L.csv", "--geometry", "g.json"}, "not allowed"));
  CHECK(Rejects({"--check-primaries"}, "needs a value"));
  CHECK(Rejects({"--check-primaries", "L.csv", "--check-primaries", "M.csv"}, "more than once"));

  std::ostringstream usage;
  oe1::PrintStage1Usage(usage, "muography_sim");
  CHECK(usage.str().find("--primaries") != std::string::npos);
  CHECK(usage.str().find("--region-pair I,J|none") != std::string::npos);
}

// ------------------------------------------------------------------ ROI mask
std::string MaskText(int r0, int r1, int c0, int c1)
{
  std::string s;
  for (int r = 0; r < 256; ++r) {
    for (int c = 0; c < 256; ++c) s.push_back((r >= r0 && r <= r1 && c >= c0 && c <= c1) ? '1' : '0');
    s.push_back('\n');
  }
  return s;
}

void TestRoiMask()
{
  const std::string text = MaskText(17, 50, 131, 138);  // the R_int frame of the pilot
  oe1::RoiMask m;
  std::string err;
  CHECK(!m.Loaded());
  CHECK(m.LoadFromBytes(text, "", err));
  CHECK(m.Loaded());
  CHECK(m.Ones() == 34u * 8u);
  CHECK(m.RowMin() == 17 && m.RowMax() == 50 && m.ColMin() == 131 && m.ColMax() == 138);
  // [iEl, iAz] orientation: the pixel (34, 135) is inside, its transpose and
  // its azimuth mirror (column 255 - 135 = 120) are not.
  CHECK(m.At(34, 135));
  CHECK(!m.At(135, 34));
  CHECK(!m.At(34, 120));
  CHECK(!m.At(16, 135) && !m.At(51, 135) && !m.At(34, 130) && !m.At(34, 139));
  CHECK(m.At(17, 131) && m.At(50, 138));
  CHECK_EQ_STR(m.FileSha256(), oe1::Sha256Hex(text));
  CHECK(m.ContentFnv() == oe1::Fnv1a64Of(text));

  oe1::RoiMask ok;
  CHECK(ok.LoadFromBytes(text, oe1::Sha256Hex(text), err));
  oe1::RoiMask bad;
  CHECK(!bad.LoadFromBytes(text, std::string(64, '0'), err));
  CHECK(err.find("SHA-256 mismatch") != std::string::npos && !bad.Loaded());

  auto broken = [&](const char* what, std::string t) {
    oe1::RoiMask x;
    std::string e;
    ++g_checks;
    if (x.LoadFromBytes(t, "", e) || x.Loaded()) {
      oe1test::Fail(__FILE__, __LINE__, std::string("mask variant accepted: ") + what);
    }
  };
  broken("no final newline", text.substr(0, text.size() - 1));
  broken("extra line", text + std::string(256, '0') + "\n");
  broken("one line short", text.substr(0, text.size() - 257));
  broken("empty", "");
  {
    std::string t = text;
    t[5] = '2';
    broken("character 2", t);
    t = text;
    t[300] = ' ';
    broken("space", t);
    t = text;
    t[256] = 'x';  // LF position of line 0 replaced
    broken("missing LF", t);
    std::string crlf;
    for (const char c : text) {
      if (c == '\n') crlf.push_back('\r');
      crlf.push_back(c);
    }
    broken("CRLF", crlf);
  }
  // All-zero mask is valid but empty.
  oe1::RoiMask empty;
  CHECK(empty.LoadFromBytes(MaskText(1, 0, 1, 0), "", err));
  CHECK(empty.Ones() == 0 && empty.RowMin() == 256 && empty.RowMax() == -1);

  TempDir dir("mask");
  oe1test::WriteAll(dir.path / "M.txt", text);
  oe1::RoiMask f;
  CHECK(f.Load((dir.path / "M.txt").string(), oe1::Sha256Hex(text), err));
  CHECK(!f.Load((dir.path / "absent.txt").string(), "", err));
  CHECK(err.find("cannot open") != std::string::npos);
}

// ------------------------------------------------------------------ muon CSV
std::vector<std::string> Split(const std::string& line)
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

oe1::MuonRowConstants Constants()
{
  oe1::MuonRowConstants k;
  k.caseCode = 6;
  k.geometryId = "geo-train-06";
  k.view = 0;
  k.variant = "B";
  k.replica = 1;
  k.listSha256 = std::string(64, 'a');
  return k;
}

oe1::MuonRecord FullRecord()
{
  oe1::MuonRecord r;
  r.id = 12;
  r.primFnv = 1;
  r.rng0 = 2;
  r.rng1 = 3;
  r.detected = 1;
  r.iel = 34;
  r.iaz = 135;
  r.inRint = 1;
  r.planeReached = true;
  r.ekinPlaneGeV = 12.5;
  r.elPlaneDeg = 7.93;
  r.relAzPlaneDeg = 1.8;
  r.status = oe1::MuonStatus::kDet;
  r.endProcess = "Transportation";
  r.nSteps = 41;
  r.haveEnd = true;
  r.endXm = 1000.0;
  r.endYm = -0.5;
  r.endZm = 1.5;
  r.endVolume = "detector";
  r.deep = true;
  r.lz = 89.5;
  r.lzTrue = 89.75;
  r.lzA1 = 89.5;
  r.lc = 340.0;
  r.le = 129.25;
  r.rhoLzGcm2 = 294.0;
  r.nStepsZ = 3;
  r.nStepsZBothOut = 1;
  r.nEntriesZ = 1;
  r.matIn = oe1::MatCode::kA1;
  r.matOut = oe1::MatCode::kA1;
  r.haveEIn = true;
  r.eInGeV = 30.5;
  r.eInLoGeV = 30.25;
  r.eInHiGeV = 30.75;
  r.haveEOut = true;
  r.eOutGeV = 29.5;
  r.eOutLoGeV = 29.25;
  r.eOutHiGeV = 29.75;
  r.maxLatMm = 0.5;
  r.maxLatZMm = 0.25;
  r.sumLatZMm = 0.75;
  return r;
}

void TestMuonCsv()
{
  const std::vector<std::string> names = {
      "caso", "geometry_id", "view", "variant", "replica", "list_sha256", "id", "prim_fnv",
      "rng0", "rng1", "detected", "iel", "iaz", "in_rint", "ekin_plane_gev", "el_plane_deg",
      "relaz_plane_deg", "status", "end_process", "n_steps", "end_x_m", "end_y_m", "end_z_m",
      "end_vol", "crossed", "lz_m", "lz_true_m", "lz_rock_m", "lz_a0_m", "lz_a1_m",
      "lz_other_m", "lc_m", "le_m", "rho_lz_gcm2", "n_steps_z", "n_steps_z_both_out",
      "n_entries_z", "n_graze_z", "mat_in", "mat_out", "e_in_gev", "e_in_lo_gev",
      "e_in_hi_gev", "e_out_gev", "e_out_lo_gev", "e_out_hi_gev", "max_lat_mm",
      "max_lat_z_mm", "sum_lat_z_mm"};
  CHECK(static_cast<int>(names.size()) == oe1::kMuonCsvColumns);
  CHECK(oe1::kMuonCsvColumns == 49);
  CHECK(Split(oe1::MuonCsvHeader()) == names);

  oe1::MuonLogShape full;
  full.fullLog = true;
  full.regionApplicable = true;
  std::string row;
  oe1::AppendMuonCsvRow(row, Constants(), full, FullRecord());
  CHECK_EQ_STR(row,
               "6,geo-train-06,0,B,1," + std::string(64, 'a') +
                   ",12,0000000000000001,0000000000000002,0000000000000003,1,34,135,1,12.5,"
                   "7.93,1.8,DET,Transportation,41,1000,-0.5,1.5,detector,1,89.5,89.75,0,0,"
                   "89.5,0,340,129.25,294,3,1,1,0,A1,A1,30.5,30.25,30.75,29.5,29.25,29.75,"
                   "0.5,0.25,0.75\n");
  CHECK(Split(row.substr(0, row.size() - 1)).size() == static_cast<std::size_t>(oe1::kMuonCsvColumns));

  // Full log, region not applicable: crossed = -1 and empty Z columns; the
  // lateral displacement of the steps near Z is empty too, max_lat_mm is not.
  oe1::MuonLogShape noRegion;
  noRegion.fullLog = true;
  noRegion.regionApplicable = false;
  row.clear();
  oe1::AppendMuonCsvRow(row, Constants(), noRegion, FullRecord());
  auto f = Split(row.substr(0, row.size() - 1));
  CHECK(f.size() == static_cast<std::size_t>(oe1::kMuonCsvColumns));
  CHECK_EQ_STR(f[24], "-1");
  for (int i = 25; i <= 45; ++i) CHECK_EQ_STR(f[static_cast<std::size_t>(i)], "");
  CHECK_EQ_STR(f[20], "1000");  // end point still logged
  CHECK_EQ_STR(f[46], "0.5");   // max_lat_mm still logged
  CHECK_EQ_STR(f[47], "");      // max_lat_z_mm: no region, not applicable
  CHECK_EQ_STR(f[48], "");      // sum_lat_z_mm

  // Min log: no end point, no crossing columns, no lateral displacement at all.
  oe1::MuonLogShape min;
  min.fullLog = false;
  min.regionApplicable = true;
  row.clear();
  oe1::AppendMuonCsvRow(row, Constants(), min, FullRecord());
  f = Split(row.substr(0, row.size() - 1));
  CHECK(f.size() == static_cast<std::size_t>(oe1::kMuonCsvColumns));
  for (int i = 20; i <= 23; ++i) CHECK_EQ_STR(f[static_cast<std::size_t>(i)], "");
  for (int i = 24; i <= 48; ++i) CHECK_EQ_STR(f[static_cast<std::size_t>(i)], "");
  CHECK_EQ_STR(f[17], "DET");
  CHECK_EQ_STR(f[19], "41");
  CHECK_EQ_STR(f[11], "34");

  // A primary that never reached the plane: value-initialised record.
  oe1::MuonRecord nd;
  nd.id = 3;
  nd.status = oe1::MuonStatus::kWorldExit;
  nd.endProcess = "Transportation";
  nd.nSteps = 9;
  row.clear();
  oe1::AppendMuonCsvRow(row, Constants(), full, nd);
  f = Split(row.substr(0, row.size() - 1));
  CHECK(f.size() == static_cast<std::size_t>(oe1::kMuonCsvColumns));
  CHECK_EQ_STR(f[10], "0");    // detected
  CHECK_EQ_STR(f[11], "-1");   // iel
  CHECK_EQ_STR(f[12], "-1");   // iaz
  CHECK_EQ_STR(f[13], "-1");   // in_rint
  CHECK_EQ_STR(f[14], "");     // ekin_plane_gev: plane not reached
  CHECK_EQ_STR(f[15], "");
  CHECK_EQ_STR(f[16], "");
  CHECK_EQ_STR(f[17], "WORLD_EXIT");
  CHECK_EQ_STR(f[24], "0");    // crossed = 0 (region applicable, no deep step)
  CHECK_EQ_STR(f[25], "0");    // lz_m = 0
  CHECK_EQ_STR(f[38], "");     // mat_in absent
  CHECK_EQ_STR(f[40], "");     // no entry energy
  CHECK_EQ_STR(f[46], "0");    // max_lat_mm: a step exists, no lateral offset
  CHECK_EQ_STR(f[47], "0");    // no step near Z: 0 (region applicable), not "not applicable"
  CHECK_EQ_STR(f[48], "0");

  // Two default records give the same bytes (no uninitialised fields).
  std::string a, b;
  oe1::MuonRecord r1, r2;
  r1.id = r2.id = 5;
  oe1::AppendMuonCsvRow(a, Constants(), full, r1);
  oe1::AppendMuonCsvRow(b, Constants(), full, r2);
  CHECK(a == b);
  CHECK(a.find("nan") == std::string::npos && a.find("inf") == std::string::npos);

  // All status names are stable.
  const char* expected[] = {"DET", "PLANE_OUT", "WORLD_EXIT", "DECAY", "MU_CAPTURE", "STOPPED",
                            "OTHER", "ABORTED"};
  for (int i = 0; i < oe1::kNumMuonStatus; ++i) {
    CHECK_EQ_STR(oe1::StatusName(static_cast<oe1::MuonStatus>(i)), expected[i]);
  }
  CHECK(oe1::kNumMuonStatus == 8);
}

void TestReconstruction()
{
  // Ids: shuffled unique 0..n-1 pass; gaps, duplicates and wrong counts fail.
  std::vector<oe1::MuonRecord> recs(100);
  for (std::size_t i = 0; i < recs.size(); ++i) recs[i].id = static_cast<std::int64_t>(99 - i);
  CHECK(oe1::SortAndCheckIds(recs, 100));
  CHECK(recs.front().id == 0 && recs.back().id == 99);
  auto gap = recs;
  gap[50].id = 500;
  CHECK(!oe1::SortAndCheckIds(gap, 100));
  auto dup = recs;
  dup[50].id = 49;
  CHECK(!oe1::SortAndCheckIds(dup, 100));
  CHECK(!oe1::SortAndCheckIds(recs, 101));
  auto few = recs;
  few.pop_back();
  CHECK(!oe1::SortAndCheckIds(few, 100));

  // Histogram from DET rows only; other statuses never count.
  std::vector<oe1::MuonRecord> m(6);
  m[0].status = oe1::MuonStatus::kDet; m[0].iel = 3; m[0].iaz = 7;
  m[1].status = oe1::MuonStatus::kDet; m[1].iel = 3; m[1].iaz = 7;
  m[2].status = oe1::MuonStatus::kDet; m[2].iel = 0; m[2].iaz = 255;
  m[3].status = oe1::MuonStatus::kPlaneOut; m[3].iel = 3; m[3].iaz = 7;  // ignored
  m[4].status = oe1::MuonStatus::kWorldExit;
  m[5].status = oe1::MuonStatus::kDet; m[5].iel = 300; m[5].iaz = 1;     // out of range
  std::size_t badBins = 0;
  const auto h = oe1::RebuildHistogram(m, 256, 256, badBins);
  CHECK(h.size() == 65536u);
  CHECK(h[3 * 256 + 7] == 2 && h[255] == 1 && badBins == 1);
  std::uint64_t total = 0;
  for (const auto v : h) total += v;
  CHECK(total == 3);
}

// ------------------------------------------------------------------ list-mode configuration
GeometryConfig Geo06()
{
  GeometryConfig g;
  g.geometry_id = "geo-train-06";
  g.base_radius_m = 509.7;
  g.height_m = 416.3;
  AnomalyConfig c;
  c.kind = "conduit";
  c.density_g_cm3 = 1.832;
  c.cx = 20.3; c.cy = -24.5; c.radius = 21.0; c.z0 = 0.0; c.z1 = 340.0;
  AnomalyConfig e;
  e.kind = "dense_body";
  e.density_g_cm3 = 3.285;
  e.cx = 32.2; e.cy = -74.1; e.cz = 137.8; e.ax = 72.9; e.ay = 61.6; e.az = 64.6;
  g.anomalies = {c, e};
  return g;
}

void TestConfigureListMode()
{
  TempDir dir("cfg");
  const auto recs = oe1test::MakeRecords(32, 11);
  const std::string path = oe1test::WriteListAndSidecar(dir.path, "L.primarios.csv", recs);
  oe1::PrimaryList list;
  std::string err;
  CHECK(list.Load(path, "", err));
  oe1::RoiMask mask;
  CHECK(mask.LoadFromBytes(MaskText(17, 50, 131, 138), "", err));

  oe1::CliOptions o;
  CHECK(Parse(ListArgs(), o, err));
  o.primaries = path;
  o.roiMask = (dir.path / "M.txt").string();
  o.regionFirst = 0;
  o.regionSecond = 1;
  o.haveRegionPair = true;
  oe1::ListModeConfig cfg;
  CHECK(oe1::ConfigureListMode(o, Geo06(), list, mask, cfg, err) == 0);
  CHECK(cfg.list == &list && cfg.mask == &mask);
  CHECK_EQ_STR(cfg.listFile, "L.primarios.csv");  // file name only: no paths in the outputs
  CHECK_EQ_STR(cfg.maskFile, "M.txt");
  CHECK_EQ_STR(cfg.geometryId, "geo-train-06");
  CHECK(cfg.view == 0 && cfg.caseCode == 6 && cfg.replica == 1);
  CHECK_EQ_STR(cfg.variant, "A");
  CHECK(cfg.transportSeed == 1026092901ULL && cfg.masterSeedCli == 3026092901ULL);
  CHECK(cfg.masterSeedApplied == 878609254L);  // 3026092901 mod 2147483647
  CHECK(cfg.logLevel == oe1::MuonLogLevel::kFull);
  CHECK(cfg.regionApplicable && cfg.regionFirst == 0 && cfg.regionSecond == 1);
  CHECK_EQ_STR(cfg.regionSecondKind, "dense_body");
  CHECK_EQ_STR(cfg.firstMaterial, "anomaly_0_mat");
  CHECK_EQ_STR(cfg.secondMaterial, "anomaly_1_mat");
  CHECK_EQ_STR(cfg.rockMaterial, "standard_rock");
  CHECK(cfg.region.Valid());
  CHECK(cfg.geometryAnomalies == 2);
  CHECK_NEAR(cfg.region.Intersect({21.36, -30.71, -10.0}, {21.36, -30.71, 400.0}).len,
             89.67389128531408, 1e-9);

  // Plan W2: a geometry with two anomalies needs --region-pair I,J or an
  // explicit --region-pair none; a forgotten pair is an argument error (2).
  o.haveRegionPair = false;
  o.regionPairNone = false;
  oe1::ListModeConfig forgotten;
  CHECK(oe1::ConfigureListMode(o, Geo06(), list, mask, forgotten, err) == 2);
  CHECK(err.find("2 anomalies") != std::string::npos);
  CHECK(err.find("--region-pair I,J") != std::string::npos);
  CHECK(err.find("--region-pair none") != std::string::npos);
  CHECK(!forgotten.regionApplicable);
  // --region-pair none is accepted: no region on purpose.
  o.regionPairNone = true;
  oe1::ListModeConfig noRegion;
  CHECK(oe1::ConfigureListMode(o, Geo06(), list, mask, noRegion, err) == 0);
  CHECK(!noRegion.regionApplicable && !noRegion.region.Valid() && noRegion.regionExplicitNone);
  CHECK(noRegion.geometryAnomalies == 2);
  // Fewer than two anomalies: nothing to choose, no pair needed.
  o.regionPairNone = false;
  {
    GeometryConfig one = Geo06();
    one.anomalies.resize(1);
    oe1::ListModeConfig c1;
    CHECK(oe1::ConfigureListMode(o, one, list, mask, c1, err) == 0);
    CHECK(!c1.regionApplicable && !c1.regionExplicitNone && c1.geometryAnomalies == 1);
    GeometryConfig none = Geo06();
    none.anomalies.clear();
    oe1::ListModeConfig c0;
    CHECK(oe1::ConfigureListMode(o, none, list, mask, c0, err) == 0);
    CHECK(!c0.regionApplicable && c0.geometryAnomalies == 0);
    // A pair on such a geometry is still an argument error.
    oe1::CliOptions pair = o;
    pair.haveRegionPair = true;
    pair.regionFirst = 0;
    pair.regionSecond = 1;
    CHECK(oe1::ConfigureListMode(pair, one, list, mask, c1, err) == 2);
  }

  // Control geometry without anomalies and no region pair.
  GeometryConfig g01 = Geo06();
  g01.geometry_id = "geo-train-01";
  g01.anomalies.clear();
  oe1::ListModeConfig wrong;
  CHECK(oe1::ConfigureListMode(o, g01, list, mask, wrong, err) == 1);  // list is geo-train-06
  CHECK(err.find("geometry_id") != std::string::npos);

  // Every disagreement between list, geometry and command line is an input error.
  {
    oe1::CliOptions v = o;
    v.view = 1;
    CHECK(oe1::ConfigureListMode(v, Geo06(), list, mask, wrong, err) == 1);
    CHECK(err.find("view_index") != std::string::npos);
    v = o;
    v.caseCode = 1;
    CHECK(oe1::ConfigureListMode(v, Geo06(), list, mask, wrong, err) == 1);
    CHECK(err.find("case_code") != std::string::npos);
    v = o;
    v.replica = 2;
    CHECK(oe1::ConfigureListMode(v, Geo06(), list, mask, wrong, err) == 1);
    CHECK(err.find("replica") != std::string::npos);
    GeometryConfig g = Geo06();
    g.base_radius_m = 509.71;
    CHECK(oe1::ConfigureListMode(o, g, list, mask, wrong, err) == 1);
    CHECK(err.find("base_radius_m") != std::string::npos);
  }
  // Invalid region pairs are argument errors (2), not input errors.
  {
    oe1::CliOptions v = o;
    v.haveRegionPair = true;
    v.regionFirst = 1;
    v.regionSecond = 0;
    CHECK(oe1::ConfigureListMode(v, Geo06(), list, mask, wrong, err) == 2);
    v.regionFirst = 0;
    v.regionSecond = 2;
    CHECK(oe1::ConfigureListMode(v, Geo06(), list, mask, wrong, err) == 2);
  }
}

// ------------------------------------------------------------------ output preflight
void TestPreflight()
{
  namespace fs = std::filesystem;
  TempDir dir("preflight");
  const std::string stem = "geo-train-06_view0";
  std::string err;
  CHECK(oe1::PreflightOutputs(dir.path.string(), stem, err));
  CHECK(!fs::exists(dir.path / (".oe1_write_probe_" + stem)));  // the probe is removed

  CHECK(!oe1::PreflightOutputs((dir.path / "absent").string(), stem, err));
  CHECK(err.find("not an existing directory") != std::string::npos);

  // A regular file is not a directory.
  oe1test::WriteAll(dir.path / "afile", "x");
  CHECK(!oe1::PreflightOutputs((dir.path / "afile").string(), stem, err));

  // Any output of this run that already exists blocks the run.
  for (const char* suffix : {"_hist.csv", ".meta.json", "_muones.csv", "_muones.csv.part"}) {
    const fs::path f = dir.path / (stem + suffix);
    oe1test::WriteAll(f, "old");
    CHECK(!oe1::PreflightOutputs(dir.path.string(), stem, err));
    CHECK(err.find("already exists") != std::string::npos);
    CHECK(oe1test::ReadAll(f) == "old");  // untouched
    fs::remove(f);
    CHECK(oe1::PreflightOutputs(dir.path.string(), stem, err));
  }
  // Another geometry / view in the same folder does not collide.
  oe1test::WriteAll(dir.path / "geo-train-06_view1_hist.csv", "other");
  CHECK(oe1::PreflightOutputs(dir.path.string(), stem, err));

  // A read-only directory is rejected (skipped when running as root).
  if (geteuid() != 0) {
    fs::create_directories(dir.path / "ro");
    fs::permissions(dir.path / "ro", fs::perms::owner_read | fs::perms::owner_exec);
    CHECK(!oe1::PreflightOutputs((dir.path / "ro").string(), stem, err));
    CHECK(err.find("not writable") != std::string::npos);
    fs::permissions(dir.path / "ro", fs::perms::owner_all);
  }
}

// ------------------------------------------------------------------ provenance
void TestProvenance()
{
  namespace fs = std::filesystem;
  TempDir dir("prov");
  const std::string geometry = "{\"geometry_id\": \"synthetic\"}\n";
  const std::string weights = "1 2 3\n", cdf = "0.5 0.75\n", meta = "{\"k\": 1}\n";
  oe1test::WriteAll(dir.path / "g.json", geometry);
  fs::create_directories(dir.path / "tables");
  oe1test::WriteAll(dir.path / "tables" / "row_weights_30d.txt", weights);
  oe1test::WriteAll(dir.path / "tables" / "p_cdf.txt", cdf);
  oe1test::WriteAll(dir.path / "tables" / "meta.json", meta);

  oe1::CliOptions o;
  o.geometry = (dir.path / "g.json").string();
  o.tables = (dir.path / "tables").string();
  oe1::ListModeConfig cfg;
  std::string err;
  CHECK(oe1::RecordProvenance(o, cfg, err));
  CHECK_EQ_STR(cfg.geometryFile, "g.json");  // a name, never a path
  CHECK_EQ_STR(cfg.geometrySha256, oe1::Sha256Hex(geometry));
  CHECK(cfg.tableSha256.size() == 3u);
  if (cfg.tableSha256.size() == 3u) {
    CHECK_EQ_STR(cfg.tableSha256[0].first, "row_weights_30d.txt");
    CHECK_EQ_STR(cfg.tableSha256[0].second, oe1::Sha256Hex(weights));
    CHECK_EQ_STR(cfg.tableSha256[1].first, "p_cdf.txt");
    CHECK_EQ_STR(cfg.tableSha256[1].second, oe1::Sha256Hex(cdf));
    CHECK_EQ_STR(cfg.tableSha256[2].first, "meta.json");
    CHECK_EQ_STR(cfg.tableSha256[2].second, oe1::Sha256Hex(meta));
  }
  // The executable's own hash, as --write-primaries records it.
  std::string self;
  CHECK(oe1::Sha256File("/proc/self/exe", self));
  CHECK(oe1::IsSha256Hex(cfg.binarySha256));
  CHECK_EQ_STR(cfg.binarySha256, self);
  // Recording twice does not accumulate entries.
  CHECK(oe1::RecordProvenance(o, cfg, err) && cfg.tableSha256.size() == 3u);

  // A missing input is an error that names the file.
  fs::remove(dir.path / "tables" / "p_cdf.txt");
  CHECK(!oe1::RecordProvenance(o, cfg, err));
  CHECK(err.find("p_cdf.txt") != std::string::npos);
  oe1::CliOptions absent = o;
  absent.geometry = (dir.path / "absent.json").string();
  CHECK(!oe1::RecordProvenance(absent, cfg, err));
  CHECK(err.find("absent.json") != std::string::npos);
}

// ------------------------------------------------------------------ --variant vs geometry
void TestVariantCheck()
{
  using oe1::VariantCheck;
  const std::vector<std::string> variantA = {
      "anomaly_0:G4Tubs:anomaly_0_mat", "anomaly_1:G4Ellipsoid:anomaly_1_mat",
      "detector:G4Box:G4_AIR", "edifice:G4Cons:standard_rock", "world:G4Box:G4_AIR"};
  const std::vector<std::string> variantB = {
      "anomaly_0:G4SubtractionSolid:anomaly_0_mat", "anomaly_1:G4Ellipsoid:anomaly_1_mat",
      "detector:G4Box:G4_AIR", "edifice:G4Cons:standard_rock", "world:G4Box:G4_AIR"};
  std::string why;
  auto check = [&](const std::string& variant, int first, int second, const std::string& fk,
                   const std::string& sk, bool overlap, const std::vector<std::string>& sig) {
    return oe1::CheckVariantAgainstSignature(variant, first, second, fk, sk, overlap, sig, why);
  };

  // geo-train-06: anomaly_0 is a G4Tubs in A and a G4SubtractionSolid in B.
  CHECK(check("A", 0, 1, "conduit", "dense_body", true, variantA) == VariantCheck::kOk);
  CHECK(why.empty());
  CHECK(check("B", 0, 1, "conduit", "dense_body", true, variantB) == VariantCheck::kOk);
  CHECK(why.empty());
  // Mislabelled runs are caught in both directions, with a message that says why.
  CHECK(check("B", 0, 1, "conduit", "dense_body", true, variantA) == VariantCheck::kMismatch);
  CHECK(why.find("G4SubtractionSolid") != std::string::npos && why.find("G4Tubs") != std::string::npos);
  CHECK(check("A", 0, 1, "conduit", "dense_body", true, variantB) == VariantCheck::kMismatch);
  CHECK(why.find("--variant A") != std::string::npos && why.find("G4Tubs") != std::string::npos);
  CHECK(why.find("G4SubtractionSolid") != std::string::npos);
  // A cavity is a G4Orb in A.
  const std::vector<std::string> cavityA = {"anomaly_0:G4Tubs:m0", "anomaly_1:G4Orb:m1"};
  CHECK(check("A", 0, 1, "conduit", "cavity", true, cavityA) == VariantCheck::kOk);
  CHECK(check("A", 0, 1, "conduit", "dense_body", true, cavityA) == VariantCheck::kMismatch);
  // Missing anomaly volume.
  const std::vector<std::string> bare = {"world:G4Box:G4_AIR"};
  CHECK(check("A", 0, 1, "conduit", "dense_body", true, bare) == VariantCheck::kMismatch);
  CHECK(why.find("no such volume") != std::string::npos);
  CHECK(check("B", 0, 1, "conduit", "dense_body", true, bare) == VariantCheck::kMismatch);
  // B subtracts the LATER anomaly from the EARLIER one: with the ellipsoid first
  // (pair 1,0) it is anomaly_0 that must be the G4SubtractionSolid.
  const std::vector<std::string> swappedB = {"anomaly_0:G4SubtractionSolid:m0",
                                             "anomaly_1:G4Tubs:m1"};
  CHECK(check("B", 1, 0, "conduit", "dense_body", true, swappedB) == VariantCheck::kOk);
  const std::vector<std::string> swappedA = {"anomaly_0:G4Ellipsoid:m0", "anomaly_1:G4Tubs:m1"};
  CHECK(check("A", 1, 0, "conduit", "dense_body", true, swappedA) == VariantCheck::kOk);
  CHECK(check("B", 1, 0, "conduit", "dense_body", true, swappedA) == VariantCheck::kMismatch);
  // Primitives that do not overlap: B is not subtracted, A and B look alike.
  CHECK(check("B", 0, 1, "conduit", "dense_body", false, variantA) == VariantCheck::kNotChecked);
  CHECK(!why.empty());
  CHECK(check("A", 0, 1, "conduit", "dense_body", false, variantA) == VariantCheck::kOk);
  // Anything but A or B.
  CHECK(check("C", 0, 1, "conduit", "dense_body", true, variantA) == VariantCheck::kMismatch);

  // Signature parsing: "anomaly_1" never matches "anomaly_10".
  const std::vector<std::string> many = {"anomaly_10:G4Orb:m", "anomaly_1:G4Tubs:m"};
  CHECK_EQ_STR(oe1::SignatureSolid(many, 1), "G4Tubs");
  CHECK_EQ_STR(oe1::SignatureSolid(many, 10), "G4Orb");
  CHECK_EQ_STR(oe1::SignatureSolid(many, 2), "");
  CHECK_EQ_STR(oe1::kExpectedRunManager, "G4TaskRunManager");
}

// ------------------------------------------------------------------ checks before BeamOn (plan S3)
// main() evaluates oe1::CheckBeforeTransport right after runManager->Initialize() and exits
// kExitRunCheckFailed before BeamOn when it reports an error; RunAction repeats it at the end of
// the run. The function is pure, so its whole decision table is tested here without Geant4.
void TestPreRunChecks()
{
  using oe1::VariantCheck;
  const std::vector<std::string> sigA = {
      "anomaly_0:G4Tubs:anomaly_0_mat", "anomaly_1:G4Ellipsoid:anomaly_1_mat",
      "detector:G4Box:G4_AIR", "edifice:G4Cons:standard_rock", "world:G4Box:G4_AIR"};
  const std::vector<std::string> sigB = {
      "anomaly_0:G4SubtractionSolid:anomaly_0_mat", "anomaly_1:G4Ellipsoid:anomaly_1_mat",
      "detector:G4Box:G4_AIR", "edifice:G4Cons:standard_rock", "world:G4Box:G4_AIR"};

  CHECK(oe1::kExitRunCheckFailed == 4);
  CHECK_EQ_STR(oe1::kErrRunManagerNotTask, "run_manager_not_task");
  CHECK_EQ_STR(oe1::kErrVariantMismatch, "variant_geometry_mismatch");

  oe1::ListModeConfig lm;
  std::string err;
  lm.regionApplicable = true;
  lm.regionFirst = 0;
  lm.regionSecond = 1;
  lm.regionFirstKind = "conduit";
  lm.regionSecondKind = "dense_body";
  CHECK(oe1::BuildRegionFromConfig(Geo06(), 0, 1, oe1::kRegionEpsM, lm.region, err));
  CHECK(lm.region.ZBoxNonEmpty());
  lm.runManagerClass = oe1::kExpectedRunManager;

  // Everything as it should be: the transport may start.
  lm.variant = "A";
  oe1::PreRunReport r = oe1::CheckBeforeTransport(lm, sigA);
  CHECK(r.Ok() && r.errors.empty() && r.messages.empty() && r.variantCheck == VariantCheck::kOk);
  lm.variant = "B";
  r = oe1::CheckBeforeTransport(lm, sigB);
  CHECK(r.Ok() && r.errors.empty() && r.variantCheck == VariantCheck::kOk);

  // A mislabelled --variant is the only error, in both directions, with a message that says why.
  r = oe1::CheckBeforeTransport(lm, sigA);  // --variant B on the geometry of A
  CHECK(!r.Ok() && r.errors == std::vector<std::string>{"variant_geometry_mismatch"});
  CHECK(r.variantCheck == VariantCheck::kMismatch && r.messages.size() == 1u);
  CHECK(r.messages[0] == r.variantWhy);
  CHECK(r.messages[0].find("--variant B") != std::string::npos &&
        r.messages[0].find("G4SubtractionSolid") != std::string::npos);
  lm.variant = "A";
  r = oe1::CheckBeforeTransport(lm, sigB);  // --variant A on the geometry of B
  CHECK(!r.Ok() && r.errors == std::vector<std::string>{"variant_geometry_mismatch"});
  CHECK(r.messages.size() == 1u && r.messages[0].find("--variant A") != std::string::npos);

  // A run manager that is not the task-based one is the only error when the variant is right.
  for (const char* cls : {"G4SerialRunManager", "G4MTRunManager", "G4RunManager", ""}) {
    lm.runManagerClass = cls;
    lm.variant = "B";
    r = oe1::CheckBeforeTransport(lm, sigB);
    CHECK(!r.Ok() && r.errors == std::vector<std::string>{"run_manager_not_task"});
    CHECK(r.variantCheck == VariantCheck::kOk && r.messages.size() == 1u);
    CHECK(r.messages[0].find(std::string("'") + cls + "'") != std::string::npos);
    CHECK(r.messages[0].find("G4TaskRunManager") != std::string::npos);
  }

  // Both wrong: both errors, in a fixed order (run manager first), one message each.
  lm.runManagerClass = "G4SerialRunManager";
  lm.variant = "A";
  r = oe1::CheckBeforeTransport(lm, sigB);
  CHECK(r.errors == (std::vector<std::string>{"run_manager_not_task", "variant_geometry_mismatch"}));
  CHECK(r.messages.size() == 2u && r.variantCheck == VariantCheck::kMismatch);

  // No region pair (--region-pair none, the geo-train-01 way): the variant cannot be checked and
  // is not; the run manager still is.
  lm.regionApplicable = false;
  lm.runManagerClass = oe1::kExpectedRunManager;
  lm.variant = "A";
  r = oe1::CheckBeforeTransport(lm, sigB);
  CHECK(r.Ok() && r.variantCheck == VariantCheck::kNotChecked && r.variantWhy.empty());
  lm.runManagerClass = "G4SerialRunManager";
  r = oe1::CheckBeforeTransport(lm, sigB);
  CHECK(r.errors == std::vector<std::string>{"run_manager_not_task"});
  CHECK(r.variantCheck == VariantCheck::kNotChecked);

  // Primitives that do not overlap: B is not subtracted, so B cannot be told from A. That is
  // "not checked", never an error.
  {
    GeometryConfig separated = Geo06();
    separated.anomalies[1].cx += 1000.0;
    oe1::ListModeConfig apart = lm;
    apart.regionApplicable = true;
    apart.runManagerClass = oe1::kExpectedRunManager;
    apart.variant = "B";
    CHECK(oe1::BuildRegionFromConfig(separated, 0, 1, oe1::kRegionEpsM, apart.region, err));
    CHECK(!apart.region.ZBoxNonEmpty());
    r = oe1::CheckBeforeTransport(apart, sigA);
    CHECK(r.Ok() && r.variantCheck == VariantCheck::kNotChecked && !r.variantWhy.empty());
  }
}

// ------------------------------------------------------------------ RunData
void TestRunData()
{
  RunData& d = RunData::Instance();
  CHECK(!d.HasErrors());
  CHECK(d.Muons().empty());
  d.ReserveMuons(4000);
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([t]() {
      for (int i = 0; i < 1000; ++i) {
        oe1::MuonRecord r;
        r.id = t * 1000 + i;
        RunData::Instance().AddMuon(std::move(r));
        if (i == 0) RunData::Instance().NoteWorker(t, "MixMaxRng");
      }
      RunData::Instance().SetError("test_error");
    });
  }
  for (auto& th : threads) th.join();
  CHECK(d.Muons().size() == 4000u);
  CHECK(oe1::SortAndCheckIds(d.Muons(), 4000));
  CHECK(d.WorkerIds().size() == 4u);
  CHECK(d.EngineNames().size() == 1u && *d.EngineNames().begin() == "MixMaxRng");
  CHECK(d.HasErrors() && d.Errors().size() == 1u);
  CHECK(!d.Finalized());
  d.MarkFinalized();
  CHECK(d.Finalized());
  // The legacy accumulators are untouched by list-mode bookkeeping.
  CHECK(d.Detected() == 0 && d.Generated() == 0);
  d.RecordHit(7, 3, 12.5);
  CHECK(d.Detected() == 1 && d.Histogram()[3 * 256 + 7] == 1);
  CHECK(d.EkinSumGeV() == 12.5 && d.MeanEkinGeV() == 12.5);
}

}  // namespace

int main()
{
  struct Step {
    const char* name;
    void (*fn)();
  };
  const Step steps[] = {{"strict command line (exit-2 class errors, modes, ranges)", TestCli},
                        {"ROI mask reader: format, orientation [iEl, iAz], sha", TestRoiMask},
                        {"per-muon CSV: columns, NA rules, formatting", TestMuonCsv},
                        {"id check and histogram reconstruction", TestReconstruction},
                        {"ConfigureListMode: consistency with the geometry", TestConfigureListMode},
                        {"output preflight: never overwrite, writable folder", TestPreflight},
                        {"provenance: binary, geometry and table SHA-256", TestProvenance},
                        {"--variant against the geometry signature", TestVariantCheck},
                        {"checks before BeamOn (run manager, --variant): decision table", TestPreRunChecks},
                        {"RunData: thread-safe bookkeeping", TestRunData}};
  for (const Step& s : steps) {
    const int before = g_fail;
    s.fn();
    std::printf("%-4s %s\n", g_fail == before ? "PASS" : "FAIL", s.name);
  }
  std::printf("test_oe1_support: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
