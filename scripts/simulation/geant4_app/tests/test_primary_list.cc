// OE1 stage 1 (P2): tests of the frozen primary list (include/PrimaryList.hh,
// src/PrimaryList.cc), the SHA-256 and FNV-1a-64 helpers, and the sidecar /
// --check-primaries logic. Standalone (no Geant4): built and run by
// tests/build_tests.sh with the production flags and with sanitizers.
//
// Reference values (SHA-256 vectors, FNV-1a-64 vectors, the canonical hash of
// a fixed two-record list, the hash of an engine-state vector) were computed
// with Python's hashlib / struct (little-endian '<q7di') and are pinned here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Oe1Hash.hh"
#include "PrimaryList.hh"
#include "Sha256.hh"
#include "check.hh"
#include "json.hpp"
#include "list_support.hh"

namespace fs = std::filesystem;
using oe1test::g_checks;
using oe1test::g_fail;

namespace {

// ------------------------------------------------------------------ helpers
using oe1test::Bits;
using oe1test::MakeProvenance;
using oe1test::MakeRecords;
using oe1test::ReadAll;
using oe1test::SameRecord;
using oe1test::TempDir;
using oe1test::WriteAll;
using oe1test::WriteListAndSidecar;

std::string MakeText(const std::vector<oe1::PrimaryRecord>& recs)
{
  std::string text = std::string(oe1::kPrimaryListHeader) + "\n";
  for (const auto& r : recs) text += oe1::FormatPrimaryRow(r);
  return text;
}

oe1::ListDefects Check(const std::vector<oe1::PrimaryRecord>& recs)
{
  oe1::ListDefects d;
  oe1::ValidatePrimaryRecords(recs, d);
  return d;
}

oe1::ListDefects CheckText(const std::string& text, std::vector<oe1::PrimaryRecord>& out)
{
  oe1::ListDefects d;
  oe1::ParsePrimaryText(text, out, d);
  oe1::ValidatePrimaryRecords(out, d);
  return d;
}

// ------------------------------------------------------------------ tests
void TestSha256()
{
  CHECK_EQ_STR(oe1::Sha256Hex(""),
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK_EQ_STR(oe1::Sha256Hex("abc"),
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK_EQ_STR(oe1::Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // One million 'a', fed in irregular chunks.
  {
    oe1::Sha256 s;
    const std::string chunk(997, 'a');
    std::size_t left = 1000000;
    while (left > 0) {
      const std::size_t take = std::min(left, chunk.size());
      s.Update(chunk.data(), take);
      left -= take;
    }
    CHECK_EQ_STR(oe1::Sha256::Hex(s.Final()),
                 "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }
  // Padding boundaries.
  const struct {
    std::size_t n;
    const char* hex;
  } cases[] = {
      {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
      {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
      {57, "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6"},
      {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
      {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
      {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
      {119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"},
      {120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"},
      {121, "e9615320128cc7a3d6078e9af05603188e5ccbf0d07d8b735d3df5e8e0c1281f"},
      {127, "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a"},
      {128, "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e"},
      {129, "c12cb024a2e5551cca0e08fce8f1c5e314555cc3fef6329ee994a3db752166ae"}};
  for (const auto& c : cases) CHECK_EQ_STR(oe1::Sha256Hex(std::string(c.n, 'a')), c.hex);

  CHECK(oe1::IsSha256Hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK(!oe1::IsSha256Hex("E3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK(!oe1::IsSha256Hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"));
  CHECK(!oe1::IsSha256Hex(""));

  TempDir dir("sha");
  const fs::path f = dir.path / "abc.bin";
  WriteAll(f, "abc");
  std::string hex;
  CHECK(oe1::Sha256File(f.string(), hex));
  CHECK_EQ_STR(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(!oe1::Sha256File((dir.path / "missing").string(), hex));
}

void TestHashHelpers()
{
  CHECK(oe1::Fnv1a64Of("") == 0xcbf29ce484222325ULL);
  CHECK(oe1::Fnv1a64Of("a") == 0xaf63dc4c8601ec8cULL);
  CHECK(oe1::Fnv1a64Of("foobar") == 0x85944171f73967e8ULL);
  CHECK(oe1::HashEngineWords({1UL, 2UL, 3UL}) == 0xda2bfb225e0d1f05ULL);
  CHECK(oe1::HashEngineWords({}) == oe1::kFnvOffsetBasis);
  CHECK_EQ_STR(oe1::Hex16(0), "0000000000000000");
  CHECK_EQ_STR(oe1::Hex16(0xda2bfb225e0d1f05ULL), "da2bfb225e0d1f05");
  CHECK_EQ_STR(oe1::Hex16(~0ULL), "ffffffffffffffff");
  std::uint64_t v = 0;
  CHECK(oe1::ParseHex16("da2bfb225e0d1f05", v) && v == 0xda2bfb225e0d1f05ULL);
  CHECK(oe1::ParseHex16("DA2BFB225E0D1F05", v) && v == 0xda2bfb225e0d1f05ULL);
  CHECK(!oe1::ParseHex16("da2bfb225e0d1f0", v));    // 15 digits
  CHECK(!oe1::ParseHex16("da2bfb225e0d1f0g", v));   // non-hex
  CHECK(!oe1::ParseHex16("da2bfb225e0d1f055", v));  // 17 digits

  // Little-endian feeding is independent of the host.
  oe1::Fnv1a64 h;
  h.I64(-2);
  h.F64(0.5);
  h.I32(-13);
  unsigned char expected[8 + 8 + 4] = {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0,    0,    0,    0,    0,    0,    0xe0, 0x3f,
                                       0xf3, 0xff, 0xff, 0xff};
  oe1::Fnv1a64 g;
  g.Bytes(expected, sizeof expected);
  CHECK(h.Value() == g.Value());
}

void TestNumberRoundTrip()
{
  // to_chars / from_chars must give back the exact bits (the list format
  // relies on it): random doubles over the whole normal range, specials.
  std::mt19937_64 gen(2026);
  std::vector<double> values = {0.0, -0.0, 1.0, -1.0, 0.1, 1e-300, 1e300,
                                std::numeric_limits<double>::min(),
                                std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::denorm_min(), 1809.7, 89.67389128531408};
  for (int i = 0; i < 100000; ++i) {
    std::uint64_t b = gen();
    double v;
    std::memcpy(&v, &b, sizeof v);
    if (std::isfinite(v)) values.push_back(v);
  }
  int bad = 0;
  for (const double v : values) {
    oe1::PrimaryRecord r;
    r.id = 0;
    r.x_m = v;
    r.y_m = -v;
    r.z_m = v;
    r.dx = 0.0;
    r.dy = 0.0;
    r.dz = -1.0;
    r.ekin_gev = 1.0;
    r.pdg = 13;
    r.charge_e = -1;
    std::vector<oe1::PrimaryRecord> back;
    oe1::ListDefects d;
    oe1::ParsePrimaryText(std::string(oe1::kPrimaryListHeader) + "\n" + oe1::FormatPrimaryRow(r),
                          back, d);
    if (back.size() != 1 || !SameRecord(back[0], r)) ++bad;
  }
  CHECK(bad == 0);
  // Exact text of a simple row (shortest round-trip form).
  oe1::PrimaryRecord r;
  r.id = 7;
  r.x_m = 1.5;
  r.y_m = -2.25;
  r.z_m = 3.0;
  r.dx = 0.0;
  r.dy = 0.0;
  r.dz = -1.0;
  r.ekin_gev = 10.0;
  r.pdg = -13;
  r.charge_e = 1;
  CHECK_EQ_STR(oe1::FormatPrimaryRow(r), "7,1.5,-2.25,3,0,0,-1,10,-13,1\n");
}

void TestCanonicalHash()
{
  std::vector<oe1::PrimaryRecord> two(2);
  two[0] = {0, 1.5, -2.25, 3.0, 0.0, 0.0, -1.0, 10.0, 13, -1};
  two[1] = {1, -1809.7, 0.1, 1234.5, 0.6, -0.8, 0.0, 3.5, -13, 1};
  // Python: fnv1a64(b''.join(struct.pack('<q7di', ...)))
  CHECK_EQ_STR(oe1::Hex16(oe1::CanonicalFnv(two)), "7549cd46cac3e7b1");
  // charge_e is not hashed (redundant with pdg); every other field is.
  auto h0 = oe1::CanonicalFnv(two);
  auto mod = two;
  mod[1].charge_e = 5;
  CHECK(oe1::CanonicalFnv(mod) == h0);
  for (int field = 0; field < 9; ++field) {
    mod = two;
    oe1::PrimaryRecord& r = mod[1];
    switch (field) {
      case 0: r.id += 1; break;
      case 1: r.x_m = std::nextafter(r.x_m, 1e9); break;
      case 2: r.y_m = std::nextafter(r.y_m, 1e9); break;
      case 3: r.z_m = std::nextafter(r.z_m, 1e9); break;
      case 4: r.dx = std::nextafter(r.dx, 1e9); break;
      case 5: r.dy = std::nextafter(r.dy, 1e9); break;
      case 6: r.dz = -0.0; break;  // +0 -> -0: different bits
      case 7: r.ekin_gev = std::nextafter(r.ekin_gev, 1e9); break;
      default: r.pdg = 13; break;
    }
    CHECK(oe1::CanonicalFnv(mod) != h0);
  }
  CHECK(oe1::CanonicalFnv({}) == oe1::kFnvOffsetBasis);
}

void TestValidation()
{
  const auto good = MakeRecords(500, 1);
  oe1::ListDefects d = Check(good);
  CHECK(d.Clean());
  CHECK(d.idMin == 0 && d.idMax == 499 && d.idsContiguous);
  CHECK(d.nMuPlus + d.nMuMinus == 500);
  CHECK(d.nMuPlus > 200 && d.nMuPlus < 350);  // ~0.5613 of 500
  CHECK(d.maxNormDev < 1e-15);

  auto mut = good;
  mut[5].id = 4;  // duplicate id and a gap
  d = Check(mut);
  CHECK(!d.Clean() && d.dupIds == 1 && !d.idsContiguous);

  mut = good;
  mut[5].id = 99999;  // out of sequence
  d = Check(mut);
  CHECK(!d.Clean() && !d.idsContiguous && d.dupIds == 0);

  mut = good;
  mut[3].dz *= 1.000000001;  // |d| off by ~1e-9 > 1e-12
  d = Check(mut);
  CHECK(!d.Clean() && d.normFail == 1 && d.maxNormDev > 1e-12);

  mut = good;
  mut[3].dz = -mut[3].dz;  // upward
  d = Check(mut);
  CHECK(!d.Clean() && d.dzNonNeg == 1);

  mut = good;
  mut[3].dx = 0.0;
  mut[3].dy = 0.0;
  mut[3].dz = 0.0;  // zero vector: norm fails, dz not negative
  d = Check(mut);
  CHECK(!d.Clean() && d.normFail == 1 && d.dzNonNeg == 1);

  mut = good;
  mut[4].ekin_gev = 0.0;
  mut[6].ekin_gev = -1.0;
  d = Check(mut);
  CHECK(!d.Clean() && d.ekinNonPos == 2);

  mut = good;
  mut[8].pdg = 11;
  d = Check(mut);
  CHECK(!d.Clean() && d.pdgBad == 1);

  mut = good;
  mut[8].charge_e = mut[8].pdg == 13 ? 1 : -1;
  d = Check(mut);
  CHECK(!d.Clean() && d.chargeMismatch == 1);

  mut = good;
  mut[9] = good[2];
  mut[9].id = 9;  // identical initial state under another id
  d = Check(mut);
  CHECK(!d.Clean() && d.dupStates == 1 && d.dupIds == 0);

  mut = good;
  mut[1].x_m = std::numeric_limits<double>::quiet_NaN();
  mut[2].ekin_gev = std::numeric_limits<double>::infinity();
  d = Check(mut);
  CHECK(!d.Clean() && d.nonFinite == 2);

  // Text level.
  std::vector<oe1::PrimaryRecord> out;
  const std::string text = MakeText(good);
  d = CheckText(text, out);
  CHECK(d.Clean() && out.size() == 500);
  bool same = out.size() == good.size();
  for (std::size_t k = 0; same && k < out.size(); ++k) same = SameRecord(out[k], good[k]);
  CHECK(same);

  auto crlf = text;
  for (std::size_t pos = 0; (pos = crlf.find('\n', pos)) != std::string::npos; pos += 2)
    crlf.insert(pos, "\r");
  out.clear();
  d = CheckText(crlf, out);
  CHECK(!d.Clean() && !d.lfOnly);

  auto badHeader = text;
  badHeader[0] = 'x';
  out.clear();
  d = CheckText(badHeader, out);
  CHECK(!d.Clean() && !d.headerOk);

  auto noNewline = text;
  noNewline.pop_back();
  out.clear();
  d = CheckText(noNewline, out);
  CHECK(!d.Clean() && !d.endsWithNewline && out.size() == 500);

  out.clear();
  d = CheckText("", out);
  CHECK(!d.Clean() && !d.headerOk);

  auto nine = text;
  nine.replace(nine.find('\n') + 1, 0, "0,1,2,3,4,5,6,7,8\n");  // 9 fields
  out.clear();
  d = CheckText(nine, out);
  CHECK(!d.Clean() && d.badRows == 1);

  auto eleven = text;
  eleven.replace(eleven.find('\n') + 1, 0, "0,1,2,3,4,5,6,7,8,9,10\n");
  out.clear();
  d = CheckText(eleven, out);
  CHECK(!d.Clean() && d.badRows == 1);

  auto badNum = text;
  badNum.replace(badNum.find('\n') + 1, 0, "0,1.2.3,2,3,4,5,6,7,13,-1\n");
  out.clear();
  d = CheckText(badNum, out);
  CHECK(!d.Clean() && d.badRows == 1);

  auto blank = text + "\n";
  out.clear();
  d = CheckText(blank, out);
  CHECK(!d.Clean() && d.badRows == 1);

  auto lead = text;
  lead.replace(lead.find('\n') + 1, 0, "0, 1,2,3,4,5,6,7,13,-1\n");  // space in a field
  out.clear();
  d = CheckText(lead, out);
  CHECK(!d.Clean() && d.badRows == 1);

  auto plus = text;
  plus.replace(plus.find('\n') + 1, 0, "0,+1,2,3,4,5,6,7,13,-1\n");  // '+' sign is not accepted
  out.clear();
  d = CheckText(plus, out);
  CHECK(!d.Clean() && d.badRows == 1);
}

nlohmann::json ReadJson(const std::string& path)
{
  std::ifstream in(path);
  return nlohmann::json::parse(in);
}

void WriteJson(const std::string& path, const nlohmann::json& j)
{
  WriteAll(path, j.dump(1) + "\n");
}

void TestFilesAndLoad()
{
  TempDir dir("list");
  const auto recs = MakeRecords(64, 3);
  const std::string path = (dir.path / "geo-train-06_v0_r1_s2026092901.primarios.csv").string();
  std::string err;

  CHECK(oe1::WritePrimaryCsv(path, recs, err));
  const std::string content = ReadAll(path);
  CHECK(content == MakeText(recs));
  // Exclusive create: a second write must fail and leave the file untouched.
  CHECK(!oe1::WritePrimaryCsv(path, MakeRecords(3, 9), err));
  CHECK(ReadAll(path) == content);
  CHECK(err.find("cannot create") != std::string::npos);

  std::vector<oe1::PrimaryRecord> back;
  oe1::ListDefects d;
  std::string sha;
  CHECK(oe1::ReadPrimaryFile(path, back, d, sha));
  CHECK_EQ_STR(sha, oe1::Sha256Hex(content));
  CHECK(back.size() == recs.size());
  oe1::ValidatePrimaryRecords(back, d);
  CHECK(d.Clean());
  CHECK(oe1::CanonicalFnv(back) == oe1::CanonicalFnv(recs));
  CHECK(!oe1::ReadPrimaryFile((dir.path / "nope.csv").string(), back, d, sha));

  const std::uint64_t canon = oe1::CanonicalFnv(recs);
  CHECK(oe1::WriteSidecar(oe1::SidecarPath(path), MakeProvenance(canon, sha), recs.size(), err));
  const std::string sidecarBytes = ReadAll(oe1::SidecarPath(path));
  CHECK(!oe1::WriteSidecar(oe1::SidecarPath(path), MakeProvenance(canon, sha), 1, err));
  CHECK(ReadAll(oe1::SidecarPath(path)) == sidecarBytes);  // refused to overwrite

  oe1::ListSidecar sc;
  CHECK(oe1::ReadSidecar(oe1::SidecarPath(path), sc, err));
  CHECK_EQ_STR(sc.format, "oe1-primarios/1");
  CHECK_EQ_STR(sc.geometryId, "geo-train-06");
  CHECK(sc.view == 0 && sc.caseCode == 6 && sc.replica == 1 && sc.n == 64);
  CHECK(sc.haveBaseRadius && sc.baseRadiusM == 509.7);
  CHECK(sc.sourceSeed == 2026092901LL);
  CHECK(sc.haveCanonFnv && sc.canonFnv == canon);
  CHECK_EQ_STR(sc.fileSha256, sha);
  {
    const nlohmann::json j = ReadJson(oe1::SidecarPath(path));
    CHECK(j.at("rng").at("key") == nlohmann::json::array({2026092901ULL, 6, 0, 0x4F455331U}));
    CHECK(j.at("rng").at("draws_per_primary") == 7);
    CHECK(j.at("start_radius_m") == 1809.7);
    CHECK(j.at("canon_fnv1a64") == oe1::Hex16(canon));
    CHECK(j.at("binary_sha256") == std::string(64, 'b'));
  }

  // Load: success.
  {
    oe1::PrimaryList pl;
    CHECK(pl.Load(path, sha, err));
    CHECK(pl.Size() == 64);
    CHECK(pl.CanonFnv() == canon);
    CHECK_EQ_STR(pl.FileSha256(), sha);
    bool same = true;
    for (std::size_t k = 0; k < 64; ++k) same = same && SameRecord(pl.At(k), recs[k]);
    CHECK(same);
    oe1::PrimaryList pl2;  // empty expected sha = no check (only the sidecar)
    CHECK(pl2.Load(path, "", err));
  }
  // Load: wrong expected SHA-256.
  {
    oe1::PrimaryList pl;
    CHECK(!pl.Load(path, std::string(64, '0'), err));
    CHECK(err.find("SHA-256 mismatch") != std::string::npos);
  }
  // Load: tampered sidecar variants.
  const nlohmann::json base = ReadJson(oe1::SidecarPath(path));
  const auto expectFail = [&](const char* what, nlohmann::json j, const char* expected) {
    WriteJson(oe1::SidecarPath(path), j);
    oe1::PrimaryList pl;
    std::string e2;
    const bool ok = pl.Load(path, sha, e2);
    ++g_checks;
    if (ok || e2.find(expected) == std::string::npos) {
      oe1test::Fail(__FILE__, __LINE__,
                    std::string("sidecar variant '") + what + "': ok=" + (ok ? "1" : "0") +
                        " err='" + e2 + "'");
    }
  };
  {
    auto j = base; j["n"] = 65;
    expectFail("n", j, "sidecar n=65");
    j = base; j["canon_fnv1a64"] = oe1::Hex16(canon ^ 1ULL);
    expectFail("canon", j, "canon_fnv1a64");
    j = base; j["file_sha256"] = std::string(64, 'f');
    expectFail("file_sha256", j, "file_sha256");
    j = base; j["format"] = "oe1-primarios/2";
    expectFail("format", j, "sidecar format");
    j = base; j.erase("geometry_id");
    expectFail("geometry_id", j, "lacks geometry_id");
    j = base; j.erase("base_radius_m");
    expectFail("base_radius_m", j, "lacks geometry_id");
    j = base; j.erase("case_code");
    expectFail("case_code", j, "lacks geometry_id");
    j = base; j["canon_fnv1a64"] = "zz";
    expectFail("canon not hex", j, "16 hex digits");
    WriteAll(oe1::SidecarPath(path), "{ not json");
    oe1::PrimaryList pl;
    CHECK(!pl.Load(path, sha, err));
    CHECK(err.find("invalid sidecar") != std::string::npos);
    fs::remove(oe1::SidecarPath(path));
    CHECK(!pl.Load(path, sha, err));
    CHECK(err.find("missing sidecar") != std::string::npos);
    WriteJson(oe1::SidecarPath(path), base);
  }
  // Load: list altered after the sidecar was written.
  {
    std::string altered = content;
    const std::size_t pos = altered.find(',', altered.find('\n')) + 1;
    altered[pos] = altered[pos] == '1' ? '2' : '1';
    WriteAll(path, altered);
    oe1::PrimaryList pl;
    CHECK(!pl.Load(path, "", err));  // sidecar file_sha256 / canon no longer match
    WriteAll(path, content);
    CHECK(pl.Load(path, sha, err));
  }
  // Load: list with defects, and an empty list, are rejected.
  {
    auto bad = recs;
    bad[10].dz = -bad[10].dz;
    const std::string p2 = WriteListAndSidecar(dir.path, "defect.csv", bad);
    oe1::PrimaryList pl;
    CHECK(!pl.Load(p2, "", err));
    CHECK(err.find("defects") != std::string::npos);
    const std::string p3 = (dir.path / "empty.csv").string();
    WriteAll(p3, std::string(oe1::kPrimaryListHeader) + "\n");
    CHECK(!pl.Load(p3, "", err));
  }
}

void TestCheckPrimaries()
{
  TempDir dirA("chkA"), dirB("chkB");
  const auto recs = MakeRecords(200, 5);
  const std::string name = "geo-train-06_v0_r2_s2026092902.primarios.csv";
  const std::string pa = WriteListAndSidecar(dirA.path, name, recs);
  const std::string pb = WriteListAndSidecar(dirB.path, name, recs);
  // Sidecars of two binaries differ only in binary_sha256; here identical.
  std::ostringstream oa, ea, ob, eb;
  CHECK(oe1::RunCheckPrimaries(pa, oa, ea) == 0);
  CHECK(oe1::RunCheckPrimaries(pb, ob, eb) == 0);
  CHECK(oa.str() == ob.str());  // identical output for the same list in another folder
  const std::string line = oa.str();
  CHECK(!line.empty() && line.back() == '\n' &&
        std::count(line.begin(), line.end(), '\n') == 1);  // exactly one summary line
  CHECK(line.rfind("OE1-CHECK-PRIMARIES file=" + name + " n=200 ", 0) == 0);
  CHECK(line.find("status=OK") != std::string::npos);
  CHECK(line.find("sidecar_fnv_match=1") != std::string::npos);
  CHECK(line.find("sidecar_sha_match=1") != std::string::npos);
  CHECK(line.find(oe1::Hex16(oe1::CanonicalFnv(recs))) != std::string::npos);
  CHECK(ea.str().empty());

  // A defect anywhere gives a non-zero status and a DEFECT line.
  {
    auto bad = recs;
    bad[50].ekin_gev = -2.0;
    const std::string pd = WriteListAndSidecar(dirA.path, "bad.csv", bad);
    std::ostringstream o, e;
    CHECK(oe1::RunCheckPrimaries(pd, o, e) == 1);
    CHECK(o.str().find("status=DEFECT") != std::string::npos);
    CHECK(o.str().find("ekin_nonpos=1") != std::string::npos);
  }
  // Missing sidecar, unreadable list.
  {
    fs::remove(oe1::SidecarPath(pa));
    std::ostringstream o, e;
    CHECK(oe1::RunCheckPrimaries(pa, o, e) == 1);
    CHECK(o.str().find("sidecar=missing") != std::string::npos);
    CHECK(o.str().find("status=DEFECT") != std::string::npos);
    std::ostringstream o2, e2;
    CHECK(oe1::RunCheckPrimaries((dirA.path / "absent.csv").string(), o2, e2) == 1);
    CHECK(o2.str().find("status=UNREADABLE") != std::string::npos);
  }
  // Sidecar that disagrees with the list content.
  {
    const std::string path = WriteListAndSidecar(dirA.path, "mismatch.csv", recs);
    auto j = ReadJson(oe1::SidecarPath(path));
    j["n"] = 199;
    fs::remove(oe1::SidecarPath(path));
    WriteJson(oe1::SidecarPath(path), j);
    std::ostringstream o, e;
    CHECK(oe1::RunCheckPrimaries(path, o, e) == 1);
    CHECK(o.str().find("status=DEFECT") != std::string::npos);
  }
}

}  // namespace

int main()
{
  struct Step {
    const char* name;
    void (*fn)();
  };
  const Step steps[] = {{"SHA-256 vectors, boundaries, file hashing", TestSha256},
                        {"FNV-1a-64, hex helpers, engine-state hash", TestHashHelpers},
                        {"to_chars / from_chars exact round trip of doubles", TestNumberRoundTrip},
                        {"canonical FNV of a list", TestCanonicalHash},
                        {"validation: every defect class is detected", TestValidation},
                        {"files: exclusive create, sidecar, PrimaryList::Load", TestFilesAndLoad},
                        {"--check-primaries summary line and status", TestCheckPrimaries}};
  for (const Step& s : steps) {
    const int before = g_fail;
    s.fn();
    std::printf("%-4s %s\n", g_fail == before ? "PASS" : "FAIL", s.name);
  }
  std::printf("test_primary_list: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
