#ifndef OE1_TESTS_LIST_SUPPORT_HH
#define OE1_TESTS_LIST_SUPPORT_HH

// Helpers shared by the standalone tests that need primary lists on disk.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "PrimaryList.hh"
#include "Sha256.hh"
#include "check.hh"

namespace oe1test {

namespace fs = std::filesystem;

struct TempDir {
  fs::path path;
  explicit TempDir(const std::string& tag)
  {
    std::random_device rd;
    path = fs::temp_directory_path() /
           ("oe1_test_" + tag + "_" + std::to_string(rd()) + "_" + std::to_string(rd()));
    fs::create_directories(path);
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

inline std::string ReadAll(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline void WriteAll(const fs::path& p, const std::string& s)
{
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << s;
}

inline std::uint64_t Bits(double v)
{
  std::uint64_t b = 0;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

inline bool SameRecord(const oe1::PrimaryRecord& a, const oe1::PrimaryRecord& b)
{
  return a.id == b.id && Bits(a.x_m) == Bits(b.x_m) && Bits(a.y_m) == Bits(b.y_m) &&
         Bits(a.z_m) == Bits(b.z_m) && Bits(a.dx) == Bits(b.dx) && Bits(a.dy) == Bits(b.dy) &&
         Bits(a.dz) == Bits(b.dz) && Bits(a.ekin_gev) == Bits(b.ekin_gev) && a.pdg == b.pdg &&
         a.charge_e == b.charge_e;
}

// Valid list: downward unit directions, positive energies, unique states.
inline std::vector<oe1::PrimaryRecord> MakeRecords(std::size_t n, std::uint64_t seed)
{
  constexpr double kPi = 3.14159265358979323846;
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<oe1::PrimaryRecord> recs(n);
  for (std::size_t k = 0; k < n; ++k) {
    oe1::PrimaryRecord& r = recs[k];
    r.id = static_cast<std::int64_t>(k);
    r.x_m = 1809.7 * (2.0 * u(gen) - 1.0);
    r.y_m = 1809.7 * (2.0 * u(gen) - 1.0);
    r.z_m = 400.0 + 900.0 * u(gen);
    const double el = (2.0 + 45.0 * u(gen)) * kPi / 180.0;
    const double az = 2.0 * kPi * u(gen);
    const double dx = -std::cos(el) * std::cos(az), dy = -std::cos(el) * std::sin(az),
                 dz = -std::sin(el);
    const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    r.dx = dx / norm;
    r.dy = dy / norm;
    r.dz = dz / norm;
    r.ekin_gev = 0.1 + 300.0 * u(gen) * u(gen) * u(gen);
    const bool plus = u(gen) < 0.5613;
    r.pdg = plus ? -13 : 13;
    r.charge_e = plus ? 1 : -1;
  }
  return recs;
}

inline oe1::ListProvenance MakeProvenance(std::uint64_t canon, const std::string& fileSha)
{
  oe1::ListProvenance p;
  p.geometryId = "geo-train-06";
  p.view = 0;
  p.baseRadiusM = 509.7;
  p.startRadiusM = 1809.7;
  p.caseCode = 6;
  p.replica = 1;
  p.sourceSeed = 2026092901ULL;
  p.sourceTag = 0x4F455331U;
  p.drawsPerPrimary = 7;
  p.canonFnv = canon;
  p.fileSha256 = fileSha;
  p.binarySha256 = std::string(64, 'b');
  return p;
}

// Writes list + sidecar for `recs` into `dir`; returns the list path.
inline std::string WriteListAndSidecar(const fs::path& dir, const std::string& name,
                                       const std::vector<oe1::PrimaryRecord>& recs)
{
  const std::string path = (dir / name).string();
  std::string err;
  CHECK(oe1::WritePrimaryCsv(path, recs, err));
  std::vector<oe1::PrimaryRecord> back;
  oe1::ListDefects d;
  std::string sha;
  CHECK(oe1::ReadPrimaryFile(path, back, d, sha));
  CHECK(oe1::WriteSidecar(oe1::SidecarPath(path),
                          MakeProvenance(oe1::CanonicalFnv(recs), sha), recs.size(), err));
  return path;
}

}  // namespace oe1test

#endif
