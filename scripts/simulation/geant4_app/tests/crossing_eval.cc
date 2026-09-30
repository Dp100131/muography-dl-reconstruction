// OE1 stage 1 (P2): command-line evaluator of the exact chord/region
// intersection (include/RegionCrossing.hh), used to compare the production
// detector with an independent numpy implementation (tests/crossing_reference.py).
//
// Usage:
//   crossing_eval [--geometry CATALOG.json --pair I,J] [--eps E] SEGMENTS.csv
//   crossing_eval ... -            (segments on standard input)
//
// Without --geometry the primitives of geo-train-06 are built in (conduit
// centre (20.3, -24.5), r = 21, z in [0, 340]; dense body centre
// (32.2, -74.1, 137.8), semi-axes 72.9 / 61.6 / 64.6, all in metres).
//
// SEGMENTS.csv: one segment per line, "ax,ay,az,bx,by,bz" in metres (world
// frame). Blank lines and lines starting with a letter or '#' are skipped.
// Output (standard output): the header "lo,hi,len,deep,graze,lc,le,near" and one
// row per segment. Doubles use std::to_chars (shortest round trip). lo / hi
// are the chord parameters of Z within [0, 1] (nan when the chord misses Z),
// len is the chord length inside Z (m), lc / le the lengths inside the bare
// cylinder / ellipsoid (m), deep / graze / near are 0 or 1 (near = the chord
// meets both primitives grown by oe1::kNearZMarginM, see RegionCrossing.hh).
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

#include "GeometryConfig.hh"
#include "Oe1ListMode.hh"
#include "RegionCrossing.hh"

namespace {

void AppendDouble(std::string& out, double v)
{
  if (std::isnan(v)) {
    out += "nan";
    return;
  }
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);
  out.append(buf, res.ptr);
}

bool ParseSix(std::string_view line, double (&v)[6])
{
  std::size_t start = 0;
  int n = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      if (n >= 6) return false;
      const std::string_view f = line.substr(start, i - start);
      const auto res = std::from_chars(f.data(), f.data() + f.size(), v[n]);
      if (f.empty() || res.ec != std::errc() || res.ptr != f.data() + f.size()) return false;
      ++n;
      start = i + 1;
    }
  }
  return n == 6;
}

int Usage()
{
  std::cerr << "usage: crossing_eval [--geometry CATALOG.json --pair I,J] [--eps E] "
               "SEGMENTS.csv|-\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string geometry, input;
  int pi = 0, pj = 1;
  double eps = oe1::kRegionEpsM;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--geometry" && i + 1 < argc) {
      geometry = argv[++i];
    }
    else if (a == "--pair" && i + 1 < argc) {
      if (std::sscanf(argv[++i], "%d,%d", &pi, &pj) != 2) return Usage();
    }
    else if (a == "--eps" && i + 1 < argc) {
      eps = std::atof(argv[++i]);
      if (!(eps >= 0.0)) return Usage();
    }
    else if (input.empty() && (a == "-" || a.rfind("--", 0) != 0)) {
      input = a;
    }
    else {
      return Usage();
    }
  }
  if (input.empty()) return Usage();

  oe1::RegionZ region;
  if (geometry.empty()) {
    region = oe1::RegionZ(oe1::CylinderPrim{20.3, -24.5, 21.0, 0.0, 340.0},
                          oe1::EllipsoidPrim{32.2, -74.1, 137.8, 72.9, 61.6, 64.6}, eps);
  }
  else {
    try {
      const GeometryConfig cfg = GeometryConfig::FromJsonFile(geometry);
      std::string err;
      if (!oe1::BuildRegionFromConfig(cfg, pi, pj, eps, region, err)) {
        std::cerr << err << "\n";
        return 2;
      }
    }
    catch (const std::exception& e) {
      std::cerr << "geometry error: " << e.what() << "\n";
      return 1;
    }
  }

  std::ifstream file;
  std::istream* in = &std::cin;
  if (input != "-") {
    file.open(input);
    if (!file) {
      std::cerr << "cannot open " << input << "\n";
      return 1;
    }
    in = &file;
  }

  std::ios::sync_with_stdio(false);
  std::string out = "lo,hi,len,deep,graze,lc,le,near\n";
  std::string line;
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::size_t nrows = 0;
  while (std::getline(*in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#' || std::isalpha(static_cast<unsigned char>(line[0])))
      continue;
    double v[6];
    if (!ParseSix(line, v)) {
      std::cerr << "bad segment line " << (nrows + 1) << ": " << line << "\n";
      return 1;
    }
    const oe1::CrossingResult r =
        region.Intersect(oe1::Vec3{v[0], v[1], v[2]}, oe1::Vec3{v[3], v[4], v[5]});
    AppendDouble(out, r.hit ? r.lo : kNaN);
    out.push_back(',');
    AppendDouble(out, r.hit ? r.hi : kNaN);
    out.push_back(',');
    AppendDouble(out, r.len);
    out += r.deep ? ",1" : ",0";
    out += r.graze ? ",1," : ",0,";
    AppendDouble(out, r.lenCyl);
    out.push_back(',');
    AppendDouble(out, r.lenEll);
    out += r.nearZ ? ",1\n" : ",0\n";
    ++nrows;
    if (out.size() > (1U << 20)) {
      std::fwrite(out.data(), 1, out.size(), stdout);
      out.clear();
    }
  }
  std::fwrite(out.data(), 1, out.size(), stdout);
  return std::fflush(stdout) == 0 ? 0 : 1;
}
