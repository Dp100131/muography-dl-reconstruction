// OE1 stage 1 (P2): tests C1-C9 (+extras) and V of the exact chord/region
// intersection in include/RegionCrossing.hh. Standalone (no Geant4), built
// and run by tests/build_tests.sh with the production flags (-O3 -DNDEBUG)
// and with -fsanitize=address,undefined.
//
// Region Z of geo-train-06 = conduit (cylinder) AND dense body (ellipsoid),
// original catalog primitives, world frame, metres. References used here are
// independent of the code under test: closed-form chord lengths in long
// double, the director's exact vertical-length formula
// (herramientas/validar_volumen_oe1.py:18-25), plain bisection on the
// membership predicate, and the volume computed by the director
// (83,663.70318 m^3, evidencias/volumen-independiente-geo-train-06.json).
// Test R1 (1e5 random segments against an independent numpy implementation
// and a dense-sampling oracle) lives in tests/crossing_reference.py and uses
// tests/crossing_eval.cc.
//
// Usage: test_region_crossing [--fast] [--catalog DIR]
//   --fast     coarser parallel-line grids for test V (sanitizer builds)
//   --catalog  directory with geo-train-06.json (checks BuildRegionFromConfig)
//
// Tolerances: absolute 1e-9 m for lengths and points (the plan's acceptance).
// For the reversal test C9 the plan asks 1e-12 relative; a length built from
// coordinates of size |a|, |b| cannot be reproduced better than the rounding
// of those coordinates (near-tangent chords amplify it), so the test uses
// 1e-12 * max(len) + 128 u (|a| + |b|) with u = 2^-53. The absolute accuracy
// (against a long-double evaluation of the same algorithm, worst case 1.3e-11 m
// over 1e5 segments up to 5 km long) is far inside the 1e-9 m acceptance.
// The volume test V uses 1 m^3 as in the plan; with --fast (the sanitizer build,
// coarser grids) it uses 2 m^3. These deviations from the plan's literal numbers,
// and the coordinate-scaled tolerance of R1, are listed in the
// "deviations_from_plan" entry of pruebas-cruces.json (tests/build_tests.sh).
// Test nearZ checks the "near Z" neighbourhood (1 m around both primitives) that
// selects the steps whose lateral displacement is summed (plan correction C1).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "GeometryConfig.hh"
#include "Oe1ListMode.hh"
#include "RegionCrossing.hh"
#include "check.hh"

namespace {

using oe1test::Fail;
using oe1test::g_checks;
using oe1test::g_fail;

using oe1::CrossingResult;
using oe1::CylinderPrim;
using oe1::EllipsoidPrim;
using oe1::RegionZ;
using oe1::Vec3;
using LD = long double;

constexpr CylinderPrim kCyl{20.3, -24.5, 21.0, 0.0, 340.0};
constexpr EllipsoidPrim kEll{32.2, -74.1, 137.8, 72.9, 61.6, 64.6};
constexpr double kEps = 1.0e-6;
constexpr double kAbsTol = 1.0e-9;  // plan: +-1e-9 m
constexpr double kPi = 3.14159265358979323846;
constexpr double kUnitRoundoff = 1.1102230246251565e-16;  // 2^-53

const RegionZ& Region()
{
  static const RegionZ z(kCyl, kEll, kEps);
  return z;
}

double Sq(double v) { return v * v; }
LD SqL(LD v) { return v * v; }

Vec3 Add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 Sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 Scale(const Vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 Lerp(const Vec3& a, const Vec3& b, double t) { return Add(a, Scale(Sub(b, a), t)); }
double Norm(const Vec3& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

// Independent membership predicates (long double).
bool InCyl(const Vec3& p)
{
  return SqL(p.x - kCyl.cx) + SqL(p.y - kCyl.cy) <= SqL(kCyl.r) && p.z >= kCyl.z0 &&
         p.z <= kCyl.z1;
}
bool InEll(const Vec3& p)
{
  return SqL((p.x - kEll.cx) / kEll.ax) + SqL((p.y - kEll.cy) / kEll.ay) +
             SqL((p.z - kEll.cz) / kEll.az) <=
         1.0L;
}
bool InZ(const Vec3& p) { return InCyl(p) && InEll(p); }

// Exit parameter of a segment whose start is inside Z and whose end is
// outside, by plain bisection (Z is convex, so there is one exit).
double BisectExit(const Vec3& inside, const Vec3& outside)
{
  double lo = 0.0, hi = 1.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (InZ(Lerp(inside, outside, mid))) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// The director's exact vertical-length formula, herramientas/
// validar_volumen_oe1.py:18-25, restricted to points inside both XY
// projections (as its nested quadrature is).
double VerticalOracle(double x, double y)
{
  if (Sq(x - kCyl.cx) + Sq(y - kCyl.cy) > Sq(kCyl.r)) return 0.0;
  const double ex = 1.0 - Sq((x - kEll.cx) / kEll.ax);
  const double inner = ex - Sq((y - kEll.cy) / kEll.ay);
  if (inner < 0.0) return 0.0;
  const double half = kEll.az * std::sqrt(inner);
  return std::max(0.0, std::min(kCyl.z1, kEll.cz + half) - std::max(kCyl.z0, kEll.cz - half));
}

bool Finite(const CrossingResult& r)
{
  const double v[] = {r.lo, r.hi, r.len, r.lenCyl, r.lenEll};
  for (const double x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// ---------------------------------------------------------------- C1
void TestC1()
{
  const Vec3 a{21.36, -30.71, -10.0}, b{21.36, -30.71, 400.0};
  const CrossingResult r = Region().Intersect(a, b);
  CHECK(r.hit);
  CHECK_NEAR(r.len, 89.67389128531408, kAbsTol);            // plan literal
  CHECK_NEAR(r.len, VerticalOracle(21.36, -30.71), kAbsTol);  // director formula
  CHECK(r.bothOut);
  CHECK(r.deep);
  CHECK(!r.graze);
  CHECK(r.lo > 0.0 && r.hi < 1.0);
  oe1::EntryCounter ec;
  CHECK(ec.Step(r, Norm(Sub(b, a))));  // one entry
  // Entry and exit heights against closed forms.
  const LD ex = 1.0L - SqL((21.36L - kEll.cx) / kEll.ax);
  const LD half = kEll.az * std::sqrt(ex - SqL((-30.71L - kEll.cy) / kEll.ay));
  const double zin = static_cast<double>(kEll.cz - half);
  const double zout = static_cast<double>(kEll.cz + half);
  CHECK_NEAR(-10.0 + r.lo * 410.0, zin, kAbsTol);
  CHECK_NEAR(-10.0 + r.hi * 410.0, zout, kAbsTol);
  CHECK_NEAR(r.lenCyl, 340.0, kAbsTol);
  CHECK_NEAR(r.lenEll, static_cast<double>(2.0L * half), kAbsTol);
  // The same line downwards, and a few more vertical lines against the oracle.
  const CrossingResult rr = Region().Intersect(b, a);
  CHECK_NEAR(rr.len, r.len, 1e-12);
  std::mt19937_64 gen(1);
  std::uniform_real_distribution<double> ux(-1.0, 42.0), uy(-46.0, -12.0);
  for (int i = 0; i < 20000; ++i) {
    const double x = ux(gen), y = uy(gen);
    const CrossingResult q = Region().Intersect({x, y, -50.0}, {x, y, 500.0});
    CHECK_NEAR(q.len, VerticalOracle(x, y), kAbsTol);
  }
}

// ---------------------------------------------------------------- C2
void TestC2()
{
  const double y = -30.71, z = 137.81;
  const Vec3 a{-100.0, y, z}, b{150.0, y, z};
  const CrossingResult r = Region().Intersect(a, b);
  const LD hc = std::sqrt(SqL(kCyl.r) - SqL(static_cast<LD>(y) - kCyl.cy));
  const double xin = static_cast<double>(kCyl.cx - hc), xout = static_cast<double>(kCyl.cx + hc);
  CHECK(r.hit && r.deep && r.bothOut);
  CHECK_NEAR(r.len, xout - xin, kAbsTol);
  CHECK_NEAR(r.len, 40.12161013718168, kAbsTol);  // design value
  CHECK_NEAR(-100.0 + r.lo * 250.0, xin, kAbsTol);
  CHECK_NEAR(-100.0 + r.hi * 250.0, xout, kAbsTol);
  CHECK_NEAR(-100.0 + r.lo * 250.0, 0.23919493140915904, kAbsTol);
  CHECK_NEAR(-100.0 + r.hi * 250.0, 40.36080506859084, kAbsTol);
  // Along y through the centroid, and along z (already C1).
  const double x = 21.36;
  const CrossingResult ry = Region().Intersect({x, -120.0, z}, {x, 20.0, z});
  const LD hcy = std::sqrt(SqL(kCyl.r) - SqL(static_cast<LD>(x) - kCyl.cx));
  const LD hey = kEll.ay * std::sqrt(1.0L - SqL((x - kEll.cx) / kEll.ax) -
                                 SqL((static_cast<LD>(z) - kEll.cz) / kEll.az));
  const LD ylo = std::max(kCyl.cy - hcy, kEll.cy - hey);
  const LD yhi = std::min(kCyl.cy + hcy, kEll.cy + hey);
  CHECK_NEAR(ry.len, static_cast<double>(yhi - ylo), kAbsTol);
}

// ---------------------------------------------------------------- C3
// Line along y at x = 41.3 - delta, z = 137.8: tangent to the conduit; its
// chord in the conduit lies inside the ellipsoid.
struct TangentCase {
  double delta;
  bool hit, deep, graze;
};

void TestC3()
{
  const TangentCase cases[] = {{1e-3, true, true, false},
                               {1e-7, true, false, true},
                               {-1e-7, false, false, true},
                               {-1e-3, false, false, false}};
  for (const TangentCase& c : cases) {
    const double x = kCyl.cx + kCyl.r - c.delta;
    const CrossingResult r = Region().Intersect({x, -100.0, 137.8}, {x, 50.0, 137.8});
    const LD d = c.delta;
    const double expected = c.delta > 0 ? static_cast<double>(2.0L * std::sqrt(d * (2.0L * kCyl.r - d))) : 0.0;
    CHECK_NEAR(r.len, expected, kAbsTol);
    CHECK(r.hit == c.hit);
    CHECK(r.deep == c.deep);
    CHECK(r.graze == c.graze);
    CHECK(Finite(r));
  }
  CHECK_NEAR(Region().Intersect({41.299, -100.0, 137.8}, {41.299, 50.0, 137.8}).len,
             0.40987315110913297, kAbsTol);
  CHECK_NEAR(Region().Intersect({41.2999999, -100.0, 137.8}, {41.2999999, 50.0, 137.8}).len,
             0.004098780406609269, kAbsTol);

  // Same for the ellipsoid: line along x at y = -12.5 - delta (its northern
  // apex), z = 137.8; the apex lies inside the conduit.
  for (const TangentCase& c : cases) {
    const double y = kEll.cy + kEll.ay - c.delta;
    const CrossingResult r = Region().Intersect({-100.0, y, 137.8}, {150.0, y, 137.8});
    const LD d = c.delta;
    const LD q = 1.0L - SqL((kEll.ay - d) / kEll.ay);
    const double expected = c.delta > 0 ? static_cast<double>(2.0L * kEll.ax * std::sqrt(q)) : 0.0;
    CHECK_NEAR(r.len, expected, kAbsTol);
    CHECK(r.hit == c.hit);
    CHECK(r.deep == c.deep);
    CHECK(r.graze == c.graze);
  }
  // A step that only starts on the surface of Z (a navigation boundary in
  // variant B) is not a graze: chord from a surface point going outwards.
  const double xs = kCyl.cx + kCyl.r;
  const CrossingResult edge = Region().Intersect({xs, -24.5, 137.8}, {xs + 5.0, -24.5, 137.8});
  CHECK(!edge.deep);
  CHECK(!edge.graze);
}

// ---------------------------------------------------------------- C4
void TestC4()
{
  const Vec3 a{20.3, -24.5, 260.0}, b{32.2, -120.0, 137.8};
  const CrossingResult r = Region().Intersect(a, b);
  CHECK(r.lenCyl > 0.0);
  CHECK(r.lenEll > 0.0);
  CHECK(r.len == 0.0);
  CHECK(!r.deep);
  CHECK(!r.graze);
  CHECK(!r.bothOut);
  // Cross-check both primitive lengths by dense sampling.
  const int k = 400000;
  const double chord = Norm(Sub(b, a));
  int inc = 0, ine = 0, inz = 0;
  for (int i = 0; i <= k; ++i) {
    const Vec3 p = Lerp(a, b, static_cast<double>(i) / k);
    inc += InCyl(p);
    ine += InEll(p);
    inz += InZ(p);
  }
  CHECK_NEAR(r.lenCyl, chord * inc / k, 2.0 * chord / k);
  CHECK_NEAR(r.lenEll, chord * ine / k, 2.0 * chord / k);
  CHECK(inz == 0);
}

// ---------------------------------------------------------------- C5
void TestC5()
{
  // Inside only the conduit: vertical above the dense body.
  const CrossingResult c = Region().Intersect({20.3, -24.5, 250.0}, {20.3, -24.5, 340.0});
  CHECK_NEAR(c.lenCyl, 90.0, kAbsTol);
  CHECK(c.lenEll == 0.0);
  CHECK(c.len == 0.0);
  CHECK(!c.hit && !c.deep);
  // Inside only the dense body: along x at y = -100 (outside the conduit's y range).
  const CrossingResult e = Region().Intersect({-100.0, -100.0, 137.8}, {200.0, -100.0, 137.8});
  CHECK(e.lenEll > 0.0);
  CHECK(e.lenCyl == 0.0);
  CHECK(e.len == 0.0);
  CHECK(!e.hit && !e.deep);
  const LD half = kEll.ax * std::sqrt(1.0L - SqL((-100.0L - kEll.cy) / kEll.ay));
  CHECK_NEAR(e.lenEll, static_cast<double>(2.0L * half), kAbsTol);
  // Conduit chord that stops short of the dense body.
  const CrossingResult s = Region().Intersect({20.3, -24.5, 300.0}, {20.3, -24.5, 250.0});
  CHECK(s.len == 0.0 && s.lenCyl > 0.0);
}

// ---------------------------------------------------------------- C6
void TestC6()
{
  // Fully inside: centroid +- 0.5 m.
  const Vec3 c{21.36, -30.71, 137.81};
  const Vec3 a = Add(c, {-0.3, 0.4, -0.5}), b = Add(c, {0.3, -0.4, 0.5});
  const CrossingResult r = Region().Intersect(a, b);
  CHECK(r.hit && r.deep);
  CHECK_NEAR(r.len, Norm(Sub(b, a)), kAbsTol);
  CHECK(r.lo == 0.0 && r.hi == 1.0);
  CHECK(!r.bothOut);
  oe1::EntryCounter ec;
  CHECK(ec.Step(r, Norm(Sub(b, a))));
  // Half inside: bisection reference, outgoing and incoming, several directions.
  std::mt19937_64 gen(6);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int i = 0; i < 2000; ++i) {
    Vec3 dir{nd(gen), nd(gen), nd(gen)};
    dir = Scale(dir, 1.0 / Norm(dir));
    const Vec3 outside = Add(c, Scale(dir, 400.0));
    if (InZ(outside)) continue;
    const double texit = BisectExit(c, outside);
    const CrossingResult out = Region().Intersect(c, outside);
    CHECK(out.hit && out.lo == 0.0 && !out.bothOut);
    CHECK_NEAR(out.hi, texit, kAbsTol / 400.0);
    CHECK_NEAR(out.len, texit * 400.0, kAbsTol);
    const CrossingResult in = Region().Intersect(outside, c);
    CHECK(in.hit && in.hi == 1.0);
    CHECK_NEAR(in.len, texit * 400.0, kAbsTol);
    CHECK_NEAR(in.lo, 1.0 - texit, kAbsTol / 400.0);
  }
}

// ---------------------------------------------------------------- C7
void TestC7()
{
  // Degenerate (zero-length) segments: inside Z, outside Z, on the surface.
  const Vec3 pts[] = {{21.36, -30.71, 137.81}, {0.0, 0.0, 0.0}, {41.3, -24.5, 137.8},
                      {20.3, -24.5, 340.0}, {1e3, 1e3, 1e3}};
  for (const Vec3& p : pts) {
    const CrossingResult r = Region().Intersect(p, p);
    CHECK(Finite(r));
    CHECK(r.len == 0.0);
    CHECK(!r.deep);
    CHECK(!r.graze);
    CHECK(!r.bothOut);
  }
  // Exactly axis-parallel segments (every combination of zero components).
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> ux(-5.0, 45.0), uy(-50.0, -10.0), uz(60.0, 210.0);
  for (int i = 0; i < 20000; ++i) {
    const Vec3 a{ux(gen), uy(gen), uz(gen)};
    for (int axis = 1; axis < 8; ++axis) {
      Vec3 b = a;
      if (axis & 1) b.x += 300.0;
      if (axis & 2) b.y -= 300.0;
      if (axis & 4) b.z += 300.0;
      const CrossingResult r = Region().Intersect(a, b);
      CHECK(Finite(r));
      CHECK(r.len >= 0.0 && r.len <= Norm(Sub(b, a)) + 1e-9);
      CHECK(r.len <= r.lenCyl + 1e-9 && r.len <= r.lenEll + 1e-9);
      if (i < 200) {  // dense-sampling agreement on a subset
        const int k = 20000;
        int inz = 0;
        for (int j = 0; j <= k; ++j) inz += InZ(Lerp(a, b, static_cast<double>(j) / k));
        CHECK_NEAR(r.len, Norm(Sub(b, a)) * inz / k, 2.0 * Norm(Sub(b, a)) / k);
      }
    }
  }
  // Chords lying exactly on a face of the z slab and on the cylinder axis.
  const CrossingResult top = Region().Intersect({20.3, -24.5, 340.0}, {60.0, -24.5, 340.0});
  CHECK(Finite(top));
  CHECK(top.len == 0.0);  // z = 340 is outside the ellipsoid (max z = 202.4)
  const CrossingResult axis = Region().Intersect({20.3, -24.5, 0.0}, {20.3, -24.5, 340.0});
  CHECK(Finite(axis));
  CHECK_NEAR(axis.lenCyl, 340.0, kAbsTol);
  // Non-finite input never yields a hit.
  const double quietNan = std::nan("");
  const CrossingResult bad = Region().Intersect({quietNan, 0.0, 0.0}, {1.0, 1.0, 1.0});
  CHECK(!bad.hit && bad.len == 0.0);
}

// ---------------------------------------------------------------- random segments
enum class SegKind { kBox, kShort, kLong };

std::array<Vec3, 2> RandomSegment(std::mt19937_64& gen, SegKind kind)
{
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  Vec3 zlo{}, zhi{};
  Region().ZBox(zlo, zhi);
  const double w = 20.0;
  auto inBox = [&]() {
    return Vec3{zlo.x - w + u01(gen) * (zhi.x - zlo.x + 2 * w),
                zlo.y - w + u01(gen) * (zhi.y - zlo.y + 2 * w),
                zlo.z - w + u01(gen) * (zhi.z - zlo.z + 2 * w)};
  };
  auto unitDir = [&]() {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vec3 d{nd(gen), nd(gen), nd(gen)};
    return Scale(d, 1.0 / Norm(d));
  };
  switch (kind) {
    case SegKind::kBox: {
      return {inBox(), inBox()};
    }
    case SegKind::kShort: {
      const Vec3 p = inBox();
      const double len = 0.05 + u01(gen) * 2.95;
      return {p, Add(p, Scale(unitDir(), len))};
    }
    case SegKind::kLong: {
      const Vec3 p = inBox();
      const Vec3 d = unitDir();
      const double l1 = 500.0 + u01(gen) * 2500.0, l2 = 500.0 + u01(gen) * 2500.0;
      return {Add(p, Scale(d, -l1)), Add(p, Scale(d, l2))};
    }
  }
  return {inBox(), inBox()};
}

std::array<Vec3, 2> RandomMixed(std::mt19937_64& gen)
{
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const double k = u01(gen);
  return RandomSegment(gen, k < 0.5 ? SegKind::kBox : (k < 0.8 ? SegKind::kShort : SegKind::kLong));
}

// ---------------------------------------------------------------- C8
void TestC8()
{
  // C1 split into 999 pieces at random points: sum of lengths, one entry.
  const Vec3 a{21.36, -30.71, -10.0}, b{21.36, -30.71, 400.0};
  const double whole = Region().Intersect(a, b).len;
  std::mt19937_64 gen(8);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  {
    std::vector<double> cuts(998);
    for (double& c : cuts) c = u01(gen);
    std::sort(cuts.begin(), cuts.end());
    cuts.insert(cuts.begin(), 0.0);
    cuts.push_back(1.0);
    double sum = 0.0;
    int entries = 0;
    oe1::EntryCounter ec;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
      const Vec3 p = Lerp(a, b, cuts[i]), q = Lerp(a, b, cuts[i + 1]);
      const CrossingResult r = Region().Intersect(p, q);
      sum += r.len;
      entries += ec.Step(r, Norm(Sub(q, p))) ? 1 : 0;
    }
    CHECK(cuts.size() == 1000);  // 999 pieces
    CHECK_NEAR(sum, whole, kAbsTol);
    CHECK(entries == 1);
  }
  // 1e4 random straight segments, each split into 10 pieces (9 cut points).
  int badSum = 0, badEntries = 0;
  double worst = 0.0;
  for (int n = 0; n < 10000; ++n) {
    const auto seg = RandomMixed(gen);
    const CrossingResult full = Region().Intersect(seg[0], seg[1]);
    std::array<double, 11> cut;
    cut[0] = 0.0;
    cut[10] = 1.0;
    for (int i = 1; i < 10; ++i) cut[static_cast<std::size_t>(i)] = u01(gen);
    std::sort(cut.begin(), cut.end());
    double sum = 0.0;
    int entries = 0;
    oe1::EntryCounter ec;
    for (std::size_t i = 0; i < 10; ++i) {
      const Vec3 p = Lerp(seg[0], seg[1], cut[i]), q = Lerp(seg[0], seg[1], cut[i + 1]);
      const CrossingResult r = Region().Intersect(p, q);
      sum += r.len;
      entries += ec.Step(r, Norm(Sub(q, p))) ? 1 : 0;
    }
    const double err = std::fabs(sum - full.len);
    worst = std::max(worst, err);
    if (err > kAbsTol) ++badSum;
    if (entries != (full.len > 0.0 ? 1 : 0)) ++badEntries;
  }
  CHECK(badSum == 0);
  CHECK(badEntries == 0);
  std::printf("  C8: worst |sum(pieces) - whole| over 1e4 polylines = %.3g m\n", worst);
}

// ---------------------------------------------------------------- C9
void TestC9()
{
  std::mt19937_64 gen(9);
  int bad = 0, badFlags = 0;
  double worstRel = 0.0;
  const int n = 100000;
  for (int i = 0; i < n; ++i) {
    const auto seg = RandomMixed(gen);
    const CrossingResult f = Region().Intersect(seg[0], seg[1]);
    const CrossingResult r = Region().Intersect(seg[1], seg[0]);
    const double rounding = 128.0 * kUnitRoundoff * (Norm(seg[0]) + Norm(seg[1]));
    const double tolLen = 1e-12 * std::max(f.len, r.len) + rounding;
    const double tolC = 1e-12 * std::max(f.lenCyl, r.lenCyl) + rounding;
    const double tolE = 1e-12 * std::max(f.lenEll, r.lenEll) + rounding;
    if (std::fabs(f.len - r.len) > tolLen || std::fabs(f.lenCyl - r.lenCyl) > tolC ||
        std::fabs(f.lenEll - r.lenEll) > tolE) {
      ++bad;
    }
    worstRel = std::max(worstRel, std::fabs(f.len - r.len) / std::max(tolLen, 1e-300));
    if (f.hit != r.hit || f.bothOut != r.bothOut || f.deep != r.deep || f.graze != r.graze ||
        f.nearZ != r.nearZ) {
      ++badFlags;
    }
    if (f.hit && r.hit) {
      // Reversed chord: lo' = 1 - hi, hi' = 1 - lo.
      if (std::fabs(f.lo - (1.0 - r.hi)) > 1e-12 || std::fabs(f.hi - (1.0 - r.lo)) > 1e-12) ++bad;
    }
  }
  CHECK(bad == 0);
  CHECK(badFlags == 0);
  std::printf("  C9: worst |len_fwd - len_rev| / tolerance over 1e5 segments = %.3g (must be <= 1)\n",
              worstRel);
}

// ---------------------------------------------------------------- near Z
// nearZ = the chord meets BOTH primitives grown by the near margin (1 m). It
// selects the steps whose lateral displacement is summed into max_lat_z_mm /
// sum_lat_z_mm (plan correction C1), so it must contain every chord that meets
// Z and stay a neighbourhood of Z, not of the whole edifice.
void TestNearZ()
{
  CHECK_NEAR(Region().NearMargin(), oe1::kNearZMarginM, 0.0);
  CHECK_NEAR(oe1::kNearZMarginM, 1.0, 0.0);

  // Chords through Z (C1, C2): near.
  const CrossingResult vertical = Region().Intersect({21.36, -30.71, -10.0}, {21.36, -30.71, 400.0});
  CHECK(vertical.hit && vertical.nearZ);
  CHECK(Region().Intersect({-100.0, -30.71, 137.81}, {150.0, -30.71, 137.81}).nearZ);

  // Beside the conduit wall (x = 41.3 is the tangent line): 0.5 m outside is
  // within the 1 m margin of the conduit and crosses the dense body: near but
  // not a hit; 1.5 m outside is not near.
  const CrossingResult outside05 = Region().Intersect({41.8, -100.0, 137.8}, {41.8, 50.0, 137.8});
  CHECK(!outside05.hit && outside05.len == 0.0 && !outside05.deep);
  CHECK(outside05.nearZ);
  const CrossingResult outside15 = Region().Intersect({42.8, -100.0, 137.8}, {42.8, 50.0, 137.8});
  CHECK(!outside15.hit && !outside15.nearZ);
  // Above the dense body on the vertical through the centroid: its nominal top
  // is at z = 182.64 and the top of the body grown by 1 m at z = 184.10.
  CHECK(!Region().Intersect({21.36, -30.71, 300.0}, {21.36, -30.71, 184.5}).nearZ);
  CHECK(Region().Intersect({21.36, -30.71, 300.0}, {21.36, -30.71, 183.5}).nearZ);
  // Below it: the nominal bottom is at z = 92.96 and the grown one at z = 91.50.
  CHECK(!Region().Intersect({21.36, -30.71, 91.0}, {21.36, -30.71, 60.0}).nearZ);
  CHECK(Region().Intersect({21.36, -30.71, 92.0}, {21.36, -30.71, 60.0}).nearZ);

  // Meeting both primitives without touching Z (C4) is near: nearZ contains hit.
  const CrossingResult c4 = Region().Intersect({20.3, -24.5, 260.0}, {32.2, -120.0, 137.8});
  CHECK(!c4.hit && c4.lenCyl > 0.0 && c4.lenEll > 0.0);
  CHECK(c4.nearZ);
  // Inside only one primitive, or far from both (C5 and a distant chord): not near.
  CHECK(!Region().Intersect({20.3, -24.5, 250.0}, {20.3, -24.5, 340.0}).nearZ);
  CHECK(!Region().Intersect({-100.0, -100.0, 137.8}, {200.0, -100.0, 137.8}).nearZ);
  CHECK(!Region().Intersect({500.0, 500.0, 50.0}, {600.0, 500.0, 50.0}).nearZ);
  // A long chord whose bounding box contains all of Z but that passes 31.7 m from
  // the conduit axis (the line y = x, in the plane z = 137.8) is not near: the
  // test uses the chord, never its bounding box.
  CHECK(!Region().Intersect({-1000.0, -1000.0, 137.8}, {1000.0, 1000.0, 137.8}).nearZ);
  CHECK(!Region().Intersect({-1500.0, 400.0, 137.8}, {1500.0, 400.0, 137.8}).nearZ);

  // Degenerate chords: a point inside Z is near, a distant point is not, NaN never.
  const Vec3 inside{21.36, -30.71, 137.81};
  CHECK(Region().Intersect(inside, inside).nearZ);
  CHECK(!Region().Intersect({1e3, 1e3, 1e3}, {1e3, 1e3, 1e3}).nearZ);
  CHECK(!Region().Intersect({std::nan(""), 0.0, 0.0}, {1.0, 1.0, 1.0}).nearZ);
  CHECK(!oe1::RegionZ().Intersect(inside, inside).nearZ);  // default region: nothing

  // Another margin: 1.5 m outside the wall is near with 5 m and not with 0.
  const RegionZ wide(kCyl, kEll, kEps, 5.0);
  CHECK_NEAR(wide.NearMargin(), 5.0, 0.0);
  CHECK(wide.Intersect({42.8, -100.0, 137.8}, {42.8, 50.0, 137.8}).nearZ);
  const RegionZ tight(kCyl, kEll, kEps, 0.0);
  CHECK(!tight.Intersect({41.8, -100.0, 137.8}, {41.8, 50.0, 137.8}).nearZ);
  CHECK(tight.Intersect({21.36, -30.71, -10.0}, {21.36, -30.71, 400.0}).nearZ);

  // Random segments: every chord that meets Z is near (nearZ contains hit).
  std::mt19937_64 gen(11);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  int hits = 0, hitNotNear = 0, nears = 0;
  const int total = 20000;
  for (int i = 0; i < total; ++i) {
    const auto seg = RandomMixed(gen);
    const CrossingResult r = Region().Intersect(seg[0], seg[1]);
    hits += r.hit;
    nears += r.nearZ;
    if (r.hit && !r.nearZ) ++hitNotNear;
  }
  CHECK(hitNotNear == 0);
  CHECK(nears >= hits && hits > 0);
  // Short steps that start 3 m to 30 m beyond the +x face of the conduit (x =
  // 41.3) and move away from it never come within the 1 m margin of the conduit.
  Vec3 zlo{}, zhi{};
  Region().ZBox(zlo, zhi);
  int farNear = 0;
  for (int i = 0; i < 5000; ++i) {
    const double dist = 3.0 + 27.0 * u01(gen);
    const Vec3 p{zhi.x + dist, zlo.y + (zhi.y - zlo.y) * u01(gen), zlo.z + (zhi.z - zlo.z) * u01(gen)};
    const Vec3 q{p.x + 3.0 * u01(gen), p.y + 3.0 * (u01(gen) - 0.5), p.z + 3.0 * (u01(gen) - 0.5)};
    if (Region().Intersect(p, q).nearZ) ++farNear;
  }
  CHECK(farNear == 0);
  std::printf("  nearZ: %d of %d random segments hit Z, %d are near\n", hits, total, nears);
}

// ---------------------------------------------------------------- extras
void TestExtras()
{
  // C10: a 3 km chord crossing Z equals the same chord seen as sub-chords.
  const Vec3 mid{21.36, -30.71, 137.81};
  const Vec3 dir{0.6, -0.48, 0.64};  // unit
  CHECK_NEAR(Norm(dir), 1.0, 1e-15);
  const Vec3 far1 = Add(mid, Scale(dir, -1500.0)), far2 = Add(mid, Scale(dir, 1500.0));
  const CrossingResult big = Region().Intersect(far1, far2);
  const Vec3 near1 = Add(mid, Scale(dir, -150.0)), near2 = Add(mid, Scale(dir, 150.0));
  const CrossingResult sub = Region().Intersect(near1, near2);
  CHECK(big.hit && big.bothOut && sub.bothOut);
  CHECK_NEAR(big.len, sub.len, 1e-8);

  // ZBox against the design numbers.
  Vec3 lo{}, hi{};
  Region().ZBox(lo, hi);
  CHECK_NEAR(lo.x, -0.7, 1e-12);
  CHECK_NEAR(hi.x, 41.3, 1e-12);
  CHECK_NEAR(lo.y, -45.5, 1e-12);
  CHECK_NEAR(hi.y, -12.5, 1e-12);
  CHECK_NEAR(lo.z, 73.2, 1e-12);
  CHECK_NEAR(hi.z, 202.4, 1e-12);

  // A chord far from both primitives is a clean miss.
  const CrossingResult miss = Region().Intersect({500.0, 500.0, 50.0}, {600.0, 500.0, 50.0});
  CHECK(!miss.hit && miss.len == 0.0 && miss.lenCyl == 0.0 && miss.lenEll == 0.0);

  // A default-constructed region never reports anything.
  const oe1::RegionZ none;
  CHECK(!none.Valid());
  CHECK(!none.Intersect(mid, far2).hit);

  // Sphere (cavity) as an ellipsoid with three equal semi-axes: closed form.
  GeometryConfig cfg;
  cfg.geometry_id = "synthetic";
  cfg.base_radius_m = 500.0;
  cfg.height_m = 400.0;
  AnomalyConfig conduit;
  conduit.kind = "conduit";
  conduit.cx = 0.0;
  conduit.cy = 0.0;
  conduit.radius = 10.0;
  conduit.z0 = 0.0;
  conduit.z1 = 300.0;
  AnomalyConfig cavity;
  cavity.kind = "cavity";
  cavity.cx = 0.0;
  cavity.cy = 0.0;
  cavity.cz = 100.0;
  cavity.radius = 30.0;
  cfg.anomalies = {conduit, cavity};
  RegionZ sphereZ;
  std::string err;
  CHECK(oe1::BuildRegionFromConfig(cfg, 0, 1, kEps, sphereZ, err));
  // Vertical line on the axis: sphere chord = 60 m, inside the conduit.
  const CrossingResult sv = sphereZ.Intersect({0.0, 0.0, -10.0}, {0.0, 0.0, 400.0});
  CHECK_NEAR(sv.len, 60.0, kAbsTol);
  // Vertical line at radius 10 - tiny: chord in the cylinder is the whole sphere chord.
  const CrossingResult sh = sphereZ.Intersect({-100.0, 0.0, 100.0}, {100.0, 0.0, 100.0});
  CHECK_NEAR(sh.len, 20.0, kAbsTol);  // limited by the conduit diameter
  // Rejection of invalid pairs.
  RegionZ dummy;
  CHECK(!oe1::BuildRegionFromConfig(cfg, 1, 0, kEps, dummy, err));  // first must be a conduit
  CHECK(!oe1::BuildRegionFromConfig(cfg, 0, 0, kEps, dummy, err));
  CHECK(!oe1::BuildRegionFromConfig(cfg, 0, 2, kEps, dummy, err));
}

// ---------------------------------------------------------------- EntryCounter
CrossingResult Piece(double lo, double hi, double chord)
{
  CrossingResult r;
  r.hit = true;
  r.lo = lo;
  r.hi = hi;
  r.len = (hi - lo) * chord;
  return r;
}

void TestEntryCounter()
{
  const double kSliver = 2.0e-6;  // what EventAction uses: 2 eps
  const CrossingResult miss;
  {
    // A zero-length step inside Z (at-rest step, repeated position) must not
    // split a run: one entry across [in, zero-chord, in].
    oe1::EntryCounter ec;
    CHECK(ec.Step(Piece(0.0, 1.0, 10.0), 10.0));
    CHECK(!ec.Step(miss, 0.0));
    CHECK(!ec.Step(Piece(0.0, 1.0, 10.0), 10.0));
    CHECK(!ec.Step(miss, 0.0, kSliver));
    CHECK(!ec.Step(Piece(0.0, 0.5, 10.0), 10.0));  // still the same run
  }
  {
    // A step that misses Z ends the run: the next Z step is a new entry.
    oe1::EntryCounter ec;
    CHECK(ec.Step(Piece(0.2, 0.6, 10.0), 10.0));
    CHECK(!ec.Step(miss, 5.0));
    CHECK(ec.Step(Piece(0.0, 0.5, 10.0), 10.0));
  }
  {
    // A run that leaves Z (hi < 1) is not continued by a step starting inside.
    oe1::EntryCounter ec;
    CHECK(ec.Step(Piece(0.3, 0.7, 10.0), 10.0));
    CHECK(ec.Step(Piece(0.0, 0.5, 10.0), 10.0));
  }
  {
    // Sliver at the entry: the endpoint lies 1e-13 m inside Z, then the real
    // step. With the sliver floor the sliver is ignored and the real step
    // starts the (single) entry; without the floor the sliver starts it and the
    // real step continues it: one entry either way.
    const double chordA = 100.0, chordB = 90.0;
    const CrossingResult sliverEnd = Piece(1.0 - 1e-15, 1.0, chordA);  // 1e-13 m long
    const CrossingResult real = Piece(0.0, 1.0, chordB);
    oe1::EntryCounter withFloor, noFloor;
    CHECK(!withFloor.Step(sliverEnd, chordA, kSliver));
    CHECK(withFloor.Step(real, chordB, kSliver));
    CHECK(noFloor.Step(sliverEnd, chordA));
    CHECK(!noFloor.Step(real, chordB));
  }
  {
    // Sliver at the exit: the step after the real one starts 1e-13 m inside Z.
    // With the floor it ends the run without starting an entry; without it,
    // it also continues the previous run (no extra entry).
    const CrossingResult real = Piece(0.0, 1.0, 90.0);
    const CrossingResult sliverStart = Piece(0.0, 1e-15, 100.0);
    oe1::EntryCounter withFloor;
    CHECK(withFloor.Step(real, 90.0, kSliver));
    CHECK(!withFloor.Step(sliverStart, 100.0, kSliver));
    CHECK(!withFloor.Step(miss, 50.0, kSliver));
    CHECK(withFloor.Step(Piece(0.1, 0.9, 30.0), 30.0, kSliver));  // a genuine second entry
  }
  {
    // A negligible chord is transparent even when it lies inside Z.
    oe1::EntryCounter ec;
    CHECK(ec.Step(Piece(0.0, 1.0, 10.0), 10.0, kSliver));
    CHECK(!ec.Step(Piece(0.0, 1.0, 1e-7), 1e-7, kSliver));
    CHECK(!ec.Step(Piece(0.0, 1.0, 10.0), 10.0, kSliver));
  }
  {
    // Reset() forgets the run.
    oe1::EntryCounter ec;
    CHECK(ec.Step(Piece(0.0, 1.0, 10.0), 10.0));
    ec.Reset();
    CHECK(ec.Step(Piece(0.0, 1.0, 10.0), 10.0));
  }
}

// The catalog primitives must be exactly the constants used in this test.
void TestCatalog(const std::string& dir)
{
  if (dir.empty()) {
    std::printf("  catalog: SKIP (no --catalog given)\n");
    return;
  }
  GeometryConfig cfg;
  try {
    cfg = GeometryConfig::FromJsonFile(dir + "/geo-train-06.json");
  }
  catch (const std::exception& e) {
    Fail(__FILE__, __LINE__, std::string("cannot read catalog: ") + e.what());
    return;
  }
  RegionZ z;
  std::string err;
  CHECK(oe1::BuildRegionFromConfig(cfg, 0, 1, oe1::kRegionEpsM, z, err));
  CHECK(z.Cylinder().cx == kCyl.cx && z.Cylinder().cy == kCyl.cy && z.Cylinder().r == kCyl.r);
  CHECK(z.Cylinder().z0 == kCyl.z0 && z.Cylinder().z1 == kCyl.z1);
  CHECK(z.Ellipsoid().cx == kEll.cx && z.Ellipsoid().cy == kEll.cy && z.Ellipsoid().cz == kEll.cz);
  CHECK(z.Ellipsoid().ax == kEll.ax && z.Ellipsoid().ay == kEll.ay && z.Ellipsoid().az == kEll.az);
  CHECK(cfg.geometry_id == "geo-train-06");
  try {
    const GeometryConfig g1 = GeometryConfig::FromJsonFile(dir + "/geo-train-01.json");
    CHECK(g1.anomalies.empty());
    RegionZ none;
    CHECK(!oe1::BuildRegionFromConfig(g1, 0, 1, oe1::kRegionEpsM, none, err));  // no anomalies
  }
  catch (const std::exception& e) {
    Fail(__FILE__, __LINE__, std::string("cannot read catalog: ") + e.what());
  }
}

// ---------------------------------------------------------------- V
struct Frame {
  Vec3 d, u, v;
};

Frame MakeFrame(const Vec3& dirIn)
{
  Frame f;
  f.d = Scale(dirIn, 1.0 / Norm(dirIn));
  const Vec3 helper = std::fabs(f.d.z) < 0.9 ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
  Vec3 u{f.d.y * helper.z - f.d.z * helper.y, f.d.z * helper.x - f.d.x * helper.z,
         f.d.x * helper.y - f.d.y * helper.x};
  f.u = Scale(u, 1.0 / Norm(u));
  f.v = {f.d.y * f.u.z - f.d.z * f.u.y, f.d.z * f.u.x - f.d.x * f.u.z,
         f.d.x * f.u.y - f.d.y * f.u.x};
  return f;
}

enum class Which { kZ, kCylinder, kEllipsoid };

// Volume = sum over a regular grid of parallel lines (midpoint rule) of
// (length inside the shape) * hu * hv, over the projection of the box [lo, hi]
// on the plane perpendicular to `dir` (cells hu x hv).
double GridVolume(const Vec3& dir, double hu, double hv, const Vec3& lo, const Vec3& hi,
                  Which which)
{
  const Frame f = MakeFrame(dir);
  double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
  for (int i = 0; i < 8; ++i) {
    const Vec3 c{(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z};
    const double uu = c.x * f.u.x + c.y * f.u.y + c.z * f.u.z;
    const double vv = c.x * f.v.x + c.y * f.v.y + c.z * f.v.z;
    umin = std::min(umin, uu);
    umax = std::max(umax, uu);
    vmin = std::min(vmin, vv);
    vmax = std::max(vmax, vv);
  }
  umin -= 1.0;
  umax += 1.0;
  vmin -= 1.0;
  vmax += 1.0;
  const Vec3 centre{0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
  const double reach = 0.5 * Norm(Sub(hi, lo)) + 2.0;
  const double cu = centre.x * f.u.x + centre.y * f.u.y + centre.z * f.u.z;
  const double cv = centre.x * f.v.x + centre.y * f.v.y + centre.z * f.v.z;
  const long nu = static_cast<long>(std::ceil((umax - umin) / hu));
  const long nv = static_cast<long>(std::ceil((vmax - vmin) / hv));
  double sum = 0.0;
  for (long i = 0; i < nu; ++i) {
    const double su = umin + (static_cast<double>(i) + 0.5) * hu - cu;
    for (long j = 0; j < nv; ++j) {
      const double sv = vmin + (static_cast<double>(j) + 0.5) * hv - cv;
      const Vec3 base = Add(centre, Add(Scale(f.u, su), Scale(f.v, sv)));
      const Vec3 a = Add(base, Scale(f.d, -reach)), b = Add(base, Scale(f.d, reach));
      const CrossingResult r = Region().Intersect(a, b);
      sum += (which == Which::kZ) ? r.len : (which == Which::kCylinder ? r.lenCyl : r.lenEll);
    }
  }
  return sum * hu * hv;
}

void TestVolume(bool fast)
{
  const double vZ = 83663.70318;  // director, evidencias/volumen-independiente-geo-train-06.json
  const double vCyl = kPi * kCyl.r * kCyl.r * (kCyl.z1 - kCyl.z0);          // 471050.4025
  const double vEll = 4.0 * kPi * kEll.ax * kEll.ay * kEll.az / 3.0;         // 1215148.5354
  CHECK_NEAR(vCyl, 471050.40, 0.01);
  CHECK_NEAR(vEll, 1215148.54, 0.01);

  Vec3 zlo{}, zhi{};
  Region().ZBox(zlo, zhi);
  // Plan tolerance: 1 m^3. The coarse grids of --fast (sanitizer builds) only
  // exercise the code for memory / UB errors and use 2 m^3.
  const double volTol = fast ? 2.0 : 1.0;
  const double hV = fast ? 0.02 : 0.01;
  const double hX = fast ? 0.04 : 0.02;
  const double hD = fast ? 0.04 : 0.02;
  const double hE = fast ? 0.1 : 0.05;

  // Z: vertical lines, lines along x, lines along detector -> centroid.
  const double vert = GridVolume({0.0, 0.0, -1.0}, hV, hV, zlo, zhi, Which::kZ);
  std::printf("  V: Z by vertical lines (h=%.3g)      = %.5f (director %.5f, diff %+.4f)\n", hV,
              vert, vZ, vert - vZ);
  CHECK_NEAR(vert, vZ, volTol);
  const double alongX = GridVolume({1.0, 0.0, 0.0}, hX, hX, zlo, zhi, Which::kZ);
  std::printf("  V: Z by lines along x (h=%.3g)       = %.5f (diff %+.4f)\n", hX, alongX,
              alongX - vZ);
  CHECK_NEAR(alongX, vZ, volTol);
  const Vec3 detector{1000.0, 0.0, 1.5}, centroid{21.36, -30.71, 137.81};
  const double alongDet = GridVolume(Sub(centroid, detector), hD, hD, zlo, zhi, Which::kZ);
  std::printf("  V: Z by lines detector->centroid (h=%.3g) = %.5f (diff %+.4f)\n", hD, alongDet,
              alongDet - vZ);
  CHECK_NEAR(alongDet, vZ, volTol);

  // Each primitive alone (validates the full-primitive lengths lc / le).
  // Conduit: lines along x (chord 2 sqrt(r^2 - dy^2), constant in z), cells
  // 0.005 m in y and 1 m in z (the chord does not depend on z, and the slab
  // 0..340 m is a whole number of cells, so the z sum is exact).
  const Vec3 clo{kCyl.cx - kCyl.r, kCyl.cy - kCyl.r, kCyl.z0};
  const Vec3 chi{kCyl.cx + kCyl.r, kCyl.cy + kCyl.r, kCyl.z1};
  const double cyl = GridVolume({1.0, 0.0, 0.0}, 0.005, 1.0, clo, chi, Which::kCylinder);
  std::printf("  V: conduit alone (cells 0.005 x 1)   = %.5f (exact %.5f, diff %+.4f)\n", cyl,
              vCyl, cyl - vCyl);
  CHECK_NEAR(cyl, vCyl, volTol);
  const Vec3 elo{kEll.cx - kEll.ax, kEll.cy - kEll.ay, kEll.cz - kEll.az};
  const Vec3 ehi{kEll.cx + kEll.ax, kEll.cy + kEll.ay, kEll.cz + kEll.az};
  const double ell = GridVolume({0.0, 0.0, -1.0}, hE, hE, elo, ehi, Which::kEllipsoid);
  std::printf("  V: dense body alone (h=%.3g)         = %.5f (exact %.5f, diff %+.4f)\n", hE,
              ell, vEll, ell - vEll);
  CHECK_NEAR(ell, vEll, volTol);
}

}  // namespace

int main(int argc, char** argv)
{
  bool fast = false;
  std::string catalog;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fast") fast = true;
    else if (a == "--catalog" && i + 1 < argc) catalog = argv[++i];
    else {
      std::fprintf(stderr, "usage: %s [--fast] [--catalog DIR]\n", argv[0]);
      return 2;
    }
  }
  using clock = std::chrono::steady_clock;
  struct Step {
    const char* name;
    void (*fn)();
  };
  const Step steps[] = {{"C1 vertical through the centroid", TestC1},
                        {"C2 horizontal through the centroid", TestC2},
                        {"C3 tangent to the conduit and to the dense body", TestC3},
                        {"C4 crosses both primitives, misses Z", TestC4},
                        {"C5 inside only one primitive", TestC5},
                        {"C6 fully inside / half inside", TestC6},
                        {"C7 degenerate and axis-parallel segments", TestC7},
                        {"C8 additivity of pieces and entry counting", TestC8},
                        {"C9 reversal of 1e5 segments", TestC9},
                        {"nearZ: chord meets both primitives grown by 1 m", TestNearZ},
                        {"extras: long chord, ZBox, sphere, invalid pairs", TestExtras},
                        {"EntryCounter: zero chords, misses and slivers", TestEntryCounter}};
  for (const Step& s : steps) {
    const int before = g_fail;
    const auto t0 = clock::now();
    s.fn();
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("%-4s %s (%.2f s)\n", g_fail == before ? "PASS" : "FAIL", s.name, dt);
  }
  {
    const int before = g_fail;
    const auto t0 = clock::now();
    TestCatalog(catalog);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("%-4s catalog primitives equal the test constants (%.2f s)\n",
                g_fail == before ? "PASS" : "FAIL", dt);
  }
  {
    const int before = g_fail;
    const auto t0 = clock::now();
    TestVolume(fast);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("%-4s V volume by parallel-line grids%s (%.2f s)\n",
                g_fail == before ? "PASS" : "FAIL", fast ? " [fast]" : "", dt);
  }
  std::printf("test_region_crossing: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
