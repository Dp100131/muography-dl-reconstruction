#ifndef OE1_REGION_CROSSING_HH
#define OE1_REGION_CROSSING_HH

#include <algorithm>
#include <cmath>
#include <limits>

// OE1 stage 1 (P2): exact intersection of a straight chord with the 3D
// region Z = (finite z-aligned cylinder) AND (axis-aligned ellipsoid), built
// from the ORIGINAL catalog primitives in world coordinates (metres). A
// sphere (cavity) is an ellipsoid with three equal semi-axes. No Geant4
// dependency: Z is never a Geant4 volume, so computing it cannot add step
// boundaries or consume random numbers.
//
// A chord a -> b is parameterised as p(t) = a + t (b - a), t in [0, 1].
// Each primitive is convex, so its intersection with the line is one closed
// t-interval; Z is convex, so the chord meets it in exactly one interval,
// also when both chord endpoints lie outside Z.
//
// What is exact and what is not (plan correction C1). The intersection of a
// CHORD with Z is exact. The chord from the pre-step to the post-step point of
// a Geant4 step is NOT the path the particle followed: the multiple-scattering
// model (WentzelVI for muons in FTFP_BERT) displaces the post-step point
// sideways by up to 0.99 x safety, also with MuHadLateralDisplacement = false
// (that flag only controls an additional Gaussian term). Local transport
// measured a largest displacement above 1e-6 mm in about 89 % of the muons
// (median over the muons of the largest displacement: 0.7 mm in rock, about
// 60 mm in air) and up to about 10 mm next to Z. The end points of a step are
// true track positions; the path between them is unknown. Lengths and
// crossing flags computed on chords are therefore reported together with the
// lateral displacement of the steps near Z (`nearZ` below), never as exact
// properties of the simulated track.
//
// Quadratics are solved in a cancellation-free form: the discriminant uses
// the Lagrange identity (p.d)^2 - |p|^2|d|^2 = -|p x d|^2, written as
// (|d| r - |p x d|)(|d| r + |p x d|), and the roots use q = -(B/2 +
// sgn(B/2) sqrt(disc)), t1 = q/A, t2 = C/q.
//
// Tangency band eps (1 um by default): "deep" = positive length inside Z
// with every primitive shrunk by eps (radius and semi-axes minus eps, z slab
// narrowed by eps at each end); "graze" = not deep, but the chord touches Z
// grown by eps at a contact interval that stays more than 2 eps (in length)
// away from both chord endpoints. Steps that merely start or end on the
// surface of Z (a navigation boundary in variant B) are therefore not
// grazes.
//
// Near Z (`nearZ`, kNearZMarginM = 1 m): the chord meets BOTH primitives grown
// by the margin (conduit radius + M with the z slab widened by M at each end;
// every semi-axis of the ellipsoid + M). Z lies inside both primitives, so
// every chord that meets Z is near (`hit` implies `nearZ`); the converse is
// false on purpose (a chord may cross both grown primitives without touching
// Z). The margin is two orders of magnitude above the largest lateral
// displacement measured next to Z (about 10 mm), so a step whose true path
// could touch Z, or whose chord-based decision the displacement could change,
// is near; and it stays small, so only the few steps beside Z are counted.
// The test uses the chord itself (interval of each grown primitive), never its
// bounding box: a long chord through air whose box contains Z is not near.
// EventAction sums the lateral displacement of the near steps (columns
// max_lat_z_mm / sum_lat_z_mm of the per-muon log).
namespace oe1 {

// Margin (m) of the "near Z" neighbourhood, see above.
constexpr double kNearZMarginM = 1.0;

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;
};

struct CylinderPrim {  // axis parallel to z, z in [z0, z1]
  double cx = 0.0, cy = 0.0, r = 0.0, z0 = 0.0, z1 = 0.0;
};

struct EllipsoidPrim {  // axis aligned, centre (cx, cy, cz), semi-axes ax/ay/az
  double cx = 0.0, cy = 0.0, cz = 0.0, ax = 0.0, ay = 0.0, az = 0.0;
};

// Closed interval of the chord parameter; empty when !(lo <= hi).
struct TInterval {
  double lo = 1.0, hi = 0.0;
  bool Empty() const { return !(lo <= hi); }
};

struct CrossingResult {
  bool hit = false;       // nominal Z meets [0,1] (a tangent point counts)
  double lo = 0.0;        // lower t of nominal Z within [0,1] (valid when hit)
  double hi = 0.0;        // upper t of nominal Z within [0,1] (valid when hit)
  double len = 0.0;       // chord length inside nominal Z (m)
  bool bothOut = false;   // len > 0 with both endpoints outside Z (0 < lo, hi < 1)
  bool deep = false;      // positive chord length inside Z shrunk by eps
  bool graze = false;     // see header comment
  bool nearZ = false;     // chord meets both primitives grown by the near margin
  double lenCyl = 0.0;    // chord length inside the nominal cylinder (m)
  double lenEll = 0.0;    // chord length inside the nominal ellipsoid (m)
};

namespace crossing_detail {

constexpr double kInf = std::numeric_limits<double>::infinity();

inline TInterval Whole() { return {-kInf, kInf}; }
inline TInterval None() { return {kInf, -kInf}; }
inline TInterval Meet(const TInterval& a, const TInterval& b)
{
  return {std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
}
inline TInterval Unit() { return {0.0, 1.0}; }

inline double LengthOf(const TInterval& i, double chordLen)
{
  return (i.hi > i.lo) ? (i.hi - i.lo) * chordLen : 0.0;
}

// {t : A t^2 + 2 Bh t + C <= 0} for A > 0, given disc = Bh^2 - A C computed
// by the caller in a cancellation-free form.
inline TInterval QuadraticSubLevel(double A, double Bh, double C, double disc)
{
  if (!(disc >= 0.0)) return None();
  const double s = std::sqrt(disc);
  const double q = -(Bh + std::copysign(s, Bh));
  if (q == 0.0) return {0.0, 0.0};  // Bh = 0 and disc = 0 imply C = 0: t = 0
  const double t1 = q / A;
  const double t2 = C / q;
  return {std::min(t1, t2), std::max(t1, t2)};
}

}  // namespace crossing_detail

// Parameter interval (unclipped) of the line a + t d inside a cylinder.
inline TInterval CylinderInterval(const Vec3& a, const Vec3& d, const CylinderPrim& c)
{
  using namespace crossing_detail;
  if (!(c.r > 0.0) || !(c.z1 >= c.z0)) return None();
  const double px = a.x - c.cx;
  const double py = a.y - c.cy;
  const double A = d.x * d.x + d.y * d.y;
  const double C = px * px + py * py - c.r * c.r;
  TInterval radial;
  if (A == 0.0) {
    radial = (C <= 0.0) ? Whole() : None();
  }
  else {
    const double Bh = px * d.x + py * d.y;
    const double rs = c.r * std::sqrt(A);
    const double cross = std::fabs(px * d.y - py * d.x);
    radial = QuadraticSubLevel(A, Bh, C, (rs - cross) * (rs + cross));
  }
  if (radial.Empty()) return radial;
  TInterval slab;
  if (d.z == 0.0) {
    slab = (a.z >= c.z0 && a.z <= c.z1) ? Whole() : None();
  }
  else {
    const double t0 = (c.z0 - a.z) / d.z;
    const double t1 = (c.z1 - a.z) / d.z;
    slab = {std::min(t0, t1), std::max(t0, t1)};
  }
  return Meet(radial, slab);
}

// Parameter interval (unclipped) of the line a + t d inside an ellipsoid.
inline TInterval EllipsoidInterval(const Vec3& a, const Vec3& d, const EllipsoidPrim& e)
{
  using namespace crossing_detail;
  if (!(e.ax > 0.0) || !(e.ay > 0.0) || !(e.az > 0.0)) return None();
  const double ux = (a.x - e.cx) / e.ax;
  const double uy = (a.y - e.cy) / e.ay;
  const double uz = (a.z - e.cz) / e.az;
  const double wx = d.x / e.ax;
  const double wy = d.y / e.ay;
  const double wz = d.z / e.az;
  const double A = wx * wx + wy * wy + wz * wz;
  const double C = ux * ux + uy * uy + uz * uz - 1.0;
  if (A == 0.0) return (C <= 0.0) ? Whole() : None();
  const double Bh = ux * wx + uy * wy + uz * wz;
  const double kx = uy * wz - uz * wy;
  const double ky = uz * wx - ux * wz;
  const double kz = ux * wy - uy * wx;
  const double cross = std::sqrt(kx * kx + ky * ky + kz * kz);
  const double sA = std::sqrt(A);
  return QuadraticSubLevel(A, Bh, C, (sA - cross) * (sA + cross));
}

class RegionZ
{
 public:
  RegionZ() = default;

  RegionZ(const CylinderPrim& cyl, const EllipsoidPrim& ell, double eps = 1.0e-6,
          double nearMargin = kNearZMarginM)
      : fCyl(cyl), fEll(ell), fEps(eps), fNearMargin(nearMargin)
  {
    fCylIn = {cyl.cx, cyl.cy, cyl.r - eps, cyl.z0 + eps, cyl.z1 - eps};
    fCylOut = {cyl.cx, cyl.cy, cyl.r + eps, cyl.z0 - eps, cyl.z1 + eps};
    fEllIn = {ell.cx, ell.cy, ell.cz, ell.ax - eps, ell.ay - eps, ell.az - eps};
    fEllOut = {ell.cx, ell.cy, ell.cz, ell.ax + eps, ell.ay + eps, ell.az + eps};
    fCylNear = {cyl.cx, cyl.cy, cyl.r + nearMargin, cyl.z0 - nearMargin, cyl.z1 + nearMargin};
    fEllNear = {ell.cx, ell.cy, ell.cz, ell.ax + nearMargin, ell.ay + nearMargin,
                ell.az + nearMargin};
    // Conservative boxes (grown primitives plus slack) used only to skip
    // chords that cannot touch a primitive; they never change a result.
    const double slack = 2.0 * eps + 1.0e-6;
    fCylLo = {cyl.cx - cyl.r - slack, cyl.cy - cyl.r - slack, cyl.z0 - slack};
    fCylHi = {cyl.cx + cyl.r + slack, cyl.cy + cyl.r + slack, cyl.z1 + slack};
    fEllLo = {ell.cx - ell.ax - slack, ell.cy - ell.ay - slack, ell.cz - ell.az - slack};
    fEllHi = {ell.cx + ell.ax + slack, ell.cy + ell.ay + slack, ell.cz + ell.az + slack};
    fCylNearLo = {fCylNear.cx - fCylNear.r - slack, fCylNear.cy - fCylNear.r - slack,
                  fCylNear.z0 - slack};
    fCylNearHi = {fCylNear.cx + fCylNear.r + slack, fCylNear.cy + fCylNear.r + slack,
                  fCylNear.z1 + slack};
    fEllNearLo = {fEllNear.cx - fEllNear.ax - slack, fEllNear.cy - fEllNear.ay - slack,
                  fEllNear.cz - fEllNear.az - slack};
    fEllNearHi = {fEllNear.cx + fEllNear.ax + slack, fEllNear.cy + fEllNear.ay + slack,
                  fEllNear.cz + fEllNear.az + slack};
    fValid = true;
  }

  bool Valid() const { return fValid; }
  double Eps() const { return fEps; }
  double NearMargin() const { return fNearMargin; }
  const CylinderPrim& Cylinder() const { return fCyl; }
  const EllipsoidPrim& Ellipsoid() const { return fEll; }

  // Axis-aligned bounding box of Z itself (intersection of the primitives'
  // exact boxes; may be empty if the primitives are disjoint).
  void ZBox(Vec3& lo, Vec3& hi) const
  {
    lo = {std::max(fCyl.cx - fCyl.r, fEll.cx - fEll.ax),
          std::max(fCyl.cy - fCyl.r, fEll.cy - fEll.ay),
          std::max(fCyl.z0, fEll.cz - fEll.az)};
    hi = {std::min(fCyl.cx + fCyl.r, fEll.cx + fEll.ax),
          std::min(fCyl.cy + fCyl.r, fEll.cy + fEll.ay),
          std::min(fCyl.z1, fEll.cz + fEll.az)};
  }

  // True when the exact boxes of the two primitives overlap (necessary for Z
  // to be non-empty).
  bool ZBoxNonEmpty() const
  {
    Vec3 lo, hi;
    ZBox(lo, hi);
    return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z;
  }

  CrossingResult Intersect(const Vec3& a, const Vec3& b) const
  {
    using namespace crossing_detail;
    CrossingResult r;
    if (!fValid) return r;
    const Vec3 d{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 smin{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
    const Vec3 smax{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};

    // Near Z: the chord meets both primitives grown by the near margin. The
    // box test only skips the (rare) interval computation for chords that are
    // far from Z; it never changes the result.
    if (Overlaps(smin, smax, fCylNearLo, fCylNearHi) &&
        Overlaps(smin, smax, fEllNearLo, fEllNearHi)) {
      r.nearZ = !Meet(CylinderInterval(a, d, fCylNear), Unit()).Empty() &&
                !Meet(EllipsoidInterval(a, d, fEllNear), Unit()).Empty();
    }

    const bool boxCyl = Overlaps(smin, smax, fCylLo, fCylHi);
    const bool boxEll = Overlaps(smin, smax, fEllLo, fEllHi);
    if (!boxCyl && !boxEll) return r;

    const double chord = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    TInterval cyl = None();
    TInterval ell = None();
    if (boxCyl) {
      cyl = Meet(CylinderInterval(a, d, fCyl), Unit());
      r.lenCyl = LengthOf(cyl, chord);
    }
    if (boxEll) {
      ell = Meet(EllipsoidInterval(a, d, fEll), Unit());
      r.lenEll = LengthOf(ell, chord);
    }
    if (!(boxCyl && boxEll)) return r;

    const TInterval z = Meet(cyl, ell);
    if (!z.Empty()) {
      r.hit = true;
      r.lo = z.lo;
      r.hi = z.hi;
      r.len = LengthOf(z, chord);
      r.bothOut = r.len > 0.0 && z.lo > 0.0 && z.hi < 1.0;
    }

    const TInterval zin = Meet(Meet(CylinderInterval(a, d, fCylIn),
                                    EllipsoidInterval(a, d, fEllIn)),
                               Unit());
    // A zero-length chord (a point) has no length inside Z: never deep.
    r.deep = chord > 0.0 && zin.hi > zin.lo;
    if (!r.deep && chord > 0.0) {
      const TInterval zout = Meet(Meet(CylinderInterval(a, d, fCylOut),
                                       EllipsoidInterval(a, d, fEllOut)),
                                  Unit());
      const double band = 2.0 * fEps / chord;
      r.graze = !zout.Empty() && zout.lo > band && zout.hi < 1.0 - band;
    }
    return r;
  }

 private:
  static bool Overlaps(const Vec3& smin, const Vec3& smax, const Vec3& lo, const Vec3& hi)
  {
    return smin.x <= hi.x && smax.x >= lo.x && smin.y <= hi.y && smax.y >= lo.y &&
           smin.z <= hi.z && smax.z >= lo.z;
  }

  CylinderPrim fCyl, fCylIn, fCylOut, fCylNear;
  EllipsoidPrim fEll, fEllIn, fEllOut, fEllNear;
  Vec3 fCylLo, fCylHi, fEllLo, fEllHi;
  Vec3 fCylNearLo, fCylNearHi, fEllNearLo, fEllNearHi;
  double fEps = 1.0e-6;
  double fNearMargin = kNearZMarginM;
  bool fValid = false;
};

// Counts the separate entries of a track into Z from the results of its
// consecutive steps. Feed EVERY step in order. Two consecutive steps form one
// continuous run inside Z when the first ends inside Z and the second starts
// there, within joinTolM metres.
//
// minLenM (default 0) is the length below which a piece of a step is not
// information: a step whose chord is <= minLenM (a zero-length step, e.g. an
// at-rest step, stays at the same point) is transparent and leaves the state
// untouched; a step whose length inside Z is <= minLenM (a miss, or a sliver
// left by an endpoint that Geant4 placed on a navigation boundary that is also
// a surface of Z, 1e-12 m inside or outside it) ends the current run.
class EntryCounter
{
 public:
  explicit EntryCounter(double joinTolM = 1.0e-9) : fJoinTolM(joinTolM) {}

  // `chordM` = |post - pre| of the step that produced `c`. Returns true if
  // this step starts a new entry into Z.
  bool Step(const CrossingResult& c, double chordM, double minLenM = 0.0)
  {
    if (!(chordM > minLenM)) return false;
    if (!(c.len > minLenM)) {
      fPrevEndsInZ = false;
      return false;
    }
    const bool continues = fPrevEndsInZ && c.lo * chordM <= fJoinTolM;
    fPrevEndsInZ = (1.0 - c.hi) * chordM <= fJoinTolM;
    return !continues;
  }

  void Reset() { fPrevEndsInZ = false; }

 private:
  double fJoinTolM;
  bool fPrevEndsInZ = false;
};

}  // namespace oe1

#endif
