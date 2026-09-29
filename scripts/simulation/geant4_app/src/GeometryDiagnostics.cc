#include "GeometryDiagnostics.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "G4BooleanSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4Navigator.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Version.hh"

namespace {

// Splits arg on delim into exactly nTokens numeric fields. Returns false on
// any parse failure or wrong token count (malformed --probe/--slice-* value).
bool SplitNumbers(const std::string& arg, char delim, std::size_t nTokens,
                   std::vector<double>& out)
{
  out.clear();
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, delim)) {
    try {
      std::size_t pos = 0;
      const double v = std::stod(tok, &pos);
      if (pos != tok.size()) return false;
      out.push_back(v);
    }
    catch (const std::exception&) {
      return false;
    }
  }
  return out.size() == nTokens;
}

// Hard cap on total slice grid points (min1..max1 x min2..max2 at spec.step):
// guards the cluster's tight per-user /home quota against an accidental
// multi-hundred-MB CSV from a single --slice-y/--slice-z flag.
constexpr long kMaxSlicePoints = 2000000;

// Inclusive point count along one axis, from an integer step index rather
// than accumulated floating point, so it matches the loop in WriteSliceCsv
// exactly (see ParseSlice / WriteSliceCsv).
long AxisPointCount(double lo, double hi, double step)
{
  return static_cast<long>(std::floor((hi - lo) / step + 0.5)) + 1;
}

// Navigator used by --probe/--slice-*; lazily bound to the "world" physical
// volume, which by construction time (called after Initialize()) exists.
G4Navigator& DiagNavigator()
{
  static G4Navigator navigator;
  static bool bound = false;
  if (!bound) {
    auto* world = G4PhysicalVolumeStore::GetInstance()->GetVolume("world");
    navigator.SetWorldVolume(world);
    bound = true;
  }
  return navigator;
}

void DescribeLocation(const G4ThreeVector& pointM, std::string& volumeName,
                      std::string& materialName, double& densityGcm3)
{
  const G4ThreeVector pointInternal(pointM.x() * m, pointM.y() * m, pointM.z() * m);
  auto* pv = DiagNavigator().LocateGlobalPointAndSetup(pointInternal, nullptr, false, true);
  if (pv == nullptr) {
    volumeName = "outside-world";
    materialName = "none";
    densityGcm3 = 0.0;
    return;
  }
  volumeName = pv->GetName();
  auto* mat = pv->GetLogicalVolume()->GetMaterial();
  materialName = mat->GetName();
  densityGcm3 = mat->GetDensity() / (g / cm3);
}

// Regular inclusive grid scan over a plane; writes the shared CSV schema and
// reports the point count back to the caller.
long WriteSliceCsv(const std::string& path, const geomdiag::SliceSpec& spec,
                   const std::function<G4ThreeVector(double, double)>& toPointM)
{
  const std::filesystem::path fsPath(path);
  if (fsPath.has_parent_path()) {
    std::filesystem::create_directories(fsPath.parent_path());
  }
  std::ofstream csv(path);
  csv << "x_m,y_m,z_m,volume,material,density_g_cm3\n";

  // Grid points are computed from an integer step index, not by
  // accumulating spec.step in a float loop counter: at fine resolution over
  // a wide range, accumulated rounding error can exceed a fixed epsilon and
  // silently drop the last, documented-inclusive grid point.
  const long n1 = AxisPointCount(spec.min1, spec.max1, spec.step);
  const long n2 = AxisPointCount(spec.min2, spec.max2, spec.step);
  long n = 0;
  for (long k1 = 0; k1 < n1; ++k1) {
    const double a1 = spec.min1 + static_cast<double>(k1) * spec.step;
    for (long k2 = 0; k2 < n2; ++k2) {
      const double a2 = spec.min2 + static_cast<double>(k2) * spec.step;
      const G4ThreeVector pointM = toPointM(a1, a2);
      std::string volumeName, materialName;
      double densityGcm3 = 0.0;
      DescribeLocation(pointM, volumeName, materialName, densityGcm3);
      csv << pointM.x() << ',' << pointM.y() << ',' << pointM.z() << ','
          << volumeName << ',' << materialName << ',' << densityGcm3 << '\n';
      ++n;
    }
  }
  return n;
}

}  // namespace

namespace geomdiag {

bool ParseProbe(const std::string& arg, G4ThreeVector& outPointM)
{
  std::vector<double> v;
  if (!SplitNumbers(arg, ',', 3, v)) return false;
  outPointM.set(v[0], v[1], v[2]);
  return true;
}

bool ParseSlice(const std::string& arg, SliceSpec& outSpec)
{
  std::vector<double> v;
  if (!SplitNumbers(arg, ':', 6, v)) return false;
  if (v[2] <= v[1] || v[4] <= v[3] || v[5] <= 0.0) return false;
  const double n1 = static_cast<double>(AxisPointCount(v[1], v[2], v[5]));
  const double n2 = static_cast<double>(AxisPointCount(v[3], v[4], v[5]));
  if (n1 * n2 > static_cast<double>(kMaxSlicePoints)) {
    std::cerr << "slice too large: " << static_cast<long long>(n1 * n2)
              << " points (max " << kMaxSlicePoints
              << "); narrow the range or coarsen the step\n";
    return false;
  }
  outSpec.coord = v[0];
  outSpec.min1 = v[1];
  outSpec.max1 = v[2];
  outSpec.min2 = v[3];
  outSpec.max2 = v[4];
  outSpec.step = v[5];
  return true;
}

int RunCheckGeometry(const std::string& geometryId, int view, long seed,
                     int resolution, double tolerance, int maxErr)
{
  // CheckOverlaps() samples random points on each solid's surface via
  // G4VSolid::GetPointOnSurface(), which draws from G4UniformRand() i.e. the
  // process-wide CLHEP::HepRandom engine (see G4QuickRand.hh / the various
  // solids' GetPointOnSurface implementations under
  // $(geant4-config --prefix)/include/Geant4). That engine is the same one
  // seeded above via CLHEP::HepRandom::setTheSeed(), and it is never
  // reseeded per call, so fixing --seed makes the whole scan reproducible.
  // Verified empirically: two runs with the same --seed produce
  // byte-identical OVERLAP-CHECK lines (see T-c).
  std::cout << "OVERLAP-CHECK-HEADER resolution=" << resolution
            << " tolerance=" << tolerance << " maxErr=" << maxErr
            << " seed=" << seed << " g4version=" << G4Version
            << " geometry=" << geometryId << " view=" << view
            << " rng-note=CLHEP::HepRandom_process_engine_via_GetPointOnSurface\n";

  auto* store = G4PhysicalVolumeStore::GetInstance();
  int nVolumes = 0;
  int nOverlapping = 0;
  for (auto* pv : *store) {
    if (pv->GetMotherLogical() == nullptr) continue;
    ++nVolumes;
    const G4bool overlap = pv->CheckOverlaps(resolution, tolerance, true, maxErr);
    if (overlap) ++nOverlapping;
    std::cout << "OVERLAP-CHECK volume=" << pv->GetName()
              << " copy=" << pv->GetCopyNo()
              << " solid=" << pv->GetLogicalVolume()->GetSolid()->GetEntityType()
              << " result=" << (overlap ? "overlap" : "ok") << "\n";
  }
  std::cout << "OVERLAP-CHECK-SUMMARY geometry=" << geometryId
            << " view=" << view << " volumes=" << nVolumes
            << " overlapping=" << nOverlapping
            << " status=" << (nOverlapping > 0 ? "OVERLAPS" : "OK") << "\n";
  std::cout.flush();
  return nOverlapping > 0 ? 1 : 0;
}

void RunDumpGeometry(long volumeStatN)
{
  auto* store = G4PhysicalVolumeStore::GetInstance();
  for (auto* pv : *store) {
    auto* lv = pv->GetLogicalVolume();
    auto* solid = lv->GetSolid();
    auto* mat = lv->GetMaterial();
    const auto* motherLv = pv->GetMotherLogical();

    double volumeM3 = 0.0;
    std::string method = "analytic";
    long nStat = 0;
    double seM3 = 0.0;

    const auto* boolSolid = dynamic_cast<const G4BooleanSolid*>(solid);
    if (boolSolid != nullptr) {
      method = "mc";
      nStat = volumeStatN;
      const double vEst = solid->EstimateCubicVolume(static_cast<G4int>(volumeStatN), 0.001);
      G4ThreeVector pMin, pMax;
      solid->BoundingLimits(pMin, pMax);
      const double vBbox = (pMax.x() - pMin.x()) * (pMax.y() - pMin.y()) * (pMax.z() - pMin.z());
      const double p = vBbox > 0.0 ? vEst / vBbox : 0.0;
      volumeM3 = vEst / (m * m * m);
      seM3 = (vBbox / (m * m * m)) * std::sqrt(p * (1.0 - p) / static_cast<double>(volumeStatN));
    }
    else {
      volumeM3 = solid->GetCubicVolume() / (m * m * m);
    }

    std::cout << "DUMP-VOLUME name=" << pv->GetName()
              << " copy=" << pv->GetCopyNo()
              << " mother=" << (motherLv != nullptr ? motherLv->GetName() : G4String("none"))
              << " logical=" << lv->GetName()
              << " solid=" << solid->GetName()
              << " type=" << solid->GetEntityType()
              << " material=" << mat->GetName()
              << " density_g_cm3=" << mat->GetDensity() / (g / cm3)
              << " volume_m3=" << volumeM3
              << " method=" << method;
    if (method == "mc") {
      std::cout << " n=" << nStat << " se_m3=" << seM3;
    }
    std::cout << "\n";
  }
  std::cout.flush();
}

void RunProbe(const G4ThreeVector& pointM)
{
  std::string volumeName, materialName;
  double densityGcm3 = 0.0;
  DescribeLocation(pointM, volumeName, materialName, densityGcm3);
  std::cout << "PROBE x_m=" << pointM.x() << " y_m=" << pointM.y()
            << " z_m=" << pointM.z() << " volume=" << volumeName
            << " material=" << materialName
            << " density_g_cm3=" << densityGcm3 << "\n";
  std::cout.flush();
}

void RunSliceY(const std::string& geometryId, int view,
               const std::string& outDir, const SliceSpec& spec)
{
  // Clear the sign bit on zero so a -0.0 coord (numerically identical to
  // 0.0) does not produce a filename with a spurious "-0".
  const double coordForName = (spec.coord == 0.0) ? 0.0 : spec.coord;
  std::ostringstream nameStream;
  nameStream << outDir << "/" << geometryId << "_view" << view
             << "_slice_y" << coordForName << ".csv";
  const std::string path = nameStream.str();
  const double y = spec.coord;
  const long points = WriteSliceCsv(
      path, spec, [y](double x, double z) { return G4ThreeVector(x, y, z); });
  std::cout << "SLICE file=" << path << " points=" << points << "\n";
  std::cout.flush();
}

void RunSliceZ(const std::string& geometryId, int view,
               const std::string& outDir, const SliceSpec& spec)
{
  // Clear the sign bit on zero so a -0.0 coord (numerically identical to
  // 0.0) does not produce a filename with a spurious "-0".
  const double coordForName = (spec.coord == 0.0) ? 0.0 : spec.coord;
  std::ostringstream nameStream;
  nameStream << outDir << "/" << geometryId << "_view" << view
             << "_slice_z" << coordForName << ".csv";
  const std::string path = nameStream.str();
  const double z = spec.coord;
  const long points = WriteSliceCsv(
      path, spec, [z](double x, double y) { return G4ThreeVector(x, y, z); });
  std::cout << "SLICE file=" << path << " points=" << points << "\n";
  std::cout.flush();
}

}  // namespace geomdiag
