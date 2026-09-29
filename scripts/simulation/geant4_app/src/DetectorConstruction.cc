#include "DetectorConstruction.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Ellipsoid.hh"
#include "G4IntersectionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Orb.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4SubtractionSolid.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "G4VSolid.hh"

DetectorConstruction::DetectorConstruction(const GeometryConfig& cfg,
                                           int viewIndex)
    : fCfg(cfg), fViewIndex(viewIndex)
{}

G4Material* DetectorConstruction::RockMaterial(double density_g_cm3,
                                               const G4String& name) const
{
  // Very low densities are numerically hostile; model them as air.
  if (density_g_cm3 < 0.05) {
    return G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  }
  // Standard-rock-like composition (SiO2) at the requested density.
  auto* nist = G4NistManager::Instance();
  auto* mat = new G4Material(name, density_g_cm3 * g / cm3, 2);
  mat->AddElement(nist->FindOrBuildElement("Si"), 1);
  mat->AddElement(nist->FindOrBuildElement("O"), 2);
  return mat;
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  auto* nist = G4NistManager::Instance();
  auto* air = nist->FindOrBuildMaterial("G4_AIR");

  // World: generous air box. Muons are injected high above the edifice on
  // backtracked lines of sight (see PrimaryGeneratorAction), so the world
  // must contain those start points.
  const double worldHalfXY =
      (viewpar::kDetDistanceM + fCfg.base_radius_m + 2500.0) * m;
  const double worldHalfZ = 4000.0 * m;
  auto* worldSolid = new G4Box("world_solid", worldHalfXY, worldHalfXY, worldHalfZ);
  auto* worldLog = new G4LogicalVolume(worldSolid, air, "world_log");
  auto* worldPhys =
      new G4PVPlacement(nullptr, {}, worldLog, "world", nullptr, false, 0, true);

  // --- Volcano cone: base radius at ground (z=0), apex at height_m.
  auto* rock = RockMaterial(fCfg.rock_density_g_cm3, "standard_rock");
  const double h = fCfg.height_m * m;
  auto* coneSolid = new G4Cons("edifice_solid",
                               0, fCfg.base_radius_m * m,   // base (at -hz)
                               0, 0.1 * m,                  // apex (at +hz)
                               h / 2, 0, 360 * deg);
  auto* coneLog = new G4LogicalVolume(coneSolid, rock, "edifice_log");
  new G4PVPlacement(nullptr, {0, 0, h / 2}, coneLog, "edifice", worldLog,
                    false, 0, true);

  // --- Anomalies: daughters of the cone volume (local frame: cone center
  //     is at z = height/2 in world, so local_z = world_z - height/2).
  //
  // Fix B1 (CSG construction): the analytic surrogate engine applies
  // anomalies in JSON order with "later wins" semantics (see
  // surrogate/geometry.py, VolcanoGeometry.density()) -- wherever two
  // anomalies overlap, the LATER one in the JSON list owns the material.
  // A plain full-volume sibling placement (the previous behaviour) instead
  // double-counts the overlap for whichever anomaly Geant4 visits last in
  // its own internal navigation order, which does not match the analytic
  // field and can silently disagree with it inside overlap regions. To
  // reproduce the analytic rule exactly, anomaly i's true region is
  // (A_i intersect edifice) minus the union of every later anomaly j > i;
  // we build that here with G4IntersectionSolid / G4SubtractionSolid.
  //
  // All raw meter quantities below (zmin/zmax/rho, base_radius_m,
  // height_m) intentionally skip the G4 unit multiplier: only ratios and
  // comparisons between them are used, and units cancel.
  const int nAnomalies = static_cast<int>(fCfg.anomalies.size());

  // Pass 1: build every primitive solid and its placement position in the
  // cone-local frame -- identical formulas to before. Also record the
  // per-anomaly horizontal half-extent and z-range needed by the
  // containment/overlap tests below. Built up front because a later
  // anomaly's primitive may be referenced by an earlier anomaly's
  // subtraction.
  std::vector<G4VSolid*> primSolid(nAnomalies, nullptr);
  std::vector<G4ThreeVector> primPos(nAnomalies);
  std::vector<double> horizExtent(nAnomalies, 0.0);
  std::vector<double> zMin(nAnomalies, 0.0), zMax(nAnomalies, 0.0);

  for (int i = 0; i < nAnomalies; ++i) {
    const auto& a = fCfg.anomalies[i];
    const G4String name = "anomaly_" + std::to_string(i);

    if (a.kind == "conduit") {
      const double halfLen = (a.z1 - a.z0) / 2 * m;
      primSolid[i] = new G4Tubs(name + "_prim", 0, a.radius * m, halfLen, 0,
                                 360 * deg);
      primPos[i] = {a.cx * m, a.cy * m, (a.z0 + a.z1) / 2 * m - h / 2};
      horizExtent[i] = a.radius;
      zMin[i] = a.z0;
      zMax[i] = a.z1;
    }
    else if (a.kind == "cavity") {
      primSolid[i] = new G4Orb(name + "_prim", a.radius * m);
      primPos[i] = {a.cx * m, a.cy * m, a.cz * m - h / 2};
      horizExtent[i] = a.radius;
      zMin[i] = a.cz - a.radius;
      zMax[i] = a.cz + a.radius;
    }
    else {  // dense_body
      primSolid[i] =
          new G4Ellipsoid(name + "_prim", a.ax * m, a.ay * m, a.az * m);
      primPos[i] = {a.cx * m, a.cy * m, a.cz * m - h / 2};
      horizExtent[i] = std::max(a.ax, a.ay);
      zMin[i] = a.cz - a.az;
      zMax[i] = a.cz + a.az;
    }
  }

  // Pass 2: apply the Boolean ops per anomaly (JSON order) and place.
  //
  // Known coverage gap: this loop supports an anomaly being BOTH clipped
  // (CLIP) and subtracted from (SUBTRACT), and supports subtracting more
  // than one later anomaly from the same anomaly, but no geometry in the
  // current 19-entry catalog exercises either combination -- every entry
  // has either clipped=yes with an empty minus list, or clipped=no with at
  // most one element in minus (verified across the full catalog via the
  // ANOMALY-CSG diagnostic line). That combined/nested path is the one
  // most likely to produce a degenerate solid (EstimateCubicVolume
  // returning a near-zero-but-nonzero result on a thin CSG sliver), so a
  // clean run against the real catalog is not evidence that this path is
  // exercised; treat it as untested until a synthetic geometry covering it
  // is added to the diagnostic suite.
  const double R = fCfg.base_radius_m;
  const double H = fCfg.height_m;

  // Clip solid used by CLIP below to bound a protruding anomaly. Reusing
  // coneSolid itself as the clip operand would make a clipped daughter's
  // boundary land exactly on the mother's own boundary wherever the
  // anomaly actually protrudes -- a documented Geant4 hazard (G4Navigator
  // tolerance ambiguity and spurious GeomNav/GeomSolids warnings on
  // coincident mother/daughter faces, e.g. a conduit's bottom cap sitting
  // on the cone's own base plane). Instead this is an exact inward offset
  // of the edifice cone: because the cone's radius is linear in z with
  // slope (R - 0.1) / H, shrinking its half-height by kClipMarginM on each
  // end and picking its two radii as shown below yields a solid whose
  // lateral radius at every shared z equals the edifice's own radius at
  // that z minus kClipMarginM exactly (both cones share the same taper
  // slope, only offset). The margin is negligible next to any anomaly's
  // extent (tens to hundreds of metres) but many orders of magnitude
  // above G4's surface tolerance (~1e-9 mm), so the clipped surface always
  // lies strictly inside the mother, never on it.
  const double kClipMarginM = 1.0e-3;  // 1 mm
  const double clipHalfHeightM = H / 2 - kClipMarginM;
  const double clipRmaxBaseM = R - (R - 0.1) * kClipMarginM / H - kClipMarginM;
  const double clipRmaxApexM =
      0.1 + (R - 0.1) * kClipMarginM / H - kClipMarginM;
  auto* clipSolid = new G4Cons("edifice_clip_solid", 0, clipRmaxBaseM * m, 0,
                                clipRmaxApexM * m, clipHalfHeightM * m, 0,
                                360 * deg);

  for (int i = 0; i < nAnomalies; ++i) {
    const auto& a = fCfg.anomalies[i];
    const G4String name = "anomaly_" + std::to_string(i);

    G4VSolid* finalSolid = primSolid[i];
    bool anyBoolean = false;
    bool clipped = false;
    std::vector<int> minusList;

    // CLIP -- sufficient (but not necessary) containment test: a "yes"
    // here can be a false negative (a contained anomaly flagged as
    // "needs clipping", in which case the intersection is a harmless
    // no-op) but must NEVER be a false positive -- a protruding anomaly
    // must never be reported as contained. rho_max upper-bounds the
    // anomaly's horizontal reach with its center-to-axis distance plus
    // its horizontal half-extent (exact for conduit/cavity, a
    // conservative box bound for the ellipsoid); R_G4(z) is the cone's
    // exact linear taper from base radius R at z=0 to apex radius 0.1 m
    // at z=H. Evaluating R_G4 at zmax gives the smallest radius the cone
    // offers anywhere in the anomaly's z-range, so rho_max <= R_G4(zmax)
    // together with zmin >= 0 and zmax <= H is sufficient for full
    // containment.
    const double rhoMax =
        std::sqrt(a.cx * a.cx + a.cy * a.cy) + horizExtent[i];
    const double rG4AtZmax = R - (R - 0.1) * zMax[i] / H;
    const bool certainlyContained =
        (zMin[i] >= 0.0) && (zMax[i] <= H) && (rhoMax <= rG4AtZmax);

    if (!certainlyContained) {
      finalSolid = new G4IntersectionSolid(name + "_in_edifice", finalSolid,
                                            clipSolid, nullptr, -primPos[i]);
      clipped = true;
      anyBoolean = true;
    }

    // SUBTRACT -- remove every LATER anomaly j > i whose axis-aligned
    // bounding box overlaps anomaly i's box (inclusive comparison). A
    // disjoint box means anomaly j cannot touch anomaly i's region, so
    // skipping the subtraction there is exact, not an approximation --
    // it is precisely the set of operations that could not change the
    // solid. Whatever material this loop still leaves inside the final
    // solid after all subtractions is exactly what the B1 rule assigns
    // to anomaly i.
    //
    // The gate itself is an AABB test, not an exact-shape test, so a
    // rarer case -- two anomalies whose boxes overlap while their real
    // shapes do not (e.g. two disks whose bounding squares touch at a
    // corner) -- still produces a real (no-op) G4SubtractionSolid node.
    // That only ever adds a redundant Boolean evaluation, never a wrong
    // result (an AABB overlap is necessary for shape overlap, so this
    // test can over-subtract but never under-subtract); it is accepted
    // here as a deliberate simplicity/performance tradeoff over an
    // exact per-kind (circle/ellipse) distance test.
    for (int j = i + 1; j < nAnomalies; ++j) {
      const auto& aj = fCfg.anomalies[j];
      const bool overlapX = (a.cx - horizExtent[i] <= aj.cx + horizExtent[j])
                             && (aj.cx - horizExtent[j] <= a.cx + horizExtent[i]);
      const bool overlapY = (a.cy - horizExtent[i] <= aj.cy + horizExtent[j])
                             && (aj.cy - horizExtent[j] <= a.cy + horizExtent[i]);
      const bool overlapZ = (zMin[i] <= zMax[j]) && (zMin[j] <= zMax[i]);
      if (!(overlapX && overlapY && overlapZ)) continue;

      const G4String stepName = name + "_minus_" + std::to_string(j);
      finalSolid = new G4SubtractionSolid(stepName, finalSolid, primSolid[j],
                                           nullptr, primPos[j] - primPos[i]);
      minusList.push_back(j);
      anyBoolean = true;
    }

    // EMPTY -- only relevant once a Boolean op has actually run: a
    // z-range that cannot reach the cone at all makes the intersection
    // (or a subtraction chain built on top of it) certainly empty; a
    // near-zero Monte Carlo volume estimate catches the remaining
    // degenerate cases (e.g. a subtraction that consumes the whole
    // solid). Skip placement and log a warning rather than register a
    // degenerate PV/LV.
    bool placed = true;
    if (anyBoolean) {
      const bool certainlyOutsideZ = (zMin[i] >= H) || (zMax[i] <= 0.0);
      const double estVolume =
          certainlyOutsideZ ? 0.0 : finalSolid->EstimateCubicVolume(1000000, 0.001);
      if (certainlyOutsideZ || estVolume == 0.0) placed = false;
    }

    if (placed) {
      auto* mat = RockMaterial(a.density_g_cm3, name + "_mat");
      auto* logVol = new G4LogicalVolume(finalSolid, mat, name + "_log");
      new G4PVPlacement(nullptr, primPos[i], logVol, name, coneLog, false, 0,
                         true);
    }

    G4String minusStr = "[";
    for (size_t k = 0; k < minusList.size(); ++k) {
      if (k > 0) minusStr += ",";
      minusStr += std::to_string(minusList[k]);
    }
    minusStr += "]";
    G4cout << "ANOMALY-CSG name=" << name << " kind=" << a.kind
           << " clipped=" << (clipped ? "yes" : "no")
           << " minus=" << minusStr
           << " solid=" << finalSolid->GetEntityType()
           << " placed=" << (placed ? "yes" : "no") << G4endl;
  }

  // --- Scoring plane: LARGE thin vertical box facing the cone axis.
  //
  // The physical detector area (1.44 m^2) enters the normalization through
  // the injection step (uniform impact points over the REAL panel). But
  // multiple scattering along km of air/rock displaces a few-GeV muon by
  // tens of meters at arrival — a panel-sized scorer would only catch the
  // ultra-energetic tail (measured: <2% efficiency, mean Ekin ~94 GeV).
  // A large scoring plane keeps the DIRECTIONAL flux estimator unbiased:
  // muons scattered off the panel are compensated by neighbors scattered
  // onto it. Arrival direction (what we bin) is what a real tracker
  // measures anyway.
  // Convergence series (2M events, seed 777, geo-train-01 view 0):
  //   half=50m: 64.2% detected | half=150m: 73.07% | half=300m: 73.81%
  // -> 600x600 m sits <0.5% from the asymptote; adopted for production.
  const double kScoringHalfM = 300.0;
  const double viewAz = fViewIndex * viewpar::kViewStepDeg * deg;
  const G4ThreeVector detPos(viewpar::kDetDistanceM * m * std::cos(viewAz),
                             viewpar::kDetDistanceM * m * std::sin(viewAz),
                             viewpar::kDetHeightM * m);
  auto* panelSolid = new G4Box("detector_solid",
                               0.5 * cm,                 // thickness (x, local)
                               kScoringHalfM * m,        // width
                               kScoringHalfM * m);       // height
  fDetectorPanel = new G4LogicalVolume(panelSolid, air, "detector_log");
  // Rotate so the panel normal (local x) points toward the cone axis.
  auto* rot = new G4RotationMatrix();
  rot->rotateZ(-viewAz);  // passive rotation for G4PVPlacement
  new G4PVPlacement(rot, detPos, fDetectorPanel, "detector", worldLog,
                    false, 0, true);

  return worldPhys;
}
