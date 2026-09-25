#include "DetectorConstruction.hh"

#include <cmath>

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Ellipsoid.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Orb.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"

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
  int idx = 0;
  for (const auto& a : fCfg.anomalies) {
    const G4String name = "anomaly_" + std::to_string(idx);
    auto* mat = RockMaterial(a.density_g_cm3, name + "_mat");
    G4LogicalVolume* logVol = nullptr;
    G4ThreeVector pos;

    if (a.kind == "conduit") {
      const double halfLen = (a.z1 - a.z0) / 2 * m;
      auto* solid =
          new G4Tubs(name, 0, a.radius * m, halfLen, 0, 360 * deg);
      logVol = new G4LogicalVolume(solid, mat, name + "_log");
      pos = {a.cx * m, a.cy * m, (a.z0 + a.z1) / 2 * m - h / 2};
    }
    else if (a.kind == "cavity") {
      auto* solid = new G4Orb(name, a.radius * m);
      logVol = new G4LogicalVolume(solid, mat, name + "_log");
      pos = {a.cx * m, a.cy * m, a.cz * m - h / 2};
    }
    else {  // dense_body
      auto* solid =
          new G4Ellipsoid(name, a.ax * m, a.ay * m, a.az * m);
      logVol = new G4LogicalVolume(solid, mat, name + "_log");
      pos = {a.cx * m, a.cy * m, a.cz * m - h / 2};
    }
    new G4PVPlacement(nullptr, pos, logVol, name, coneLog, false, 0, true);
    ++idx;
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
