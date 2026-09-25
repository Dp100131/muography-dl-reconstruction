#include "PrimaryGeneratorAction.hh"

#include <cmath>

#include "G4Event.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include "RunData.hh"

namespace {
constexpr double kDeg = M_PI / 180.0;
}

PrimaryGeneratorAction::PrimaryGeneratorAction(int viewIndex,
                                               const GeometryConfig& cfg,
                                               const FluxTables* tables)
    : fTables(tables)
{
  fGun = new G4ParticleGun(1);
  fLookAzRad = (viewIndex * viewpar::kViewStepDeg + 180.0) * kDeg;
  // Start beyond the far rim of the edifice, with margin.
  fStartRadiusM = viewpar::kDetDistanceM + cfg.base_radius_m + 300.0;
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  RunData::Instance().RecordGenerated();

  // --- Direction: protocol-weighted elevation row + uniform azimuth.
  const int row = fTables->SampleRow(G4UniformRand());
  const double elevDeg = fTables->RowElevationDeg(row) +
      (2.0 * G4UniformRand() - 1.0) * fTables->RowHalfStepDeg();
  const double relAz =
      (2.0 * G4UniformRand() - 1.0) * viewpar::kAzHalfDeg * kDeg;

  const double elev = elevDeg * kDeg;
  const double azWorld = fLookAzRad + relAz;
  // Arrival "from" direction (z > 0); travel direction is its opposite.
  const G4ThreeVector from(std::cos(elev) * std::cos(azWorld),
                           std::cos(elev) * std::sin(azWorld),
                           std::sin(elev));
  const G4ThreeVector arrivalDir = -from;

  // --- Momentum from the real-EcoMug conditional spectrum.
  const double pTot = fTables->SampleMomentum(elevDeg, G4UniformRand());

  // --- Impact point: uniform over the physical panel; inject backtracked.
  const double viewAz = fLookAzRad - M_PI;  // detector's world azimuth
  const G4ThreeVector detCenter(
      viewpar::kDetDistanceM * m * std::cos(viewAz),
      viewpar::kDetDistanceM * m * std::sin(viewAz),
      viewpar::kDetHeightM * m);
  const G4ThreeVector panelY(-std::sin(viewAz), std::cos(viewAz), 0.0);
  const G4ThreeVector panelZ(0.0, 0.0, 1.0);
  const double half = viewpar::kDetSideM / 2 * m;
  const G4ThreeVector impact = detCenter
      + panelY * ((2.0 * G4UniformRand() - 1.0) * half)
      + panelZ * ((2.0 * G4UniformRand() - 1.0) * half);
  const double pathLen = (fStartRadiusM / std::cos(elev)) * m;
  const G4ThreeVector start = impact - arrivalDir * pathLen;

  fGun->SetParticlePosition(start);
  G4ParticleDefinition* particle =
      fTables->SampleChargeIsPositive(G4UniformRand())
          ? static_cast<G4ParticleDefinition*>(G4MuonPlus::Definition())
          : static_cast<G4ParticleDefinition*>(G4MuonMinus::Definition());
  fGun->SetParticleDefinition(particle);
  fGun->SetParticleMomentumDirection(arrivalDir);
  // Kinetic energy set explicitly (momentum tables are GeV/c) — avoids the
  // per-event G4ParticleGun redefinition warning (GBs of log at 1e8 events).
  constexpr double kMuonMassGeV = 0.1056583745;
  const double ekinGeV =
      std::sqrt(pTot * pTot + kMuonMassGeV * kMuonMassGeV) - kMuonMassGeV;
  fGun->SetParticleEnergy(ekinGeV * GeV);
  fGun->GeneratePrimaryVertex(event);
}
