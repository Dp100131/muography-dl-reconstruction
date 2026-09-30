#include "SteppingAction.hh"

#include <cmath>

#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "RunData.hh"

namespace {
constexpr double kDeg = M_PI / 180.0;

double wrapAngle(double a)
{
  while (a > M_PI) a -= 2 * M_PI;
  while (a <= -M_PI) a += 2 * M_PI;
  return a;
}

// Bin index matching numpy.linspace(min, max, N) CENTERS as used by the
// surrogate: spacing = (max-min)/(N-1), idx = round((v-min)/spacing).
int binIndex(double v, double vmin, double vmax, int n)
{
  const double spacing = (vmax - vmin) / (n - 1);
  const int idx = static_cast<int>(std::lround((v - vmin) / spacing));
  return (idx >= 0 && idx < n) ? idx : -1;
}
}  // namespace

SteppingAction::SteppingAction(const DetectorConstruction* det,
                               double lookAzRad)
    : fDet(det), fLookAzRad(lookAzRad)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  auto* track = step->GetTrack();
  if (track->GetParentID() != 0) return;  // primaries only (belt & braces)
  if (fEv != nullptr) fEv->OnPrimaryStep(step);  // OE1 list mode only

  const auto* post = step->GetPostStepPoint();
  const auto* volume = post->GetTouchableHandle()->GetVolume();
  if (volume == nullptr) return;
  if (volume->GetLogicalVolume() != fDet->DetectorPanel()) return;

  // Arrival direction: where the muon came FROM (z > 0 for downward travel).
  const G4ThreeVector from = -post->GetMomentumDirection();
  const double elev = std::asin(std::min(1.0, std::max(-1.0, from.z())));
  const double relAz =
      wrapAngle(std::atan2(from.y(), from.x()) - fLookAzRad);

  const int iAz = binIndex(relAz / kDeg, -viewpar::kAzHalfDeg,
                           viewpar::kAzHalfDeg, viewpar::kNAz);
  const int iEl = binIndex(elev / kDeg, viewpar::kElMinDeg,
                           viewpar::kElMaxDeg, viewpar::kNEl);
  if (iAz >= 0 && iEl >= 0) {
    RunData::Instance().RecordHit(iAz, iEl,
                                  post->GetKineticEnergy() / GeV);
  }
  if (fEv != nullptr) {  // OE1 list mode only: same integers and energy as above
    fEv->OnPlane(iAz, iEl, post->GetKineticEnergy() / GeV, elev / kDeg, relAz / kDeg);
  }
  track->SetTrackStatus(fStopAndKill);  // job done for this primary
}
