#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

class DetectorConstruction;
class EventAction;

// Records the primary muon's entry into the detector panel: bins the arrival
// direction into the shared 256x256 angular grid and kills the track.
class SteppingAction : public G4UserSteppingAction
{
 public:
  SteppingAction(const DetectorConstruction* det, double lookAzRad);

  void UserSteppingAction(const G4Step* step) override;

  // OE1 stage 1 (P2): list-mode per-muon hooks; stays null in legacy runs.
  void SetEventAction(EventAction* ev) { fEv = ev; }

 private:
  const DetectorConstruction* fDet;
  double fLookAzRad;
  EventAction* fEv = nullptr;
};

#endif
