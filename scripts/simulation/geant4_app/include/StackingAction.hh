#ifndef STACKING_ACTION_HH
#define STACKING_ACTION_HH

#include "G4UserStackingAction.hh"

// Kills every secondary at creation. Transmission muography only asks
// whether the PRIMARY muon crosses the edifice and reaches the panel;
// tracking delta rays / EM showers in km of rock would multiply the CPU
// cost for zero signal. Documented in the run metadata (physics_note).
class StackingAction : public G4UserStackingAction
{
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override;
};

#endif
