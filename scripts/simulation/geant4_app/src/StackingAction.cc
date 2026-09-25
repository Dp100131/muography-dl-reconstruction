#include "StackingAction.hh"

#include "G4Track.hh"

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(
    const G4Track* track)
{
  return track->GetParentID() == 0 ? fUrgent : fKill;
}
