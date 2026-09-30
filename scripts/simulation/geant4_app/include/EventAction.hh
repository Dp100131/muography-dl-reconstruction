#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4UserEventAction.hh"

#include "MuonRecord.hh"
#include "Oe1ListMode.hh"
#include "RegionCrossing.hh"

class G4Event;
class G4Material;
class G4Step;
class G4VPhysicalVolume;
class G4VProcess;

// OE1 stage 1 (P2), list mode only (worker threads). Per primary muon:
//  - BeginOfEventAction reseeds the worker engine with the key
//    {event_id, s_tr, case_code, TAG_TR} (G4Random::setTheSeeds(key, 4)), so
//    the transport stream depends only on (s_tr, case, id); records rng0 =
//    FNV-1a-64 of engine->put() right after the reseed and prim_fnv, the
//    hash of the Geant4 primary;
//  - OnPrimaryStep / OnPlane (called by SteppingAction) accumulate the
//    final state (step count, last process / status / volume, plane entry)
//    and, with --muon-log full, the chord crossings of the region Z and the
//    lateral displacement of the steps.
//    --muon-log min still runs OnPrimaryStep on every primary step: it keeps
//    the O(1) status bookkeeping there (step count, last process / step status
//    / post-step volume and energy; it is needed to classify the muons that do
//    not reach the plane) and skips only the chord / region / energy /
//    material / lateral-displacement accumulation;
//  - EndOfEventAction records rng1 and pushes the record into RunData.
// Nothing here draws random numbers or queries the navigator; engine->put()
// only reads the state.
//
// The pre->post chord of a step is NOT the path the muon followed: WentzelVI
// displaces the post-step point laterally (plan correction C1). The chord
// lengths in Z are therefore reported with the lateral displacement of the
// steps near Z (max_lat_z_mm, sum_lat_z_mm), and max_lat_mm over all steps is a
// diagnostic only.
//
// Pieces of a step of length <= 2 eps inside Z (slivers left by endpoints on a
// navigation boundary, or a tiny step deep inside Z) enter the chord-length
// sums (lz_m, ...: every piece is added, however short) but not `crossed`, the
// step / entry / material / energy fields; so crossed = 1 always implies
// n_steps_z >= 1 and a non-empty mat_in.
class EventAction : public G4UserEventAction
{
 public:
  explicit EventAction(const oe1::ListModeConfig& cfg);

  void BeginOfEventAction(const G4Event* event) override;
  void EndOfEventAction(const G4Event* event) override;

  // Every step of the primary (SteppingAction, before its early returns).
  void OnPrimaryStep(const G4Step* step);
  // Primary entering the scoring plane (SteppingAction, before the kill);
  // iAz/iEl are -1 outside the grid.
  void OnPlane(int iAz, int iEl, double ekinGeV, double elDeg, double relAzDeg);

 private:
  oe1::MatCode Classify(const G4Material* mat) const;

  const oe1::ListModeConfig& fCfg;
  // The seed key must outlive setTheSeeds(): MixMaxRng keeps the pointer.
  long fSeedKey[4] = {0, 0, 0, 0};
  bool fWorkerNoted = false;

  oe1::MuonRecord fRec;
  // Last primary step, for the final status.
  const G4VProcess* fLastProcess = nullptr;
  G4StepStatus fLastStepStatus = fUndefined;
  double fLastPostEkin = 0.0;
  const G4VPhysicalVolume* fLastPostVolume = nullptr;
  G4ThreeVector fLastPostPos;
  // Continuity of Z runs across consecutive steps (entry counting).
  oe1::EntryCounter fEntries;
};

#endif
