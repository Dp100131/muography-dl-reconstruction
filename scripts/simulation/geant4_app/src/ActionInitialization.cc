#include "ActionInitialization.hh"

#include <cmath>

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "Oe1ListMode.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "StackingAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(const DetectorConstruction* det,
                                           const GeometryConfig& cfg,
                                           int viewIndex, uint64_t seed,
                                           std::string outDir, int nThreads,
                                           const FluxTables* tables,
                                           double exposureDays,
                                           const oe1::ListModeConfig* listMode)
    : fDet(det), fCfg(cfg), fViewIndex(viewIndex), fSeed(seed),
      fOutDir(std::move(outDir)), fNThreads(nThreads), fTables(tables),
      fExposureDays(exposureDays), fLm(listMode)
{}

void ActionInitialization::BuildForMaster() const
{
  if (fLm != nullptr) {  // OE1 list mode: the master RunAction writes the log
    auto* run = new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                              fExposureDays);
    run->SetListMode(fLm);
    SetUserAction(run);
    return;
  }
  SetUserAction(new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                              fExposureDays));
}

void ActionInitialization::Build() const
{
  if (fLm != nullptr) {  // OE1 list mode
    BuildListMode();
    return;
  }
  SetUserAction(new PrimaryGeneratorAction(fViewIndex, fCfg, fTables));
  SetUserAction(new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                              fExposureDays));
  SetUserAction(new StackingAction());

  const double lookAzRad =
      (fViewIndex * viewpar::kViewStepDeg + 180.0) * M_PI / 180.0;
  SetUserAction(new SteppingAction(fDet, lookAzRad));
}

void ActionInitialization::BuildListMode() const
{
  // Same actions as Build(), plus the per-muon EventAction; the generator
  // replays the list instead of sampling.
  auto* generator = new PrimaryGeneratorAction(fViewIndex, fCfg, fTables);
  generator->SetPrimaryList(fLm->list);
  SetUserAction(generator);
  auto* run = new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                            fExposureDays);
  run->SetListMode(fLm);
  SetUserAction(run);
  SetUserAction(new StackingAction());
  auto* event = new EventAction(*fLm);
  SetUserAction(event);

  const double lookAzRad =
      (fViewIndex * viewpar::kViewStepDeg + 180.0) * M_PI / 180.0;
  auto* stepping = new SteppingAction(fDet, lookAzRad);
  stepping->SetEventAction(event);
  SetUserAction(stepping);
}
