#include "ActionInitialization.hh"

#include <cmath>

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "StackingAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(const DetectorConstruction* det,
                                           const GeometryConfig& cfg,
                                           int viewIndex, uint64_t seed,
                                           std::string outDir, int nThreads,
                                           const FluxTables* tables,
                                           double exposureDays)
    : fDet(det), fCfg(cfg), fViewIndex(viewIndex), fSeed(seed),
      fOutDir(std::move(outDir)), fNThreads(nThreads), fTables(tables),
      fExposureDays(exposureDays)
{}

void ActionInitialization::BuildForMaster() const
{
  SetUserAction(new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                              fExposureDays));
}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fViewIndex, fCfg, fTables));
  SetUserAction(new RunAction(fCfg, fViewIndex, fSeed, fOutDir, fNThreads,
                              fExposureDays));
  SetUserAction(new StackingAction());

  const double lookAzRad =
      (fViewIndex * viewpar::kViewStepDeg + 180.0) * M_PI / 180.0;
  SetUserAction(new SteppingAction(fDet, lookAzRad));
}
