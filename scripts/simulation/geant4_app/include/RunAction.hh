#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include <cstdint>
#include <string>

#include "G4UserRunAction.hh"

#include "GeometryConfig.hh"

class G4Run;

// Master-thread RunAction writes the angular histogram (CSV) plus a JSON
// metadata sidecar in the contract shared with the surrogate pipeline
// (tools/csv_to_npz.py converts to the NPZ schema).
class RunAction : public G4UserRunAction
{
 public:
  RunAction(const GeometryConfig& cfg, int viewIndex, uint64_t seed,
            std::string outDir, int nThreads, double exposureDays);

  void BeginOfRunAction(const G4Run* run) override;
  void EndOfRunAction(const G4Run* run) override;

 private:
  GeometryConfig fCfg;
  int fViewIndex;
  uint64_t fSeed;
  std::string fOutDir;
  int fNThreads;
  double fExposureDays;
  double fWallStart = 0.0;
};

#endif
