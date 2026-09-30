#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include <cstdint>
#include <string>
#include <vector>

#include "G4UserRunAction.hh"

#include "GeometryConfig.hh"

class G4Run;
namespace oe1 {
struct ListModeConfig;

// OE1 stage 1 (P2): "name:SolidType:Material" of every physical volume of the
// geometry that is built, sorted (SolidType = G4VSolid::GetEntityType()). It
// shows which variant was built (in B anomaly_0 is a G4SubtractionSolid, in A a
// G4Tubs). Defined in RunAction.cc; main() calls it right after
// runManager->Initialize() (before BeamOn) and RunAction at the end of the run.
std::vector<std::string> BuiltGeometrySignature();
}  // namespace oe1

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

  // OE1 stage 1 (P2): list-mode outputs (per-muon log, meta key
  // "oe1_etapa1", checks) on the master. Null (the default) = legacy run.
  void SetListMode(const oe1::ListModeConfig* lm) { fLm = lm; }

 private:
  GeometryConfig fCfg;
  int fViewIndex;
  uint64_t fSeed;
  std::string fOutDir;
  int fNThreads;
  double fExposureDays;
  double fWallStart = 0.0;
  const oe1::ListModeConfig* fLm = nullptr;
};

#endif
