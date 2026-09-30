#ifndef ACTION_INITIALIZATION_HH
#define ACTION_INITIALIZATION_HH

#include <cstdint>
#include <string>

#include "G4VUserActionInitialization.hh"

#include "FluxTables.hh"
#include "GeometryConfig.hh"

class DetectorConstruction;
namespace oe1 {
struct ListModeConfig;
}

class ActionInitialization : public G4VUserActionInitialization
{
 public:
  ActionInitialization(const DetectorConstruction* det,
                       const GeometryConfig& cfg, int viewIndex,
                       uint64_t seed, std::string outDir, int nThreads,
                       const FluxTables* tables, double exposureDays,
                       const oe1::ListModeConfig* listMode = nullptr);

  void BuildForMaster() const override;
  void Build() const override;

 private:
  // OE1 stage 1 (P2): worker actions of a list-mode run.
  void BuildListMode() const;

  const DetectorConstruction* fDet;
  GeometryConfig fCfg;
  int fViewIndex;
  uint64_t fSeed;
  std::string fOutDir;
  int fNThreads;
  const FluxTables* fTables;
  double fExposureDays;
  const oe1::ListModeConfig* fLm = nullptr;  // null: legacy run
};

#endif
