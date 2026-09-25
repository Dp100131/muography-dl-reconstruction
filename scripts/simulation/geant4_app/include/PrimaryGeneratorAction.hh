#ifndef PRIMARY_GENERATOR_ACTION_HH
#define PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"

#include "FluxTables.hh"
#include "GeometryConfig.hh"

class G4ParticleGun;
class G4Event;

// Table-driven primary generator.
//
// Direction: elevation row sampled by the protocol open-sky weights
// (lambda_open = Phi_int*A*d_omega, Guan-normalized — identical acceptance
// convention to the surrogate, so E9 cross-comparisons isolate transport
// physics), uniform azimuth inside the +/-30 deg window, uniform jitter
// within the row band. Momentum: real-EcoMug inverse CDF conditional on
// elevation. Injection: backtracked line of sight through a uniform impact
// point on the physical panel, starting outside the edifice.
//
// A naive sky-plane generation would waste ~1e9 muons per detected event;
// this directed scheme plus the protocol normalization (36.2M incidents per
// view per 30 days, see tables/meta.json) makes the campaign tractable.
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
 public:
  PrimaryGeneratorAction(int viewIndex, const GeometryConfig& cfg,
                         const FluxTables* tables);
  ~PrimaryGeneratorAction() override;

  void GeneratePrimaries(G4Event* event) override;

 private:
  G4ParticleGun* fGun = nullptr;
  const FluxTables* fTables;
  double fLookAzRad;     // world azimuth the detector looks toward
  double fStartRadiusM;  // horizontal backtrack distance from the detector
};

#endif
