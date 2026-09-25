#ifndef DETECTOR_CONSTRUCTION_HH
#define DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"

#include "GeometryConfig.hh"

class G4LogicalVolume;
class G4Material;

// Builds the volcano (cone + anomalies) from the shared JSON catalog plus a
// thin detector panel at the requested view position. Coordinate frame is the
// SAME as the surrogate engine: cone axis at origin, ground at z=0, detector
// at distance kDetDistanceM and azimuth view_index*60 deg looking at the axis.
class DetectorConstruction : public G4VUserDetectorConstruction
{
 public:
  DetectorConstruction(const GeometryConfig& cfg, int viewIndex);

  G4VPhysicalVolume* Construct() override;

  const G4LogicalVolume* DetectorPanel() const { return fDetectorPanel; }

 private:
  G4Material* RockMaterial(double density_g_cm3, const G4String& name) const;

  GeometryConfig fCfg;
  int fViewIndex;
  G4LogicalVolume* fDetectorPanel = nullptr;
};

#endif
