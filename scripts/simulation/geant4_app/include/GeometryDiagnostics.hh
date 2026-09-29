#ifndef GEOMETRY_DIAGNOSTICS_HH
#define GEOMETRY_DIAGNOSTICS_HH

#include <string>

#include "G4ThreeVector.hh"

// Post-Initialize() geometry diagnostics: overlap checking, a volume/material
// dump, point probes and 2D material slices. Read-only against the already
// constructed geometry; never touches the run loop (no BeamOn). Intended to
// run once, on the master thread, right after G4RunManager::Initialize().
namespace geomdiag {

// A generic axis-aligned scan spec for --slice-y / --slice-z:
//   coord            fixed plane coordinate (metres)
//   min1/max1, step  first free axis range (metres, inclusive)
//   min2/max2        second free axis range (metres, inclusive)
struct SliceSpec {
  double coord = 0.0;
  double min1 = 0.0, max1 = 0.0;
  double min2 = 0.0, max2 = 0.0;
  double step = 0.0;
};

// Parses "X,Y,Z" (metres). Returns false on malformed input.
bool ParseProbe(const std::string& arg, G4ThreeVector& outPointM);

// Parses "COORD:MIN1:MAX1:MIN2:MAX2:STEP" (metres). Returns false on
// malformed input (wrong token count, non-numeric, empty range or step<=0)
// or when the resulting grid would exceed the point-count cap enforced in
// GeometryDiagnostics.cc (guards the cluster's tight per-user quota against
// an oversized CSV from a single flag).
bool ParseSlice(const std::string& arg, SliceSpec& outSpec);

// --check-geometry: CheckOverlaps() on every physical volume that has a
// mother logical volume. Prints one OVERLAP-CHECK line per volume plus an
// OVERLAP-CHECK-SUMMARY line. Returns 1 if any overlap was found, 0
// otherwise.
int RunCheckGeometry(const std::string& geometryId, int view, long seed,
                     int resolution, double tolerance, int maxErr);

// --dump-geometry: one DUMP-VOLUME line per physical volume (material,
// density, cubic volume). volumeStatN is the MC point budget used for
// Boolean solids (G4BooleanSolid), whose volume has no closed form.
void RunDumpGeometry(long volumeStatN);

// --probe: material lookup at a single world-frame point (metres).
void RunProbe(const G4ThreeVector& pointM);

// --slice-y / --slice-z: material lookup on a regular grid over the plane
// y=coord (resp. z=coord); writes a CSV under outDir named
// "<geometryId>_view<view>_slice_y<coord>.csv" (resp. "..._slice_z...csv").
void RunSliceY(const std::string& geometryId, int view,
               const std::string& outDir, const SliceSpec& spec);
void RunSliceZ(const std::string& geometryId, int view,
               const std::string& outDir, const SliceSpec& spec);

}  // namespace geomdiag

#endif
