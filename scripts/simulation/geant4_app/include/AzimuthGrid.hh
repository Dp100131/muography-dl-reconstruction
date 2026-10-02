#ifndef AZIMUTH_GRID_HH
#define AZIMUTH_GRID_HH
#include <algorithm>
#include <cmath>
#include "GeometryConfig.hh"
namespace azgrid {
constexpr const char* kVersion = "az-edges-v1";
constexpr double kStepDeg = 2.0 * viewpar::kAzHalfDeg / viewpar::kNAz;
// Cells [edge_i, edge_i+1); include the exact final endpoint in the last cell.
inline int Bin(double degrees) {
  if (!std::isfinite(degrees) || degrees < -viewpar::kAzHalfDeg ||
      degrees > viewpar::kAzHalfDeg) return -1;
  if (degrees == viewpar::kAzHalfDeg) return viewpar::kNAz - 1;
  return std::min(viewpar::kNAz - 1, static_cast<int>(
      std::floor((degrees + viewpar::kAzHalfDeg) / kStepDeg)));
}
inline double Center(int i) {
  return -viewpar::kAzHalfDeg + (i + 0.5) * kStepDeg;
}
}
#endif
