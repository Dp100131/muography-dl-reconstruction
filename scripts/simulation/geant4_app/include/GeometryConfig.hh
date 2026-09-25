#ifndef GEOMETRY_CONFIG_HH
#define GEOMETRY_CONFIG_HH

#include <string>
#include <vector>

// Mirrors the JSON schema of simulation/geometry_catalog/geometries/*.json —
// the SAME files consumed by the surrogate engine (single source of truth).
struct AnomalyConfig {
  std::string kind;          // "conduit" | "cavity" | "dense_body"
  double density_g_cm3 = 0.0;
  // conduit: cx, cy, radius, z0, z1 · cavity: cx, cy, cz, radius
  // dense_body: cx, cy, cz, ax, ay, az   (all meters)
  double cx = 0, cy = 0, cz = 0;
  double radius = 0;
  double z0 = 0, z1 = 0;
  double ax = 0, ay = 0, az = 0;
};

struct GeometryConfig {
  std::string geometry_id;
  double base_radius_m = 0.0;
  double height_m = 0.0;
  double rock_density_g_cm3 = 2.65;
  std::vector<AnomalyConfig> anomalies;

  static GeometryConfig FromJsonFile(const std::string& path);
};

// Detector/view contract — MUST match simulation/surrogate/imaging.py
namespace viewpar {
constexpr int    kNAz = 256;
constexpr int    kNEl = 256;
constexpr double kAzHalfDeg = 30.0;
constexpr double kElMinDeg = 2.0;
constexpr double kElMaxDeg = 47.0;
constexpr double kDetSideM = 1.2;       // 120 x 120 cm (MuTe-like)
constexpr double kDetDistanceM = 1000.0;
constexpr double kDetHeightM = 1.5;
constexpr double kViewStepDeg = 60.0;   // view_index * 60 degrees
}  // namespace viewpar

#endif
