#include "GeometryConfig.hh"

#include <fstream>
#include <stdexcept>

#include "json.hpp"

using nlohmann::json;

GeometryConfig GeometryConfig::FromJsonFile(const std::string& path)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open geometry file: " + path);
  json j = json::parse(in);

  GeometryConfig cfg;
  cfg.geometry_id = j.at("geometry_id").get<std::string>();
  const auto& ed = j.at("edifice");
  if (ed.at("shape").get<std::string>() != "cone")
    throw std::runtime_error("unsupported edifice shape (only 'cone')");
  cfg.base_radius_m = ed.at("base_radius_m").get<double>();
  cfg.height_m = ed.at("height_m").get<double>();
  cfg.rock_density_g_cm3 = ed.at("rock_density_g_cm3").get<double>();

  for (const auto& a : j.at("anomalies")) {
    AnomalyConfig ac;
    ac.kind = a.at("kind").get<std::string>();
    ac.density_g_cm3 = a.at("density_g_cm3").get<double>();
    const auto& p = a.at("params");
    ac.cx = p.at("cx_m").get<double>();
    ac.cy = p.at("cy_m").get<double>();
    if (ac.kind == "conduit") {
      ac.radius = p.at("radius_m").get<double>();
      ac.z0 = p.at("z0_m").get<double>();
      ac.z1 = p.at("z1_m").get<double>();
    }
    else if (ac.kind == "cavity") {
      ac.cz = p.at("cz_m").get<double>();
      ac.radius = p.at("radius_m").get<double>();
    }
    else if (ac.kind == "dense_body") {
      ac.cz = p.at("cz_m").get<double>();
      ac.ax = p.at("ax_m").get<double>();
      ac.ay = p.at("ay_m").get<double>();
      ac.az = p.at("az_m").get<double>();
    }
    else {
      throw std::runtime_error("unknown anomaly kind: " + ac.kind);
    }
    cfg.anomalies.push_back(ac);
  }
  return cfg;
}
