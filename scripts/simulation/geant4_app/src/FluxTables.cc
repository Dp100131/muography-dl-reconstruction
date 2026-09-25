#include "FluxTables.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

FluxTables::FluxTables(const std::string& dir)
{
  // --- row weights
  {
    std::ifstream in(dir + "/row_weights_30d.txt");
    if (!in) throw std::runtime_error("missing row_weights_30d.txt in " + dir);
    std::string line;
    std::vector<double> weights;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double elev = 0, w = 0;
      ss >> elev >> w;
      fRowElevDeg.push_back(elev);
      weights.push_back(w);
    }
    if (fRowElevDeg.size() < 2)
      throw std::runtime_error("row_weights_30d.txt: too few rows");
    fRowHalfStepDeg = (fRowElevDeg[1] - fRowElevDeg[0]) / 2.0;

    double total = 0;
    for (double w : weights) total += w;
    fTotal30d = total;
    fRowCumulative.resize(weights.size());
    double acc = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
      acc += weights[i] / total;
      fRowCumulative[i] = acc;
    }
    fRowCumulative.back() = 1.0;
  }

  // --- momentum CDFs
  {
    std::ifstream in(dir + "/p_cdf.txt");
    if (!in) throw std::runtime_error("missing p_cdf.txt in " + dir);
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("# band_edges_deg:", 0) == 0) {
        std::istringstream ss(line.substr(17));
        double e;
        while (ss >> e) fBandEdgesDeg.push_back(e);
        continue;
      }
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      std::vector<double> cdf;
      double v;
      while (ss >> v) cdf.push_back(v);
      if (cdf.size() > 8) fBandCdf.push_back(std::move(cdf));
    }
    if (fBandEdgesDeg.size() != fBandCdf.size() + 1)
      throw std::runtime_error("p_cdf.txt: bands/edges mismatch");
  }

  // --- meta (charge ratio)
  {
    std::ifstream in(dir + "/meta.json");
    if (in) {
      nlohmann::json j = nlohmann::json::parse(in);
      const double ratio = j.value("charge_ratio_mu_plus_minus", 1.28);
      fProbMuPlus = ratio / (1.0 + ratio);
    }
  }
}

int FluxTables::SampleRow(double u) const
{
  auto it = std::lower_bound(fRowCumulative.begin(), fRowCumulative.end(), u);
  return static_cast<int>(std::distance(fRowCumulative.begin(), it));
}

double FluxTables::SampleMomentum(double elevDeg, double u) const
{
  int band = static_cast<int>(
      std::upper_bound(fBandEdgesDeg.begin(), fBandEdgesDeg.end(), elevDeg) -
      fBandEdgesDeg.begin()) - 1;
  band = std::max(0, std::min<int>(band, fBandCdf.size() - 1));

  const auto& cdf = fBandCdf[band];
  const double x = u * (cdf.size() - 1);
  const size_t i = std::min(static_cast<size_t>(x), cdf.size() - 2);
  const double frac = x - static_cast<double>(i);
  return cdf[i] + frac * (cdf[i + 1] - cdf[i]);
}
