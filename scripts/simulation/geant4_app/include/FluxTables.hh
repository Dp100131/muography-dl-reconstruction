#ifndef FLUX_TABLES_HH
#define FLUX_TABLES_HH

#include <string>
#include <vector>

// Loads the table-driven generator inputs produced by
// analysis/build_generator_tables.py:
//   row_weights_30d.txt : open-sky expected counts per elevation row (30 d),
//                         protocol convention lambda_open = Phi_int*A*d_omega
//   p_cdf.txt           : momentum inverse CDF per elevation band (real
//                         EcoMug spectra, GeV/c)
//   meta.json           : charge ratio and totals
//
// Sampling contract: direction weighting reproduces the protocol acceptance
// exactly (so Geant4-vs-surrogate differences are transport physics only);
// the momentum spectrum carries the real generator physics.
class FluxTables
{
 public:
  explicit FluxTables(const std::string& dir);

  // u in [0,1): elevation row index sampled by open-sky weight.
  int SampleRow(double u) const;

  // Elevation (deg) of a row center; kRowHalfStepDeg gives the jitter range.
  double RowElevationDeg(int row) const { return fRowElevDeg[row]; }
  double RowHalfStepDeg() const { return fRowHalfStepDeg; }

  // u in [0,1): momentum (GeV/c) from the band containing elev_deg.
  double SampleMomentum(double elevDeg, double u) const;

  // u in [0,1): true -> mu+.
  bool SampleChargeIsPositive(double u) const
  {
    return u < fProbMuPlus;
  }

  double TotalCounts30d() const { return fTotal30d; }

 private:
  std::vector<double> fRowElevDeg;
  std::vector<double> fRowCumulative;  // normalized cumulative weights
  double fRowHalfStepDeg = 0.0;

  std::vector<double> fBandEdgesDeg;
  std::vector<std::vector<double>> fBandCdf;  // [band][quantile] -> p GeV/c

  double fProbMuPlus = 0.561;  // ratio 1.28 -> 1.28/2.28
  double fTotal30d = 0.0;
};

#endif
