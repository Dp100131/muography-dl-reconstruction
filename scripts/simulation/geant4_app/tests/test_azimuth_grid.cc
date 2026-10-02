// Standalone test of the SAME header used by SteppingAction; no Geant4 needed.
// c++ -std=c++17 -I scripts/simulation/geant4_app/include \
//   scripts/simulation/geant4_app/tests/test_azimuth_grid.cc -o test_azimuth_grid
#include "AzimuthGrid.hh"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
void check(bool good) { if (!good) throw std::runtime_error("azimuth grid test failed"); }
int main() {
  check(azgrid::Bin(-30.0)==0); check(azgrid::Bin(30.0)==255);
  check(azgrid::Bin(std::nextafter(-30.0, -31.0))==-1);
  check(azgrid::Bin(std::nextafter(30.0, 31.0))==-1);
  check(azgrid::Bin(std::numeric_limits<double>::quiet_NaN())==-1);
  check(azgrid::Bin(std::numeric_limits<double>::infinity())==-1);
  for (int i=0;i<256;++i) check(azgrid::Bin(azgrid::Center(i))==i);
  for (int i=1;i<256;++i) check(azgrid::Bin(-30+i*azgrid::kStepDeg)==i);
  std::array<int,256> histogram{};
  for (int i=0;i<1048576;++i) {
    int bin=azgrid::Bin(-30+60*(i+0.5)/1048576);
    check(bin>=0 && bin<256); ++histogram[bin];
  }
  for (int n:histogram) check(n==4096);
  std::cout << "PASS: endpoints, outside, nonfinite, centers, edges, 1048576 directions\n";
}
