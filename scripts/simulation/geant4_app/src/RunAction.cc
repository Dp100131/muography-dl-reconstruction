#include "RunAction.hh"

#include <chrono>
#include <fstream>

#include "G4Run.hh"
#include "G4Threading.hh"
#include "G4Version.hh"

#include "RunData.hh"
#include "json.hpp"

namespace {
double nowSeconds()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
}  // namespace

RunAction::RunAction(const GeometryConfig& cfg, int viewIndex, uint64_t seed,
                     std::string outDir, int nThreads, double exposureDays)
    : fCfg(cfg), fViewIndex(viewIndex), fSeed(seed),
      fOutDir(std::move(outDir)), fNThreads(nThreads),
      fExposureDays(exposureDays)
{}

void RunAction::BeginOfRunAction(const G4Run*)
{
  if (IsMaster()) fWallStart = nowSeconds();
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  if (!IsMaster()) return;

  const double wall = nowSeconds() - fWallStart;
  const auto& data = RunData::Instance();
  const std::string stem =
      fOutDir + "/" + fCfg.geometry_id + "_view" + std::to_string(fViewIndex);

  // Histogram CSV: one row per elevation bin (matches surrogate row order).
  {
    std::ofstream out(stem + "_hist.csv");
    const auto& h = data.Histogram();
    for (int iEl = 0; iEl < viewpar::kNEl; ++iEl) {
      for (int iAz = 0; iAz < viewpar::kNAz; ++iAz) {
        out << h[static_cast<size_t>(iEl) * viewpar::kNAz + iAz];
        out << (iAz + 1 < viewpar::kNAz ? ',' : '\n');
      }
    }
  }

  // Metadata sidecar (schema aligned with the surrogate .meta.json).
  {
    nlohmann::json meta;
    meta["engine"] = std::string("geant4-") + G4Version;
    meta["geometry_id"] = fCfg.geometry_id;
    meta["view_index"] = fViewIndex;
    meta["seed"] = fSeed;
    meta["threads"] = fNThreads;
    meta["events_transported"] = run->GetNumberOfEvent();
    meta["ecomug_draws_total"] = data.Generated();
    meta["detected_in_bins"] = data.Detected();
    meta["mean_ekin_detected_gev"] = data.MeanEkinGeV();
    meta["wall_seconds"] = wall;
    meta["seconds_per_event"] =
        run->GetNumberOfEvent() > 0 ? wall / run->GetNumberOfEvent() : 0.0;
    meta["grid"] = {{"n_az", viewpar::kNAz},
                    {"n_el", viewpar::kNEl},
                    {"az_half_deg", viewpar::kAzHalfDeg},
                    {"el_min_deg", viewpar::kElMinDeg},
                    {"el_max_deg", viewpar::kElMaxDeg}};
    meta["detector"] = {{"side_m", viewpar::kDetSideM},
                        {"distance_m", viewpar::kDetDistanceM},
                        {"height_m", viewpar::kDetHeightM}};
    meta["exposure_days"] = fExposureDays;
    meta["sampling"] = "table-driven: protocol lambda_open row weights "
                       "(Guan-normalized, surrogate convention) + real-EcoMug "
                       "momentum spectra; backtracked line of sight, uniform "
                       "panel impact";
    meta["physics_note"] = "secondaries killed at creation; primary-only "
                           "transport for transmission counting";

    std::ofstream out(stem + ".meta.json");
    out << meta.dump(1) << "\n";
  }

  G4cout << "RUN DONE " << fCfg.geometry_id << " view " << fViewIndex
         << ": events=" << run->GetNumberOfEvent()
         << " detected=" << data.Detected()
         << " wall_s=" << wall << G4endl;
}
