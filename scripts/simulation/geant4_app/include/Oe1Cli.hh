#ifndef OE1_CLI_HH
#define OE1_CLI_HH

#include <cstdint>
#include <ostream>
#include <string>

#include "GeometryConfig.hh"
#include "Oe1ListMode.hh"
#include "PrimaryList.hh"
#include "RoiMask.hh"

// OE1 stage 1 (P2): command line of the new modes. No Geant4 dependency.
//
// If any OE1 flag appears, the WHOLE command line is parsed strictly: an
// unknown token, a missing or malformed value, a repeated flag, a P0
// geometry-diagnostic flag or a flag not allowed in the selected mode is an
// error (main exits 2). Without OE1 flags nothing here runs and the legacy
// parser alone decides, exactly as before.
namespace oe1 {

enum class CliMode { kNone, kList, kWrite, kCheck };

struct CliOptions {
  CliMode mode = CliMode::kNone;

  // Legacy value flags, as validated by the strict parser.
  std::string geometry, tables, out;
  int view = -1;
  uint64_t seed = 0;
  long long events = -1;
  int threads = 1;

  // List mode (--primaries).
  std::string primaries, primariesSha256, roiMask, roiMaskSha256, variant;
  uint64_t transportSeed = 0;
  uint32_t caseCode = 0;
  int replica = -1;
  // --region-pair I,J (conduit, dense_body|cavity) or --region-pair none. With
  // neither, a geometry with two or more anomalies is an argument error
  // (ConfigureListMode returns 2): the pair must be chosen or waived on purpose.
  bool haveRegionPair = false;
  bool regionPairNone = false;
  int regionFirst = -1, regionSecond = -1;
  MuonLogLevel logLevel = MuonLogLevel::kFull;

  // Generation (--write-primaries) and check (--check-primaries) modes.
  std::string writeOut;
  std::string checkFile;
};

// True if any OE1 stage-1 flag appears anywhere on the command line.
bool AnyStage1Flag(int argc, char** argv);

// Strict parse; false with a message on any violation.
bool ParseStage1(int argc, char** argv, CliOptions& opt, std::string& err);

void PrintStage1Usage(std::ostream& os, const char* prog);

// List mode, once the geometry JSON is loaded: checks the list sidecar
// against the geometry and the command line, builds the region Z and fills
// `out` (runtime provenance is added later by main). Returns 0 on success,
// 1 on an input mismatch, 2 on an argument that is invalid for this geometry.
// Returns 2 also when the geometry has two or more anomalies and neither
// --region-pair I,J nor --region-pair none was given (plan W2): a forgotten
// pair would silently turn every crossing field into "not applicable".
int ConfigureListMode(const CliOptions& opt, const GeometryConfig& cfg,
                      const PrimaryList& list, const RoiMask& mask,
                      ListModeConfig& out, std::string& err);

// List mode: records in `out` the SHA-256 of this executable (/proc/self/exe,
// as --write-primaries does for its sidecar), of the geometry JSON (opt.geometry)
// and of the three generator table files (row_weights_30d.txt, p_cdf.txt and
// meta.json of opt.tables). False with a message when a file cannot be read.
bool RecordProvenance(const CliOptions& opt, ListModeConfig& out, std::string& err);

// List mode, before the run manager exists: --out must be an existing,
// writable directory and none of the outputs of this run (<stem>_hist.csv,
// <stem>.meta.json, <stem>_muones.csv, <stem>_muones.csv.part) may exist, so
// a transport is never started that would overwrite a previous result or
// fail to write its own. `stem` is <geometry_id>_view<k>. Returns false with
// a message otherwise.
bool PreflightOutputs(const std::string& outDir, const std::string& stem, std::string& err);

}  // namespace oe1

#endif
