#ifndef RUN_DATA_HH
#define RUN_DATA_HH

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "GeometryConfig.hh"

// Process-wide accumulator shared by all worker threads. The histogram is
// merged under a mutex only at detector hits (rare relative to steps), and
// the counters are atomics — safe and cheap for this workload.
class RunData
{
 public:
  static RunData& Instance()
  {
    static RunData instance;
    return instance;
  }

  void RecordGenerated() { fGenerated.fetch_add(1, std::memory_order_relaxed); }

  void RecordHit(int azBin, int elBin, double ekinGeV)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    fHist[static_cast<size_t>(elBin) * viewpar::kNAz + azBin] += 1;
    fDetected += 1;
    fEkinSumGeV += ekinGeV;
  }

  const std::vector<uint64_t>& Histogram() const { return fHist; }
  uint64_t Generated() const { return fGenerated.load(); }
  uint64_t Detected() const { return fDetected; }
  double MeanEkinGeV() const
  {
    return fDetected > 0 ? fEkinSumGeV / static_cast<double>(fDetected) : 0.0;
  }

 private:
  RunData() : fHist(static_cast<size_t>(viewpar::kNAz) * viewpar::kNEl, 0) {}

  std::vector<uint64_t> fHist;
  std::atomic<uint64_t> fGenerated{0};
  uint64_t fDetected = 0;
  double fEkinSumGeV = 0.0;
  std::mutex fMutex;
};

#endif
