#ifndef RUN_DATA_HH
#define RUN_DATA_HH

#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "GeometryConfig.hh"
#include "MuonRecord.hh"

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

  // --- OE1 stage 1 (P2), list mode only; unused (empty) in legacy runs.
  // Writers take the same mutex as RecordHit; readers run on the master
  // after the event loop has finished.
  double EkinSumGeV() const { return fEkinSumGeV; }
  void ReserveMuons(std::size_t n)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    fMuons.reserve(n);
  }
  void AddMuon(oe1::MuonRecord&& rec)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    fMuons.push_back(std::move(rec));
  }
  std::vector<oe1::MuonRecord>& Muons() { return fMuons; }
  void NoteWorker(int threadId, const std::string& engineName)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    fWorkerIds.insert(threadId);
    fEngineNames.insert(engineName);
  }
  const std::set<int>& WorkerIds() const { return fWorkerIds; }
  const std::set<std::string>& EngineNames() const { return fEngineNames; }
  void SetError(const std::string& code)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    fErrors.insert(code);
  }
  const std::set<std::string>& Errors() const { return fErrors; }
  bool HasErrors() const { return !fErrors.empty(); }
  // Set by the master RunAction when its list-mode finalisation completed; main()
  // returns 4 when a list-mode run ends without it (e.g. BeamOn refused).
  void MarkFinalized() { fFinalized.store(true); }
  bool Finalized() const { return fFinalized.load(); }

 private:
  RunData() : fHist(static_cast<size_t>(viewpar::kNAz) * viewpar::kNEl, 0) {}

  std::vector<uint64_t> fHist;
  std::atomic<uint64_t> fGenerated{0};
  uint64_t fDetected = 0;
  double fEkinSumGeV = 0.0;
  std::mutex fMutex;
  std::vector<oe1::MuonRecord> fMuons;
  std::set<int> fWorkerIds;
  std::set<std::string> fEngineNames;
  std::set<std::string> fErrors;
  std::atomic<bool> fFinalized{false};
};

#endif
