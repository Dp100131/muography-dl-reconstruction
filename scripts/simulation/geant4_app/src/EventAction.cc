#include "EventAction.hh"

#include <algorithm>
#include <string>
#include <utility>

#include "G4Event.hh"
#include "G4Material.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

#include "Oe1Hash.hh"
#include "RunData.hh"

namespace {

// prim_fnv: FNV-1a-64 of the primary as Geant4 sees it at BeginOfEventAction
// (ok = exactly one vertex with one muon). The bytes hashed, in this order,
// all little-endian:
//   x, y, z          float64, vertex position in mm  (= list x_m, y_m, z_m * 1000)
//   dx, dy, dz       float64, momentum direction     (= list d * (1 / sqrt(dx^2 + dy^2 + dz^2)):
//                    G4ParticleGun applies CLHEP unit(), which multiplies each
//                    component by the reciprocal of the norm; dividing by the
//                    norm gives other last bits)
//   ekin             float64, kinetic energy in MeV  (= list ekin_gev * 1000)
//   pdg              int32, PDG code (13 or -13)
// It can be recomputed offline from a list row, e.g. in Python:
//   inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)
//   payload = struct.pack('<7d', x*1000.0, y*1000.0, z*1000.0, dx*inv, dy*inv, dz*inv,
//                         ekin*1000.0) + struct.pack('<i', pdg)
//   h = 0xcbf29ce484222325; for b in payload: h = ((h ^ b) * 0x100000001b3) & (2**64 - 1)
// (see scripts/cluster/oe1_stage1/README.md). The multiplications and the norm
// are evaluated exactly as written: x*1000.0 is the double product, and the norm
// is the left-to-right sum dx*dx + dy*dy + dz*dz.
uint64_t PrimaryFnv(const G4Event* event, bool& ok)
{
  oe1::Fnv1a64 h;
  ok = false;
  if (event->GetNumberOfPrimaryVertex() != 1) return h.Value();
  const G4PrimaryVertex* v = event->GetPrimaryVertex(0);
  if (v == nullptr || v->GetNumberOfParticle() != 1) return h.Value();
  const G4PrimaryParticle* p = v->GetPrimary(0);
  if (p == nullptr) return h.Value();
  h.F64(v->GetX0());
  h.F64(v->GetY0());
  h.F64(v->GetZ0());
  const G4ThreeVector& dir = p->GetMomentumDirection();
  h.F64(dir.x());
  h.F64(dir.y());
  h.F64(dir.z());
  h.F64(p->GetKineticEnergy());
  h.I32(p->GetPDGcode());
  ok = (p->GetPDGcode() == 13 || p->GetPDGcode() == -13);
  return h.Value();
}

}  // namespace

EventAction::EventAction(const oe1::ListModeConfig& cfg) : fCfg(cfg) {}

void EventAction::BeginOfEventAction(const G4Event* event)
{
  // Runs after Geant4's own per-event reseed and GeneratePrimaries, before
  // any track exists: from here on the stream depends only on the key.
  const G4int id = event->GetEventID();
  fSeedKey[0] = static_cast<long>(id);
  fSeedKey[1] = static_cast<long>(fCfg.transportSeed);
  fSeedKey[2] = static_cast<long>(fCfg.caseCode);
  fSeedKey[3] = static_cast<long>(oe1::kTagTransport);
  G4Random::setTheSeeds(fSeedKey, 4);
  CLHEP::HepRandomEngine* engine = G4Random::getTheEngine();

  fRec = oe1::MuonRecord{};
  fRec.id = id;
  fRec.rng0 = oe1::HashEngineWords(engine->put());

  if (!fWorkerNoted) {
    fWorkerNoted = true;
    RunData::Instance().NoteWorker(G4Threading::G4GetThreadId(), engine->name());
  }
  bool primaryOk = false;
  fRec.primFnv = PrimaryFnv(event, primaryOk);
  if (!primaryOk) RunData::Instance().SetError("primary_not_single_muon");

  fLastProcess = nullptr;
  fLastStepStatus = fUndefined;
  fLastPostEkin = 0.0;
  fLastPostVolume = nullptr;
  fLastPostPos = G4ThreeVector();
  fEntries.Reset();
}

oe1::MatCode EventAction::Classify(const G4Material* mat) const
{
  if (mat == nullptr) return oe1::MatCode::kOther;
  const G4String& name = mat->GetName();
  if (name == fCfg.rockMaterial) return oe1::MatCode::kRock;
  if (name == fCfg.firstMaterial) return oe1::MatCode::kA0;
  if (name == fCfg.secondMaterial) return oe1::MatCode::kA1;
  if (name == "G4_AIR") return oe1::MatCode::kAir;
  return oe1::MatCode::kOther;
}

void EventAction::OnPrimaryStep(const G4Step* step)
{
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  ++fRec.nSteps;
  fLastProcess = post->GetProcessDefinedStep();
  fLastStepStatus = post->GetStepStatus();
  fLastPostEkin = post->GetKineticEnergy();
  fLastPostVolume = post->GetPhysicalVolume();
  if (fCfg.logLevel != oe1::MuonLogLevel::kFull) return;

  const G4ThreeVector& p0 = pre->GetPosition();
  const G4ThreeVector& p1 = post->GetPosition();
  fLastPostPos = p1;
  // Lateral offset (mm) of the post-step point from the line pre + s * dir_pre.
  // It is NOT zero in practice: WentzelVI displaces the post-step point
  // sideways (by up to 0.99 x safety) also with MuHadLateralDisplacement =
  // false, so the pre->post chord is not the path that was followed. It is
  // recorded for every step (max_lat_mm, a diagnostic) and, below, accumulated
  // over the steps near Z (max_lat_z_mm / sum_lat_z_mm). sum_lat_z_mm is the
  // accumulated transverse displacement of the steps near Z: a scale of the
  // uncertainty of the chord-based lengths in Z, NOT a bound. At an incidence
  // angle theta the length error is ~ lat / sin(theta); for a chord skimming a
  // surface it grows like sqrt(2 R lat) (R <= 86.3 m here), so report it as
  // <~ 2 sqrt(2 R_max lat). A chord that penetrates Z by less than
  // max_lat_z_mm may be classified differently from the real path.
  const double lat = (p1 - p0).cross(pre->GetMomentumDirection()).mag();
  if (lat > fRec.maxLatMm) fRec.maxLatMm = lat;
  if (!fCfg.regionApplicable) return;

  const oe1::Vec3 a{p0.x() / m, p0.y() / m, p0.z() / m};
  const oe1::Vec3 b{p1.x() / m, p1.y() / m, p1.z() / m};
  const oe1::CrossingResult c = fCfg.region.Intersect(a, b);
  if (c.nearZ) {
    ++fRec.nStepsNearZ;
    fRec.sumLatZMm += lat;
    if (lat > fRec.maxLatZMm) fRec.maxLatZMm = lat;
  }
  fRec.lc += c.lenCyl;
  fRec.le += c.lenEll;
  if (c.graze) ++fRec.nGrazeZ;
  const double chord = (p1 - p0).mag() / m;
  // Slivers: Geant4 leaves the endpoint of a step on a navigation boundary to
  // ~1e-12 m, and where that boundary is also a surface of Z (in B the
  // anomaly_0 / anomaly_1 interface inside the conduit) the analytic interval
  // of the step is a sliver of ~1e-12 m on either side of it. A sliver carries
  // no information but would set mat_in / mat_out with the material of the
  // wrong side and count a step. The chord-length sums below keep every sliver
  // (they add up every piece of every chord); the categorical fields (crossed, steps, entries,
  // mat_in / mat_out, e_in / e_out) ignore a piece of length <= the tangency
  // band 2 eps. This includes `crossed`: a step that is deep but not longer
  // than 2 eps (for example a 1 um step inside Z) sets nothing, so crossed = 1
  // always implies n_steps_z >= 1 and a non-empty mat_in.
  const double sliverM = 2.0 * fCfg.region.Eps();
  if (fEntries.Step(c, chord, sliverM)) ++fRec.nEntriesZ;
  if (!(c.len > 0.0)) return;

  // Material the navigator assigned to the whole step: the pre-step volume's.
  const G4Material* mat = pre->GetMaterial();
  const oe1::MatCode code = Classify(mat);
  fRec.lz += c.len;
  fRec.lzTrue += c.len * (step->GetStepLength() / m) / chord;
  switch (code) {
    case oe1::MatCode::kRock: fRec.lzRock += c.len; break;
    case oe1::MatCode::kA0: fRec.lzA0 += c.len; break;
    case oe1::MatCode::kA1: fRec.lzA1 += c.len; break;
    default: fRec.lzOther += c.len; break;
  }
  if (mat != nullptr) fRec.rhoLzGcm2 += (mat->GetDensity() / (g / cm3)) * c.len * 100.0;

  // Categorical fields: a sliver is not a Z step (see above).
  if (!(c.len > sliverM)) return;
  if (c.deep) fRec.deep = true;
  ++fRec.nStepsZ;
  if (c.bothOut) ++fRec.nStepsZBothOut;

  // Energy is only defined at step points: interpolate along the chord and
  // keep the step's [min, max] of (E_post, E_pre) as the bracket.
  const double ePre = pre->GetKineticEnergy() / GeV;
  const double ePost = post->GetKineticEnergy() / GeV;
  const double eLo = std::min(ePre, ePost);
  const double eHi = std::max(ePre, ePost);
  if (!fRec.haveEIn) {
    fRec.haveEIn = true;
    fRec.matIn = code;
    fRec.eInGeV = ePre + (ePost - ePre) * c.lo;
    fRec.eInLoGeV = eLo;
    fRec.eInHiGeV = eHi;
  }
  fRec.haveEOut = true;
  fRec.matOut = code;
  fRec.eOutGeV = ePre + (ePost - ePre) * c.hi;
  fRec.eOutLoGeV = eLo;
  fRec.eOutHiGeV = eHi;
}

void EventAction::OnPlane(int iAz, int iEl, double ekinGeV, double elDeg, double relAzDeg)
{
  fRec.planeReached = true;
  fRec.ekinPlaneGeV = ekinGeV;
  fRec.elPlaneDeg = elDeg;
  fRec.relAzPlaneDeg = relAzDeg;
  if (iAz >= 0 && iEl >= 0) {
    fRec.detected = 1;
    fRec.iel = iEl;
    fRec.iaz = iAz;
    fRec.inRint = fCfg.mask->At(iEl, iAz) ? 1 : 0;
    fRec.status = oe1::MuonStatus::kDet;
  }
  else {
    fRec.status = oe1::MuonStatus::kPlaneOut;
  }
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  fRec.rng1 = oe1::HashEngineWords(G4Random::getTheEngine()->put());
  fRec.endProcess = (fLastProcess != nullptr) ? std::string(fLastProcess->GetProcessName())
                                              : std::string("none");
  if (event->IsAborted()) {
    fRec.status = oe1::MuonStatus::kAborted;
  }
  else if (!fRec.planeReached) {
    if (fRec.nSteps == 0) fRec.status = oe1::MuonStatus::kOther;
    else if (fLastStepStatus == fWorldBoundary || fLastPostVolume == nullptr)
      fRec.status = oe1::MuonStatus::kWorldExit;
    else if (fRec.endProcess.find("Decay") != std::string::npos)
      fRec.status = oe1::MuonStatus::kDecay;
    else if (fRec.endProcess.find("Capture") != std::string::npos)
      fRec.status = oe1::MuonStatus::kMuCapture;
    else if (fLastPostEkin <= 0.0)
      fRec.status = oe1::MuonStatus::kStopped;
    else
      fRec.status = oe1::MuonStatus::kOther;
  }
  if (fCfg.logLevel == oe1::MuonLogLevel::kFull && fRec.nSteps > 0) {
    fRec.haveEnd = true;
    fRec.endXm = fLastPostPos.x() / m;
    fRec.endYm = fLastPostPos.y() / m;
    fRec.endZm = fLastPostPos.z() / m;
    fRec.endVolume = (fLastPostVolume != nullptr) ? std::string(fLastPostVolume->GetName())
                                                  : std::string("OutOfWorld");
  }
  RunData::Instance().AddMuon(std::move(fRec));
  fRec = oe1::MuonRecord{};
}
