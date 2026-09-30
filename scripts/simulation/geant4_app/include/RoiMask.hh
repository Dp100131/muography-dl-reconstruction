#ifndef OE1_ROI_MASK_HH
#define OE1_ROI_MASK_HH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "GeometryConfig.hh"
#include "Oe1Hash.hh"
#include "Sha256.hh"

// OE1 stage 1 (P2): frozen angular mask (R_int) exported as text.
//
// Format (strict): exactly kNEl lines of exactly kNAz characters '0'/'1',
// each line terminated by LF, nothing else. Line = iEl (row of hist.csv),
// character = iAz (column of hist.csv), i.e. the same [iEl, iAz] indexing as
// RunData / hist.csv and as the npz array. No Geant4 dependency.
namespace oe1 {

class RoiMask
{
 public:
  static constexpr int kRows = viewpar::kNEl;
  static constexpr int kCols = viewpar::kNAz;

  // Loads and validates the file; if expectedSha256 is non-empty the file's
  // SHA-256 must equal it. On failure returns false with a message.
  bool Load(const std::string& path, const std::string& expectedSha256, std::string& err)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      err = "cannot open ROI mask file: " + path;
      return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) {
      err = "read error on ROI mask file: " + path;
      return false;
    }
    return LoadFromBytes(ss.str(), expectedSha256, err);
  }

  bool LoadFromBytes(const std::string& bytes, const std::string& expectedSha256,
                     std::string& err)
  {
    const std::size_t lineLen = static_cast<std::size_t>(kCols) + 1;
    if (bytes.size() != lineLen * static_cast<std::size_t>(kRows)) {
      err = "ROI mask must be " + std::to_string(kRows) + " LF-terminated lines of " +
            std::to_string(kCols) + " '0'/'1' characters (size " +
            std::to_string(bytes.size()) + " bytes)";
      return false;
    }
    std::vector<unsigned char> bits(static_cast<std::size_t>(kRows) * kCols, 0);
    std::size_t ones = 0;
    int rowMin = kRows, rowMax = -1, colMin = kCols, colMax = -1;
    for (int r = 0; r < kRows; ++r) {
      const std::size_t base = static_cast<std::size_t>(r) * lineLen;
      if (bytes[base + kCols] != '\n') {
        err = "ROI mask line " + std::to_string(r) + " is not LF-terminated at column " +
              std::to_string(kCols);
        return false;
      }
      for (int c = 0; c < kCols; ++c) {
        const char ch = bytes[base + static_cast<std::size_t>(c)];
        if (ch != '0' && ch != '1') {
          err = "ROI mask line " + std::to_string(r) + " has a character other than 0/1";
          return false;
        }
        if (ch == '1') {
          bits[static_cast<std::size_t>(r) * kCols + static_cast<std::size_t>(c)] = 1;
          ++ones;
          if (r < rowMin) rowMin = r;
          if (r > rowMax) rowMax = r;
          if (c < colMin) colMin = c;
          if (c > colMax) colMax = c;
        }
      }
    }
    const std::string sha = Sha256Hex(bytes);
    if (!expectedSha256.empty() && sha != expectedSha256) {
      err = "ROI mask SHA-256 mismatch: file " + sha + " != expected " + expectedSha256;
      return false;
    }
    fBits.swap(bits);
    fOnes = ones;
    fRowMin = rowMin;
    fRowMax = rowMax;
    fColMin = colMin;
    fColMax = colMax;
    fFnv = Fnv1a64Of(bytes);
    fSha256 = sha;
    fLoaded = true;
    return true;
  }

  bool Loaded() const { return fLoaded; }
  bool At(int iEl, int iAz) const
  {
    return fBits[static_cast<std::size_t>(iEl) * kCols + static_cast<std::size_t>(iAz)] != 0;
  }
  std::size_t Ones() const { return fOnes; }
  int RowMin() const { return fRowMin; }
  int RowMax() const { return fRowMax; }
  int ColMin() const { return fColMin; }
  int ColMax() const { return fColMax; }
  uint64_t ContentFnv() const { return fFnv; }
  const std::string& FileSha256() const { return fSha256; }

 private:
  std::vector<unsigned char> fBits;
  std::size_t fOnes = 0;
  int fRowMin = -1, fRowMax = -1, fColMin = -1, fColMax = -1;
  uint64_t fFnv = 0;
  std::string fSha256;
  bool fLoaded = false;
};

}  // namespace oe1

#endif
