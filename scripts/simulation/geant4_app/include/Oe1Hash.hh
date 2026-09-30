#ifndef OE1_HASH_HH
#define OE1_HASH_HH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// OE1 stage 1 (P2): FNV-1a-64 helpers. No Geant4 dependency.
//
// Multi-byte values are always fed in little-endian byte order, independent
// of the host, so every hash below can be reproduced offline, e.g. in Python
// with struct.pack('<q', ...), struct.pack('<d', ...), struct.pack('<i', ...).
namespace oe1 {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;  // 0xcbf29ce484222325
constexpr uint64_t kFnvPrime = 1099511628211ULL;               // 0x00000100000001b3

class Fnv1a64
{
 public:
  void Bytes(const void* data, std::size_t n)
  {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      fHash ^= static_cast<uint64_t>(p[i]);
      fHash *= kFnvPrime;
    }
  }

  void U64(uint64_t v)
  {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
    Bytes(b, 8);
  }

  void I64(int64_t v) { U64(static_cast<uint64_t>(v)); }

  void I32(int32_t v)
  {
    const auto u = static_cast<uint32_t>(v);
    unsigned char b[4];
    for (int i = 0; i < 4; ++i) b[i] = static_cast<unsigned char>(u >> (8 * i));
    Bytes(b, 4);
  }

  // IEEE-754 binary64 bit pattern, little-endian (signed zeros and NaN
  // payloads are hashed as stored).
  void F64(double v)
  {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    U64(bits);
  }

  uint64_t Value() const { return fHash; }

 private:
  uint64_t fHash = kFnvOffsetBasis;
};

inline uint64_t Fnv1a64Of(const std::string& bytes)
{
  Fnv1a64 h;
  h.Bytes(bytes.data(), bytes.size());
  return h.Value();
}

// Hash of a CLHEP engine state as returned by HepRandomEngine::put() (each
// word as a little-endian uint64). Reading put() consumes no random numbers.
inline uint64_t HashEngineWords(const std::vector<unsigned long>& words)
{
  Fnv1a64 h;
  for (const unsigned long w : words) h.U64(static_cast<uint64_t>(w));
  return h.Value();
}

// 16 lowercase hex digits, zero padded.
inline std::string Hex16(uint64_t v)
{
  static const char* digits = "0123456789abcdef";
  std::string s(16, '0');
  for (int i = 15; i >= 0; --i) {
    s[static_cast<std::size_t>(i)] = digits[v & 0xFU];
    v >>= 4;
  }
  return s;
}

// Parses exactly 16 hex digits (either case). Returns false otherwise.
inline bool ParseHex16(const std::string& s, uint64_t& out)
{
  if (s.size() != 16) return false;
  uint64_t v = 0;
  for (const char c : s) {
    int d = -1;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    if (d < 0) return false;
    v = (v << 4) | static_cast<uint64_t>(d);
  }
  out = v;
  return true;
}

}  // namespace oe1

#endif
