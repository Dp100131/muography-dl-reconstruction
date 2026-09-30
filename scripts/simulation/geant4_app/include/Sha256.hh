#ifndef OE1_SHA256_HH
#define OE1_SHA256_HH

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

// OE1 stage 1 (P2): minimal SHA-256 (FIPS 180-4), no Geant4 dependency.
// Used to verify the --primaries-sha256 / --roi-mask-sha256 values given on
// the command line against the files actually read, and to record the
// generator binary's own SHA-256 in a primary-list sidecar. Checked against
// Python's hashlib in tests/test_primary_list.cc.
namespace oe1 {

class Sha256
{
 public:
  Sha256() { Reset(); }

  void Reset()
  {
    static const uint32_t kInit[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                      0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                      0x1f83d9abU, 0x5be0cd19U};
    for (int i = 0; i < 8; ++i) fH[i] = kInit[i];
    fBufLen = 0;
    fTotalBytes = 0;
  }

  void Update(const void* data, std::size_t n)
  {
    const auto* p = static_cast<const unsigned char*>(data);
    fTotalBytes += n;
    while (n > 0) {
      const std::size_t take = (64 - fBufLen < n) ? 64 - fBufLen : n;
      for (std::size_t i = 0; i < take; ++i) fBuf[fBufLen + i] = p[i];
      fBufLen += take;
      p += take;
      n -= take;
      if (fBufLen == 64) {
        Block(fBuf);
        fBufLen = 0;
      }
    }
  }

  // Finishes the digest; the object must be Reset() before reuse.
  std::array<unsigned char, 32> Final()
  {
    const uint64_t bitLen = fTotalBytes * 8U;
    const unsigned char one = 0x80;
    Update(&one, 1);
    const unsigned char zero = 0x00;
    while (fBufLen != 56) Update(&zero, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i)
      len[i] = static_cast<unsigned char>(bitLen >> (8 * (7 - i)));
    Update(len, 8);
    std::array<unsigned char, 32> out{};
    for (int i = 0; i < 8; ++i) {
      for (int k = 0; k < 4; ++k) {
        out[static_cast<std::size_t>(4 * i + k)] =
            static_cast<unsigned char>(fH[i] >> (8 * (3 - k)));
      }
    }
    return out;
  }

  static std::string Hex(const std::array<unsigned char, 32>& digest)
  {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (const unsigned char b : digest) {
      s.push_back(digits[b >> 4]);
      s.push_back(digits[b & 0x0F]);
    }
    return s;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void Block(const unsigned char* p)
  {
    static const uint32_t kK[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
        0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
        0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
        0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
        0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(p[4 * i]) << 24) |
             (static_cast<uint32_t>(p[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(p[4 * i + 2]) << 8) |
             static_cast<uint32_t>(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = fH[0], b = fH[1], c = fH[2], d = fH[3];
    uint32_t e = fH[4], f = fH[5], g = fH[6], h = fH[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + S1 + ch + kK[i] + w[i];
      const uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    fH[0] += a;
    fH[1] += b;
    fH[2] += c;
    fH[3] += d;
    fH[4] += e;
    fH[5] += f;
    fH[6] += g;
    fH[7] += h;
  }

  uint32_t fH[8];
  unsigned char fBuf[64];
  std::size_t fBufLen = 0;
  uint64_t fTotalBytes = 0;
};

inline std::string Sha256Hex(const std::string& bytes)
{
  Sha256 s;
  s.Update(bytes.data(), bytes.size());
  return Sha256::Hex(s.Final());
}

// SHA-256 of a file's bytes. Returns false if the file cannot be read.
inline bool Sha256File(const std::string& path, std::string& hexOut)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  Sha256 s;
  char buf[1 << 16];
  while (in) {
    in.read(buf, sizeof buf);
    const std::streamsize got = in.gcount();
    if (got > 0) s.Update(buf, static_cast<std::size_t>(got));
  }
  if (in.bad()) return false;
  hexOut = Sha256::Hex(s.Final());
  return true;
}

// True for exactly 64 lowercase hex digits (the sha256sum output format).
inline bool IsSha256Hex(const std::string& s)
{
  if (s.size() != 64) return false;
  for (const char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

}  // namespace oe1

#endif
