#ifndef OE1_TESTS_CHECK_HH
#define OE1_TESTS_CHECK_HH

// Minimal check harness shared by the standalone tests (no framework, no
// assert(): the tests must also run under -DNDEBUG).
#include <cmath>
#include <cstdio>
#include <string>

namespace oe1test {

inline int g_checks = 0;
inline int g_fail = 0;

inline void Fail(const char* file, int line, const std::string& msg)
{
  ++g_fail;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, msg.c_str());
}

}  // namespace oe1test

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++oe1test::g_checks;                                                     \
    if (!(cond)) oe1test::Fail(__FILE__, __LINE__, #cond);                   \
  } while (0)

#define CHECK_NEAR(actual, expected, tol)                                            \
  do {                                                                               \
    ++oe1test::g_checks;                                                             \
    const double a__ = static_cast<double>(actual);                                  \
    const double e__ = static_cast<double>(expected);                                \
    if (!(std::fabs(a__ - e__) <= static_cast<double>(tol))) {                       \
      char b__[256];                                                                 \
      std::snprintf(b__, sizeof b__, "%s = %.17g, expected %.17g (tol %.3g)",        \
                    #actual, a__, e__, static_cast<double>(tol));                    \
      oe1test::Fail(__FILE__, __LINE__, b__);                                        \
    }                                                                                \
  } while (0)

#define CHECK_EQ_STR(actual, expected)                                               \
  do {                                                                               \
    ++oe1test::g_checks;                                                             \
    const std::string a__ = (actual);                                                \
    const std::string e__ = (expected);                                              \
    if (a__ != e__) {                                                                \
      oe1test::Fail(__FILE__, __LINE__,                                              \
                    std::string(#actual) + " = '" + a__ + "', expected '" + e__ + "'"); \
    }                                                                                \
  } while (0)

#endif
