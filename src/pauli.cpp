#include "qmetal/pauli.h"

#include <stdexcept>

namespace qmetal {

static uint32_t popcount64(uint64_t x) { return __builtin_popcountll(x); }

bool PauliString::valid(uint32_t n) const {
  const uint64_t all = (n >= 64) ? ~0ull : ((1ull << n) - 1ull);
  if ((x_mask | y_mask | z_mask) & ~all) return false;
  // A qubit may carry at most one Pauli.
  return !((x_mask & y_mask) | (x_mask & z_mask) | (y_mask & z_mask));
}

uint32_t PauliString::weight() const {
  return popcount64(x_mask | y_mask | z_mask);
}

PauliString PauliString::parse(const std::string &s) {
  PauliString p;
  if (s.size() > 64) throw std::runtime_error("PauliString: too long");
  for (size_t q = 0; q < s.size(); q++) {
    const uint64_t bit = 1ull << q;
    switch (s[q]) {
      case 'I': case 'i': break;
      case 'X': case 'x': p.x_mask |= bit; break;
      case 'Y': case 'y': p.y_mask |= bit; break;
      case 'Z': case 'z': p.z_mask |= bit; break;
      default:
        throw std::runtime_error(std::string("PauliString: bad character '") +
                                 s[q] + "'");
    }
  }
  return p;
}

std::string PauliString::to_string(uint32_t n) const {
  std::string s(n, 'I');
  for (uint32_t q = 0; q < n; q++) {
    const uint64_t bit = 1ull << q;
    if (x_mask & bit) s[q] = 'X';
    else if (y_mask & bit) s[q] = 'Y';
    else if (z_mask & bit) s[q] = 'Z';
  }
  return s;
}

PauliString PauliString::single(char axis, uint32_t q) {
  PauliString p;
  const uint64_t bit = 1ull << q;
  switch (axis) {
    case 'X': case 'x': p.x_mask = bit; break;
    case 'Y': case 'y': p.y_mask = bit; break;
    case 'Z': case 'z': p.z_mask = bit; break;
    case 'I': case 'i': break;
    default: throw std::runtime_error("PauliString::single: bad axis");
  }
  return p;
}

}  // namespace qmetal
