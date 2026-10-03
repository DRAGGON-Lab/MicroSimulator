#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "cm/stokes_flow.hpp"

namespace cm::detail {
using BodyVector = std::array<double, 3>;

inline double body_dot(const BodyVector& a, const BodyVector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline BodyVector body_rotate(const std::array<double, 4>& q, const BodyVector& v) {
  const BodyVector t{2 * (q[2] * v[2] - q[3] * v[1]), 2 * (q[3] * v[0] - q[1] * v[2]),
                     2 * (q[1] * v[1] - q[2] * v[0])};

  return {v[0] + q[0] * t[0] + q[2] * t[2] - q[3] * t[1],
          v[1] + q[0] * t[1] + q[3] * t[0] - q[1] * t[2],
          v[2] + q[0] * t[2] + q[1] * t[1] - q[2] * t[0]};
}

inline std::pair<BodyVector, BodyVector> body_endpoints(const CapsuleBody& body) {
  const auto axis = body_rotate(body.orientation, {1, 0, 0});
  auto a = body.position, b = body.position;

  for (std::size_t d = 0; d < 3; ++d) {
    a[d] -= 0.5 * body.length * axis[d];
    b[d] += 0.5 * body.length * axis[d];
  }

  return {a, b};
}

inline std::pair<BodyVector, BodyVector> body_closest_segments(const BodyVector& a0,
                                                               const BodyVector& a1,
                                                               const BodyVector& b0,
                                                               const BodyVector& b1) {
  BodyVector da{}, db{}, r{};

  for (std::size_t d = 0; d < 3; ++d) {
    da[d] = a1[d] - a0[d];
    db[d] = b1[d] - b0[d];
    r[d] = a0[d] - b0[d];
  }

  const double a = body_dot(da, da), b = body_dot(da, db), c = body_dot(da, r),
               e = body_dot(db, db), f = body_dot(db, r);
  double s = 0, t = 0;

  if (a == 0 && e > 0) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else if (e == 0 && a > 0) {
    s = std::clamp(-c / a, 0.0, 1.0);
  } else if (a > 0 && e > 0) {
    const double denominator = a * e - b * b;

    if (denominator > 1e-14 * a * e) {
      s = std::clamp((b * f - c * e) / denominator, 0.0, 1.0);
    }

    t = (b * s + f) / e;

    if (t < 0) {
      t = 0;
      s = std::clamp(-c / a, 0.0, 1.0);
    } else if (t > 1) {
      t = 1;
      s = std::clamp((b - c) / a, 0.0, 1.0);
    }
  }

  BodyVector pa{}, pb{};

  for (std::size_t d = 0; d < 3; ++d) {
    pa[d] = a0[d] + s * da[d];
    pb[d] = b0[d] + t * db[d];
  }

  return {pa, pb};
}

inline double body_segment_box_distance_squared(const BodyVector& start, const BodyVector& end,
                                                const BodyVector& lo, const BodyVector& hi) {
  BodyVector delta{};
  std::vector<double> cuts{0, 1};

  for (std::size_t a = 0; a < 3; ++a) {
    delta[a] = end[a] - start[a];

    if (delta[a] != 0) {
      for (double bound : {lo[a], hi[a]}) {
        const double t = (bound - start[a]) / delta[a];

        if (t > 0 && t < 1) {
          cuts.push_back(t);
        }
      }
    }
  }

  std::sort(cuts.begin(), cuts.end());
  auto value = [&](double t) {
    double d = 0;

    for (std::size_t a = 0; a < 3; ++a) {
      const double x = start[a] + t * delta[a];
      const double r = x - std::clamp(x, lo[a], hi[a]);
      d += r * r;
    }

    return d;
  };
  double best = std::min(value(0), value(1));

  for (std::size_t i = 1; i < cuts.size(); ++i) {
    const double t = (cuts[i] + cuts[i - 1]) / 2;
    double aa = 0, bb = 0;

    for (std::size_t a = 0; a < 3; ++a) {
      const double x = start[a] + t * delta[a];

      if (x >= lo[a] && x <= hi[a]) {
        continue;
      }

      const double bound = x < lo[a] ? lo[a] : hi[a];
      aa += delta[a] * delta[a];
      bb += delta[a] * (start[a] - bound);
    }

    best = std::min(best, value(cuts[i]));

    if (aa > 0) {
      best = std::min(best, value(std::clamp(-bb / aa, cuts[i - 1], cuts[i])));
    }
  }

  return best;
}

// Bao et al., arXiv:1505.07529, section 2.2.
// For r in [0,1], returns phi(r-3),...,phi(r+2). The rationalized root avoids
// cancellation at integer nodes. This C3 six-point delta obeys all four moments.
inline std::array<double, 6> immersed_delta_weights(double r) {
  const double K = 59.0 / 60 - std::sqrt(29.0) / 20;
  const double r2 = r * r, r3 = r2 * r;
  // Derive beta from the sum-of-squares condition; the extra r multiplying
  // (K+r^2) in the arXiv HTML rendering of equation 2.16 is inconsistent with it.
  const double beta = 9.0 / 4 - 1.5 * (K + r2) + (22.0 / 3 - 7 * K) * r - 7.0 / 3 * r3;
  const double a = (3 * K - 1) * r + r3, b = (4 - 3 * K) * r - r3;
  const double gamma = -11 * r2 / 32 + 3 * (2 * K + r2) * r2 / 32 + a * a / 72 + b * b / 18;
  const double f = -2 * gamma / (beta + std::sqrt(std::max(0.0, beta * beta - 112 * gamma)));

  return {f,
          -3 * f - 1.0 / 16 + (K + r2) / 8 + a / 12,
          2 * f + 0.25 + b / 6,
          2 * f + 5.0 / 8 - (K + r2) / 4,
          -3 * f + 0.25 - b / 6,
          f - 1.0 / 16 + (K + r2) / 8 - a / 12};
}

struct BodyMarker {
  BodyVector offset;
  std::uint32_t body;
  double axial_fraction;
};

inline std::vector<BodyMarker> capsule_surface_markers(const std::vector<CapsuleBody>& bodies,
                                                       double spacing) {
  std::vector<BodyMarker> markers;

  for (std::size_t body = 0; body < bodies.size(); ++body) {
    const auto& b = bodies[body];
    auto ring = [&](double x, double radius) {
      const unsigned count =
          std::max(4U, static_cast<unsigned>(std::ceil(2 * std::numbers::pi * radius / spacing)));

      for (unsigned j = 0; j < count; ++j) {
        const double theta = 2 * std::numbers::pi * (double(j) + 0.5) / count;
        markers.push_back(
            {body_rotate(b.orientation, {x, radius * std::cos(theta), radius * std::sin(theta)}),
             static_cast<std::uint32_t>(body),
             b.length == 0 ? std::copysign(0.5, x) : std::clamp(x / b.length, -0.5, 0.5)});
      }
    };
    const unsigned cylinders = static_cast<unsigned>(std::ceil(b.length / spacing));

    for (unsigned i = 0; i < cylinders; ++i) {
      ring(b.length * ((double(i) + 0.5) / cylinders - 0.5), b.radius);
    }

    const unsigned caps =
        std::max(1U, static_cast<unsigned>(std::ceil(std::numbers::pi * b.radius / (2 * spacing))));

    for (unsigned i = 0; i < caps; ++i) {
      const double theta = (double(i) + 0.5) * std::numbers::pi / (2 * caps);

      for (int sign : {-1, 1}) {
        ring(sign * (0.5 * b.length + b.radius * std::cos(theta)), b.radius * std::sin(theta));
      }
    }
  }

  return markers;
}
}  // namespace cm::detail
