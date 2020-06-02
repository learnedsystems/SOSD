#pragma once

#include <random>
#include <math.h>
#include <boost/math/distributions/poisson.hpp>
#include <dtl/dtl.hpp>

namespace dtl {
namespace bloomfilter {

/// Computes an approximation of the false positive probability.
/// Assuming independence for the probabilities of each bit being set.
f64
fpr(u64 m,
    u64 n,
    u64 k) {
  return std::pow(1.0 - std::pow(1.0 - (1.0 / m), k * n), k);
}

f64
fpr_k_partitioned(u64 m,
                  u64 n,
                  u64 k) {
  f64 c = (m * 1.0) / n;
  return fpr((m * 1.0)/k, n, 1);
}
//f64
//fpr_k_partitioned(u64 m,
//                  u64 n,
//                  u64 k) {
//  f64 c = (m * 1.0) / n;
//  return std::pow(1.0 - std::exp(-(k*1.0) / c), k);
//}


f64
fpr_blocked(u64 m,
            u64 n,
            f64 k,
            u64 B, /* block size in bits */
            f64 epsilon = 0.001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = B / c;
  boost::math::poisson_distribution<> poisson(lambda);

  for ($i32 i = 0; i < 10000; i++) {
    f += boost::math::pdf(poisson, i) * fpr(B, i, k);
  }
  return f;
}

f64
fpr_blocked_k_partitioned(u64 m,
                          u64 n,
                          u64 k,
                          u64 B, /* block size in bits */
                          f64 epsilon = 0.001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = (B * 1.0) / c;
  boost::math::poisson_distribution<> poisson(lambda);

  std::random_device rd;
  std::mt19937 gen(rd());
  for ($i32 i = 0; i < 1000; i++) {
    f += boost::math::pdf(poisson, i) * fpr_k_partitioned(B, i, k);
  }
  return f;
}

f64
fpr_blocked_sectorized(u64 m,
                       u64 n,
                       u64 k,
                       u64 B, /* block size in bits */
                       u64 S, /* sector size in bits */
                       f64 epsilon = 0.001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = (B * 1.0) / c;
  $f64 s = (B * 1.0) / S;
  boost::math::poisson_distribution<> poisson(lambda);

  std::random_device rd;
  std::mt19937 gen(rd());
  for ($i32 i = 0; i < 1000; i++) {
    f += boost::math::pdf(poisson, i) * std::pow(fpr(S, i, (k * 1.0)/s), s);
  }
  return f;
}

} // namespace bloomfilter

namespace cuckoofilter {

//// TODO
f64
fpr(u64 associativity,
    u64 tag_bitlength,
    f64 load_factor) {
//  return (2.0 /*k=2*/ * associativity * load_factor) / (std::pow(2, tag_bitlength) - 1); // no duplicates
  return 1 - std::pow(1.0 - 1 / (std::pow(2.0, tag_bitlength) - 1), 2.0 * associativity * load_factor); // counting - with duplicates
}


} // namespace cuckoofilter
} // namespace dtl