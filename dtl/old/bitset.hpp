#pragma once

#include "adept.hpp"

template<u64 N>
struct bs {
  static_assert(is_power_of_two(N));

  $u64 v[N] __attribute__ ((aligned (64)));
  bs(){};
  bs(const bs<N>& other) {
    for (u64 i = 0; i < N; i++) {
      v[i] = other.v[i];
    }
  }
  bs operator&(const bs<N>& o) const noexcept {
    bs<N> n(*this);
    for (u64 i = 0; i < N; i++) {
      n.v[i] = v[i] & o.v[i];
    }
    return n;
  }

  u64 reduce_and() const noexcept {
    bs<N> n(*this);
    for (size_t i = 0; i < log_2<N>::value; i++) {
      u64 offset = 1 << i;
      u64 step = offset << 1;
      for (size_t j = 0; j < N; j += step) {
        n.v[j] = n.v[j] & n.v[j + offset];
      }
    }
    return n.v[0];
  }

  bool operator==(const bs<N>& o) const noexcept {
    bs<N> eq;
    for (u64 i = 0; i < N; i++) {
      eq.v[i] = (v[i] == o.v[i]) ? -1 : 0;
    }
    return eq.reduce_and();
  }
};
