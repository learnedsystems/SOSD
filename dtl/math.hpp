#pragma once

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>

#include <array>
#include <bitset>
#include <type_traits>

namespace dtl {

/// Ranks a permutation of order N to an integer in the range of [0, N!).
/// The mapping is bijective, see unrank() function.
/// Both, the rank as well as the unrank functions are based on the algorithm of
/// Myrvold and Ruskey which is in O(N).
template<typename T, size_t N>
uint64_t rank(std::array<T, N> pi) {
  static_assert(std::is_unsigned<T>::value, "Template parameter T must be an unsigned integer.");
  std::array<T, N> inv_pi;
  for (size_t i = 0; i < N; i++) {
    inv_pi[pi[i]] = i;
  }
  std::array<T, N> s;
  for (size_t i = N; i > 0; i--) {
    s[i - 1] = pi[i - 1];
    std::swap(pi[i - 1], pi[inv_pi[i - 1]]);
    std::swap(inv_pi[s[i - 1]], inv_pi[i - 1]);
  }
  uint64_t r = 0;
  for (size_t i = 1; i <= N; i++) {
    r = r * i + s[i - 1];
  }
  return r;
}

template<typename T, size_t N>
std::array<T, N>
unrank(uint64_t r) {
  static_assert(std::is_unsigned<T>::value, "Template parameter T must be an unsigned integer.");
  std::array<T, N> pi;
  for (size_t i = 0; i < N; i++) {
    pi[i] = i;
  }
  for (size_t i = N; i > 0; i--) {
    std::swap(pi[i - 1], pi[r % i]);
    r = r / i;
  }
  return pi;
}

/// Computes (iteratively) the factorial of n.
constexpr uint64_t
factorial(const uint64_t n) {
  uint64_t factorial = n;
  for (size_t i = n; i > 1; i--) {
    factorial *= i;
  }
  return factorial;
}

/// Computes the binomial coefficient N choose K.
constexpr uint64_t
n_choose_k(const uint64_t n, const uint64_t k) {
  if (k == 0) {
    return 1;
  }
  if (2 * k > n) {
    return n_choose_k(n, n - k);
  }
  uint64_t result = n - k + 1;
  for (size_t i = 2; i <= k; i++) {
    result = result * (n - k + i);
    result = result / i;
  }
  return result;
};

/// Computes the n-th catalan number.
/// Remarks:
///   n = 20 exceeds 32 bit (n = 19 requires 31 bits)
///   n = 37 exceeds 64 bit
//FIXME: returns wrong results for n > 33
constexpr uint64_t
catalan_number(const uint64_t n) {
  return n_choose_k(2 * n, n) / (n + 1);
}

constexpr size_t
ballot_number(const size_t i, const size_t j) {
  const size_t n = i + 1;
  const size_t k = (i + j) / 2 + 1;
  return static_cast<size_t>(((static_cast<double>(j) + 1) / (static_cast<double>(i) + 1)) * n_choose_k(n, k));
};

/// Computes the number of paths from (i,j) to (2n,0)
constexpr size_t
number_of_paths(const size_t n, const size_t i, const size_t j) {
  return ballot_number(2 * n - i, j);
};

/// Unrank a binary tree with N inner nodes
/// Returns bit encoding of the inner nodes
template<size_t N>
size_t rank_tree(std::bitset<N*2> encoding) {
  //Ranking(b1b2 . . . bn)
  size_t b[N];
  size_t pos = 0;
  for (size_t j = 0; j < encoding.size(); j++) {
    if (encoding[j]) {
      b[pos] = j + 1;
      pos++;
    }
  }

  size_t c[N];
  c[0] = 2;
  for (size_t j = 1; j < N; j++) {
    c[j] = std::max(b[j] + 1, 2 * (j+1));
  }
  size_t nr = 1;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = c[i]; j < b[i]; j++) {
      nr = nr + number_of_paths(N, N - i, N + i - j);
    }
  }
  return nr;
}

/// Unrank a binary tree with N inner nodes
/// Returns bit encoding of the inner nodes
template<size_t N>
std::bitset<N*2> unrank_tree(size_t rank) {
	size_t open = 1;
	size_t close = 0;
	size_t pos = 0;
	std::bitset<N*2> encoding;
	while (encoding.count() < N) {
		size_t k = number_of_paths(N, open + close, open - close);
		if (k <= rank) {
			rank = rank - k;
			close++;
		}
    else {
			encoding.set(pos);
			open++;
		}
		pos++;
	}
	return encoding;
}

constexpr bool is_power_of_two(size_t x) {
  return x == 1 ? true : (x && ((x & (x - 1)) == 0));
}

constexpr u64 next_power_of_two(u64 value) {
  return 1ull << ((sizeof(u64) << 3) - __builtin_clzll(value - 1));
}

constexpr u64 prev_power_of_two(u64 value) {
  return next_power_of_two(value) >> 1;
}

struct trunc {
  static size_t byte(uint64_t max) {
    if (!max) return 0;
    const uint64_t r = (8 - (__builtin_clzll(max) >> 3));
    return next_power_of_two(r);
  }
  static size_t bit(uint64_t max) {
    return max ? (64 - (__builtin_clzll(max))) : 0;
  }
};

__forceinline__ __host__ __device__
constexpr u32
log_2(const u32 n) {
  return 8 * sizeof(u32) - dtl::bits::lz_count(n) - 1;
};

__forceinline__ __host__ __device__
constexpr u64
log_2(const u64 n) {
  return 8 * sizeof(u64) - dtl::bits::lz_count(n) - 1;
};


/// Compile-time template expansions
namespace ct {

  /// Computes N! at compile time.
  template<size_t N>
  struct factorial {
    enum : size_t {
      value = N * factorial<N - 1>::value
    };
  };

  template<>
  struct factorial<0> {
    enum : size_t {
      value = 1
    };
  };


  template<size_t N, size_t K>
  struct n_choose_k {
    static const size_t n = N;
    static const size_t k = (2 * K > N) ? N - K : K;
    enum : size_t {
      value = n_choose_k<N - 1, K - 1>::value + n_choose_k<N - 1, K>::value
    };
  };

  template<size_t N>
  struct n_choose_k<N, 0> {
    enum : size_t {
      value = 1
    };
  };

  template<size_t K>
  struct n_choose_k<0, K> {
    enum : size_t {
      value = 0
    };
  };

  template<size_t N>
  struct n_choose_k<N, N> {
    enum : size_t {
      value = 1
    };
  };


  template<size_t N>
  struct catalan_number {
    enum : size_t {
      value = n_choose_k<2 * N, N>::value / (N + 1)
    };
  };

  template<size_t i, size_t j>
  struct ballot_number {
    static const size_t n = i + 1;
    static const size_t k = (i + j) / 2 + 1;
    enum : size_t {
      value = static_cast<size_t>(((static_cast<double>(j) + 1) / (static_cast<double>(i) + 1)) * n_choose_k<n, k>::value)
    };
  };

  /// Computes the number of paths from (i,j) to (2n,0)
  template<size_t n, size_t i, size_t j>
  struct number_of_paths {
    enum : size_t {
      value = ballot_number<2 * n - i, j>::value
    };
  };

  template<u64 n>
  struct lz_count_u32 {
    static constexpr u64 value = __builtin_clz(n);
  };

  template<u64 n>
  struct lz_count_u64 {
    static constexpr u64 value = __builtin_clzll(n);
  };

  template<size_t n>
  struct log_2{
    enum : size_t {
      value = 8 * sizeof(size_t) - lz_count_u64<n>::value - 1
    };
  };

  template<u32 n>
  struct log_2_u32{
    enum : u32 {
      value = n == 0 ? 0 : 8 * sizeof(u32) - lz_count_u32<n>::value - 1
    };
  };

  template<u64 n>
  struct log_2_u64{
    enum : u64 {
      value = n == 0 ? 0 : 8 * sizeof(u64) - lz_count_u64<n>::value - 1
    };
  };


  template<u64 n>
  struct pop_count {
    static constexpr u64 value = (n & 1) + pop_count<(n >> 1)>::value;
  };

  template<>
  struct pop_count<0ull> {
    static constexpr u64 value = 0;
  };

}


} // namespace dtl
