#pragma once

#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"
#include "bloomfilter_h3.hpp"

namespace dtl {

/// A high-performance blocked Bloom filter, whereas a block corresponds to a word.
/// The hash bits are provided by two hash function.
template<typename Tk,      // the key type
    template<typename Ty> class HashFn,     // the first hash function to use
    template<typename Ty> class HashFn2,    // the second hash function to use
    template<typename Ty> class HashFn3,    // the second hash function to use
    typename Tw = u64,     // the word type to use for the bitset
    typename Alloc = std::allocator<Tw>,
    u32 K = 5,             // the number of bits set per inserted element
    u1 Sectorized = false
>
struct bloomfilter_h3_mod {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using allocator_t = Alloc;
  using size_t = $u64;
//  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;


  // Inspect the given hash function
  static_assert(
      std::is_same<decltype(HashFn<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");
  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr u32 hash_fn_cnt = 3;


  // The number of hash functions to use.
  static constexpr u32 k = K;
  static_assert(k > 1, "Parameter 'k' must be at least '2'.");

  // Split each word into multiple sectors (sub words, with a length of a power of two).
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = Sectorized;

// incompatible with C++11
//  static constexpr u32 compute_sector_cnt() {
//    if (!sectorized) return 1;
//    u32 k_pow_2 = dtl::next_power_of_two(k);
//    static_assert((word_bitlength / k_pow_2) != 0, "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
//    return word_bitlength / (word_bitlength / k_pow_2);
//  }

  static constexpr u32 compute_sector_cnt() {
    static_assert(!sectorized || ((word_bitlength / dtl::next_power_of_two(k)) != 0), "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
    return (!sectorized) ? 1
                         : word_bitlength / (word_bitlength / dtl::next_power_of_two(k));
  }
  static constexpr u32 sector_cnt = compute_sector_cnt();
  static constexpr u32 sector_bitlength = word_bitlength / sector_cnt;
  // the number of bits needed to address the individual bits within a sector
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask() { return sector_bitlength - 1; }

  // the number of remaining bits of the FIRST hash value (used to identify the word)
  static constexpr i32 remaining_hash_bit_cnt = static_cast<i32>(hash_value_bitlength) - sector_bitlength_log2;
  static constexpr u64 min_m = 2 * word_bitlength; // Using only one word would cause undefined behaviour in bit shifts later on.
  static constexpr u64 max_m = (1ull << remaining_hash_bit_cnt) * word_bitlength;

  // ---- Members ----
  const hash_value_t word_cnt; // the number of words/blocks
  const hash_value_t word_cnt_log2; // The (minimum) number of bits required to address the individual words of the bitvector
  const dtl::fast_divisor_u32_t fast_divisor;
  const allocator_t allocator;
  std::vector<word_t, allocator_t> word_array;
  // ----


  static constexpr
  size_t
  determine_word_cnt(const size_t length) {
    u32 desired_word_cnt = (length + (word_bitlength - 1)) / word_bitlength;
    u32 actual_word_cnt = dtl::next_cheap_magic(desired_word_cnt).divisor;
    u32 min_word_cnt = static_cast<size_t>(min_m / word_bitlength);
    return std::max(actual_word_cnt, min_word_cnt);
  }


  __forceinline__
  size_t
  length() const noexcept {
    return word_array.size() * sizeof(word_t) * 8;
  }


  /// C'tor
  explicit
  bloomfilter_h3_mod(const size_t length,
                     const allocator_t allocator = allocator_t())
      : word_cnt(determine_word_cnt(length)),
        word_cnt_log2(dtl::log_2(dtl::next_power_of_two(word_cnt))),
        fast_divisor(dtl::next_cheap_magic(word_cnt)),
        allocator(allocator),
        word_array(word_cnt, 0, this->allocator) {
    if ((word_cnt * word_bitlength) > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// Copy c'tor
  bloomfilter_h3_mod(const bloomfilter_h3_mod&) = default;
  bloomfilter_h3_mod(const bloomfilter_h3_mod& other,
                     const allocator_t& allocator)
      : word_cnt(other.word_cnt),
        word_cnt_log2(other.word_cnt_log2),
        fast_divisor(other.fast_divisor),
        allocator(allocator),
        word_array(other.word_array.begin(), other.word_array.end(), this->allocator) { }

  ~bloomfilter_h3_mod() {
    word_array.clear();
    word_array.shrink_to_fit();
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator type)
  template<typename AllocOfCopy = Alloc>
  bloomfilter_h3_mod<Tk, HashFn, HashFn2, HashFn3, Tw, AllocOfCopy, K, Sectorized>
  make_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using return_t = bloomfilter_h3_mod<Tk, HashFn, HashFn2, HashFn3, Tw, AllocOfCopy, K, Sectorized>;
    return_t bf_copy(word_cnt * word_bitlength, alloc);
    bf_copy.word_array.clear();
    bf_copy.word_array.insert(bf_copy.word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  bloomfilter_h3_mod<Tk, HashFn, HashFn2, HashFn2, Tw, AllocOfCopy, K, Sectorized>*
  make_heap_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using bf_t = bloomfilter_h3_mod<Tk, HashFn, HashFn2, HashFn2, Tw, AllocOfCopy, K, Sectorized>;
    bf_t* bf_copy = new bf_t(word_cnt * word_bitlength, alloc);
    bf_copy->word_array.clear();
    bf_copy->word_array.insert(bf_copy->word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }


  __forceinline__ __host__ __device__
  const hash_value_t
  which_word(const hash_value_t hash_val) const noexcept {
    const auto word_idx = dtl::fast_mod_u32(hash_val >> (hash_value_bitlength - word_cnt_log2), fast_divisor);
    return word_idx;
  }


  __forceinline__ __unroll_loops__ __host__ __device__
  static word_t
  which_bits(const hash_value_t first_hash_val,
             const hash_value_t second_hash_val,
             const hash_value_t third_hash_val,
             const size_t word_cnt_log2) noexcept {
    u32 first_bit_idx = (first_hash_val >> (hash_value_bitlength - word_cnt_log2 - sector_bitlength_log2)) & sector_mask();
    word_t word = word_t(1) << first_bit_idx;
    constexpr u32 k_2nd = boost::static_unsigned_min<k, 6u>::value;
    for ($u32 i = 1; i < k_2nd; i++) {
      u32 shift = (hash_value_bitlength - 2) - (i * sector_bitlength_log2);
      u32 bit_idx = (second_hash_val >> shift) & sector_mask();
      u32 sector_offset = (i * sector_bitlength) & word_bitlength_mask;
      word |= word_t(1) << (bit_idx + sector_offset);
    }
    for ($u32 i = k_2nd; i < boost::static_unsigned_min<k, k_2nd + 5u>::value; i++) {
      u32 shift = (hash_value_bitlength - 2) - ((i-k_2nd) * sector_bitlength_log2);
      u32 bit_idx = (third_hash_val >> shift) & sector_mask();
      u32 sector_offset = ((i-k_2nd) * sector_bitlength) & word_bitlength_mask;
      word |= word_t(1) << (bit_idx + sector_offset);
    }
    return word;
  }


  __forceinline__
  void
  insert(const key_t& key) noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    const hash_value_t third_hash_val = HashFn3<key_t>::hash(key);
    const hash_value_t word_idx = which_word(first_hash_val);
    word_t word = word_array[word_idx];
    word |= which_bits(first_hash_val, second_hash_val, third_hash_val, word_cnt_log2);
    word_array[word_idx] = word;
  }


  __forceinline__
  u1
  contains(const key_t& key) const noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    const hash_value_t third_hash_val = HashFn3<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val);
    const word_t search_mask = which_bits(first_hash_val, second_hash_val, third_hash_val, word_cnt_log2);
    return (word_array[word_idx] & search_mask) == search_mask;
  }


  u64
  popcnt() const noexcept {
    return std::accumulate(word_array.begin(), word_array.end(), 0ull,
                           [](u64 cntr, word_t word) { return cntr + dtl::bits::pop_count(word); });
  }


  f64
  load_factor() const noexcept {
    f64 m = word_cnt * word_bitlength;
    return popcnt() / m;
  }


  u32
  hash_function_cnt() const noexcept {
    return hash_fn_cnt;
  }


  void
  print_info() const noexcept {
    std::cout << "-- bloomfilter parameters --" << std::endl;
    std::cout << "static" << std::endl;
    std::cout << "  h:                    " << hash_fn_cnt << std::endl;
    std::cout << "  k:                    " << k << std::endl;
    std::cout << "  word bitlength:       " << word_bitlength << std::endl;
    std::cout << "  hash value bitlength: " << hash_value_bitlength << std::endl;
    std::cout << "  sectorized:           " << (sectorized ? "true" : "false") << std::endl;
    std::cout << "  sector count:         " << sector_cnt << std::endl;
    std::cout << "  sector bitlength:     " << sector_bitlength << std::endl;
    std::cout << "  hash bits per sector: " << sector_bitlength_log2 << std::endl;
    std::cout << "  hash bits per word:   " << (k * sector_bitlength_log2) << std::endl;
    std::cout << "  hash bits wasted:     " << (sectorized ? (word_bitlength - (sector_bitlength * k)) : 0) << std::endl;
    std::cout << "  remaining hash bits:  " << remaining_hash_bit_cnt << std::endl;
    std::cout << "  max m:                " << max_m << std::endl;
    std::cout << "  max size [MiB]:       " << (max_m / 8.0 / 1024.0 / 1024.0 ) << std::endl;
    std::cout << "dynamic" << std::endl;
    std::cout << "  m:                    " << (word_cnt * word_bitlength) << std::endl;
    f64 size_MiB = (word_cnt * word_bitlength) / 8.0 / 1024.0 / 1024.0;
    if (size_MiB < 1) {
      std::cout << "  size [KiB]:           " << (size_MiB * 1024) << std::endl;
    }
    else {
      std::cout << "  size [MiB]:           " << size_MiB << std::endl;
    }
    std::cout << "  population count:     " << popcnt() << std::endl;
    std::cout << "  load factor:          " << load_factor() << std::endl;
  }


  void
  print() const noexcept {
    std::cout << "-- bloomfilter dump --" << std::endl;
    $u64 i = 0;
    for (const word_t word : word_array) {
      std::cout << std::bitset<word_bitlength>(word);
      i++;
      if (i % (128 / word_bitlength) == 0) {
        std::cout << std::endl;
      }
      else {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }


};

} // namespace dtl
