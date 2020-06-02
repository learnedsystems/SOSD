#pragma once

#include "index.hpp"
#include "sma.hpp"
#include "tree_mask.hpp"
#include "zone_mask.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include <immintrin.h>

namespace dtl {

/// A PSMA lookup table for the type T. Each table entry consists of an instance of V (e.g., a range in the
/// default implementation).
template<typename T, typename V>
class psma_table {
  static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                "Template parameter 'T' must be an integral unsigned type.");

public:

  // a PSMA consists of 256 range entries for each byte of T
  static constexpr u32 size = 256 * sizeof(T);

  V entries[size];

  // compute the PSMA slot for a given value
  inline u32
  get_slot(const T value) const noexcept {
    // number of remaining bytes (note: clz is undefined for 0)
    const u64 r = value ? (7 - (__builtin_clzll(value) >> 3)) : 0;
    // the index of the most significant non-zero byte
    const u64 m = (value >> (r << 3));
    // return the slot in lookup table
    return static_cast<u32>(m + (r << 8));
  }

};

/// PSMA implementation based on the paper of Lang et al. 'Data Blocks: Hybrid OLTP and OLAP on Compressed Storage
/// using both Vectorization and Compilation'
template<typename T>
class psma {
public:
  using value_t = typename std::remove_cv<T>::type;
  using unsigned_value_t = typename std::make_unsigned<value_t>::type;

  using sma_t = sma<value_t>;
  using table_t = psma_table<unsigned_value_t, range>;

  static constexpr u32 size = table_t::size;

  sma_t _sma;
  table_t table;

  struct psma_builder {
    psma& ref;

    inline void
    operator()(const T *const values, const size_t n, std::function<u1(u64)> is_null) noexcept {

      // requires two passes
      // 1. build the SMA
      auto sma_build = ref._sma.builder();
      sma_build(values, n, is_null);
      sma_build.done();

      // 2. populate the PSMA table
      for ($u32 i = 0; i != n; i++) {
        // compute the (unsigned) delta to the min value
        unsigned_value_t delta_value = values[i] - ref._sma.min_value;
        u64 slot_id = ref.table.get_slot(delta_value);
        auto& range = ref.table.entries[slot_id];
        if (range.is_empty()) {
          range = {i, i + 1};
        }
        else {
          range.end = i + 1;
        }
      }
    }

    inline void
    done() {
      // nothing to do here.
      // builder performs in-place updates
    }

  };

  inline psma_builder
  builder() {
    // reset table entries
    for ($u32 i = 0; i != size; i++) {
      table.entries[i].reset();
    }
    // return a builder instance
    return psma_builder { *this };
  }

  // c'tor
  psma() noexcept {
    // initialize the lookup table with empty range
    for ($u32 i = 0; i != size; i++) {
      table.entries[i].reset();
    }
  }

  inline range
  lookup(const predicate& p) const noexcept {
    value_t value = *reinterpret_cast<value_t*>(p.value_ptr);
    unsigned_value_t delta = value - _sma.min_value;
    value_t second_value; // in case of between predicates
    unsigned_value_t second_delta;

    u32 s = table.get_slot(delta);
    auto r = table.entries[s];
    if (p.comparison_operator == op::EQ) return r;

    $u32 b = 0;
    $u32 e = 0;
    switch (p.comparison_operator) {
      case op::LT:
      case op::LE:
        b = 0;
        e = s;
        break;
      case op::GT:
      case op::GE:
        b = s + 1;
        e = size;
        break;
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<value_t*>(p.second_value_ptr);
        second_delta = second_value - _sma.min_value;
        b = table.get_slot(delta);
        e = table.get_slot(second_delta);
        break;
    }
    for ($u32 i = b; i <= e; i++) {
      r = r | table.entries[i];
    }
    return r;
  }

};


/// A combination of a PSMA lookup table and a Zone Mask.
/// M = the number of bits per table entry.
template<typename T, u64 N, u64 M>
class psma_zone_mask {
public:

  using value_t = typename std::remove_cv<T>::type;
  using unsigned_value_t = typename std::make_unsigned<value_t>::type;

  using mask_t = zone_mask<N, M>;

  using sma_t = sma<value_t>;
  using table_t = psma_table<unsigned_value_t, mask_t>;

  static constexpr u32 size = table_t::size;

  sma_t _sma;
  table_t table;

  struct psma_builder {
    psma_zone_mask& ref;

    inline void
    operator()(const T *const values, const size_t n, std::function<u1(u64)> is_null) noexcept {
      // requires two passes

      // 1. build the SMA
      auto sma_build = ref._sma.builder();
      sma_build(values, n, is_null);
      sma_build.done();

      // 2. build populate PSMA table
      for ($u32 i = 0; i != n; i++) {
        // compute the (unsigned) delta to the min value
        unsigned_value_t delta_value = values[i] - ref._sma.min_value;
        u64 slot_id = ref.table.get_slot(delta_value);
        auto& entry = ref.table.entries[slot_id];
        entry.set(i);
      }
    }

    inline void
    done() {
      // nothing to do here.
      // builder performs in-place updates
    }

  };

  inline psma_builder
  builder() {
    // reset table entries
    for (uint32_t i = 0; i != size; i++) {
      table.entries[i].reset();
    }
    // return a builder instance
    return psma_builder { *this };
  }


//  inline void
//  update(const T* const values, const size_t n) noexcept {
//    for (uint32_t i = 0; i != n; i++) {
//      auto& entry = table.entries[table.get_slot(values[i])];
//      entry.set(i);
//    }
//  }

  inline mask_t
  lookup(const predicate& p) const noexcept {
    value_t value = *reinterpret_cast<value_t*>(p.value_ptr);
    unsigned_value_t delta = value - _sma.min_value;
    value_t second_value; // in case of between predicates
    unsigned_value_t second_delta;

    u32 s = table.get_slot(delta);
    auto r = table.entries[s];
    if (p.comparison_operator == op::EQ) return r;

    $u32 b = 0;
    $u32 e = 0;
    switch (p.comparison_operator) {
      case op::LT:
      case op::LE:
        b = 0;
        e = s;
        break;
      case op::GT:
      case op::GE:
        b = s + 1;
        e = size;
        break;
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<value_t*>(p.second_value_ptr);
        second_delta = second_value - _sma.min_value;
        b = table.get_slot(delta);
        e = table.get_slot(second_delta);
        break;
    }
    for ($u64 i = b; i <= e; i++) {
      r = r | table.entries[i];
    }
    return r;
  }

};


template<typename T, u64 N>
using psma_bitmask = psma_zone_mask<T, N, N>;


/// A combination of a PSMA lookup table and a Zone Mask.
/// M = the number of bits per table entry.
template<typename T, u64 N, u64 M>
class psma_tree_mask {
public:
  using value_t = typename std::remove_cv<T>::type;
  using unsigned_value_t = typename std::make_unsigned<value_t>::type;

  using sma_t = sma<value_t>;
  using mask_t = tree_mask<N, M>;
  using table_t = psma_table<unsigned_value_t, mask_t>;

  static constexpr u32 size = table_t::size;

  sma_t _sma;
  table_t table;

  struct psma_builder {
    psma_tree_mask& ref;

    inline void
    operator()(const value_t *const values, const size_t n, std::function<u1(u64)> is_null) noexcept {
      // built on top of psma_bitmap
      auto bm = std::make_unique<psma_bitmask<value_t, N>>();
      auto bm_build = bm->builder();
      bm_build(values, n, is_null);
      bm_build.done();

      // copy SMA
      ref._sma = bm->_sma;

      // encode bitmaps to tree masks
      for ($u32 i = 0; i != size; i++) {
        ref.table.entries[i] = bm->table.entries[i].data;
      }
    }

    inline void
    done() {
      // nothing to do here.
      // builder performs in-place updates
    }

  };

  inline psma_builder
  builder() {
    // reset table entries
    for ($u32 i = 0; i != size; i++) {
      table.entries[i].reset();
    }
    // return a builder instance
    return psma_builder { *this };
  }

//  inline void
//  update(const psma_bitmask<T, N>& src) noexcept {
//    for (uint32_t i = 0; i != table_t::size; i++) {
//      table.entries[i].set(src.table.entries[i].data);
//    }
//  }

  inline std::bitset<N>
  lookup(const predicate& p) const noexcept {
    value_t value = *reinterpret_cast<value_t*>(p.value_ptr);
    unsigned_value_t delta = value - _sma.min_value;
    value_t second_value; // in case of between predicates
    unsigned_value_t second_delta;

    u32 s = table.get_slot(delta);
    auto r = table.entries[s].get();
    if (p.comparison_operator == op::EQ) return r;

    $u32 b = 0;
    $u32 e = 0;
    switch (p.comparison_operator) {
      case op::LT:
      case op::LE:
        b = 0;
        e = s;
        break;
      case op::GT:
      case op::GE:
        b = s + 1;
        e = size;
        break;
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<value_t*>(p.second_value_ptr);
        second_delta = second_value - _sma.min_value;
        b = table.get_slot(delta);
        e = table.get_slot(second_delta);
        break;
    }
    for (size_t i = b; i <= e; i++) {
      r = r | table.entries[i].get();
    }
    return r;
  }

};


} // namespace dtl