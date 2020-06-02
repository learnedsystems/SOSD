#pragma once

#include "dtl.hpp"
#include "index.hpp"

#include <functional>
#include <limits>

namespace dtl {

template<typename T>
class sma {

public:

  using value_type = typename std::remove_cv<T>::type;

  value_type min_value;
  value_type max_value;
  $u1 has_null_values;

  struct sma_builder {
    sma& ref;

    inline void
    operator()(const value_type* const values, const size_t n, std::function<u1(u64)> is_null) noexcept {
      for ($u32 i = 0; i != n; i++) {
        if (is_null(i)) {
          ref.has_null_values = true;
          continue;
        }
        const value_type v = values[i];
        if (v < ref.min_value) {
          ref.min_value = v;
        }
        if (v > ref.max_value) {
          ref.max_value = v;
        }
      }
    }

    inline void
    done() {
      // nothing to do here.
      // builder performs in-place updates
    }

  };

  inline sma_builder
  builder() {
    // reset SMAs
    min_value = std::numeric_limits<value_type>::max();
    max_value = std::numeric_limits<value_type>::min();
    has_null_values = false;
    // return a builder instance
    return sma_builder { *this };
  }

  sma() {
    min_value = std::numeric_limits<value_type>::min();
    max_value = std::numeric_limits<value_type>::max();
    has_null_values = false;
  }

//  inline void
//  update(const T *const values, const size_t n) noexcept {
//    min_value = std::numeric_limits<value_type>::max();
//    max_value = std::numeric_limits<value_type>::min();
//    for ($u32 i = 0; i != n; i++) {
//      const T v = values[i];
//      if (v < min_value) {
//        min_value = v;
//      }
//      if (v > max_value) {
//        max_value = v;
//      }
//    }
//  }

  // query: x between value_lower and value_upper
  inline bool
  lookup(const op p, const value_type value_lower, const value_type value_upper) const noexcept {
    const bool left_inclusive = p == op::BETWEEN || p == op::BETWEEN_RO;
    const bool right_inclusive = p == op::BETWEEN || p == op::BETWEEN_LO;

    const value_type lo = value_lower + !left_inclusive;
    const value_type hi = value_upper - !right_inclusive;

    return ((lo >= min_value && lo <= max_value)
           || (hi >= min_value && hi <= max_value)) && lo <= hi;
  }

  // query: x op value
  inline bool
  lookup(const op p, const value_type value) const noexcept {
    switch (p) {
      case op::EQ:
        return lookup(op::BETWEEN_O, value, value);
      case op::LT:
        return lookup(op::BETWEEN_LO, std::numeric_limits<value_type>::min(), value);
      case op::LE:
        return lookup(op::BETWEEN_O, std::numeric_limits<value_type>::min(), value);
      case op::GT:
        return lookup(op::BETWEEN_RO, value, std::numeric_limits<value_type>::max());
      case op::GE:
        return lookup(op::BETWEEN_O, value, std::numeric_limits<value_type>::max());
    }
    return true;
  }

  inline bool
  lookup(const predicate& p) const noexcept {
    value_type value = *reinterpret_cast<value_type*>(p.value_ptr);
    value_type second_value; // in case of between predicates
    switch (p.comparison_operator) {
      case op::EQ:
      case op::LT:
      case op::LE:
      case op::GT:
      case op::GE:
        return lookup(p.comparison_operator, value);
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<value_type*>(p.second_value_ptr);
        return lookup(p.comparison_operator, value, second_value);
    }

  }

};

} // namespace dtl