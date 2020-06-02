#pragma once

#include "dtl.hpp"

namespace dtl {

enum class op {
  GE, GT, EQ, LT, LE,
  IS_NULL, IS_NOT_NULL,
  IN,
  BETWEEN, BETWEEN_LO, BETWEEN_RO, BETWEEN_O
};

enum class logical_op {
  AND, OR
};

template<typename T>
struct between {
  bool
  operator()(const T& lower, const T& value, const T& upper) const noexcept {
    return std::logical_and<u1>()(
        std::greater_equal<T>()(value, lower),
        std::less_equal<T>()(value, upper)
    );
  }
};


template<typename T>
struct between_left_open {
  bool
  operator()(const T& lower, const T& value, const T& upper) const noexcept {
    return std::logical_and<u1>()(
        std::greater<T>()(value, lower),
        std::less_equal<T>()(value, upper)
    );
  }
};


template<typename T>
struct between_right_open {
  bool
  operator()(const T& lower, const T& value, const T& upper) const noexcept {
    return std::logical_and<u1>()(
        std::greater_equal<T>()(value, lower),
        std::less<T>()(value, upper)
    );
  }
};


template<typename T>
struct between_open {
  bool
  operator()(const T& lower, const T& value, const T& upper) const noexcept {
    return std::logical_and<u1>()(
        std::greater<T>()(value, lower),
        std::less<T>()(value, upper)
    );
  }
};


/// a monadic predicate (e.g., attr OP const)
struct predicate {
  op comparison_operator;
  void* value_ptr;
  void* second_value_ptr = nullptr; // in case of BETWEEN
};


struct range {
  $u32 begin;
  $u32 end;

  inline void
  reset() {
    begin = 0;
    end = 0;
  }

  inline bool
  is_empty() const {
    return begin == end;
  }

  inline range
  operator|(const range &other) {
    if (is_empty()) return other;
    if (other.is_empty()) return range{begin, end};
    return range{std::min(begin, other.begin), std::max(end, other.end)};
  }

  inline range
  operator&(const range &other) {
    if (is_empty()) return range{0, 0};
    if (other.is_empty()) return range{0, 0};
    range r{std::max(begin, other.begin), std::min(end, other.end)};
    if (r.begin >= r.end) r = {0, 0};
    return r;
  }
};

} // namespace dtl