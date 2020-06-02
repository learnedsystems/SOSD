#pragma once

#ifndef _DTL_STORAGE_INCLUDED
#error "Never use <dtl/storage/scan.hpp> directly; include <dtl/storage.hpp> instead."
#endif

#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bitset.hpp>
#include <dtl/index.hpp>
#include <dtl/zone_mask.hpp>

#include "column_block.hpp"
#include "types.hpp"

namespace dtl {

//TODO should go somewhere else
template<u64 N>
using selection_mask = dtl::bitset<N>;
using selection_vector = std::vector<$u32>;


//TODO should go somewhere else
template<u64 N>
selection_mask<N>
to_mask(const dtl::range& r) {
  selection_mask<N> m;
  for (std::size_t i = r.begin; i < r.end; i++) {
    m[i] = true;
  }
  return m;
}

template<u64 N>
selection_mask<N>
to_mask(u1 b) {
  selection_mask<N> m;
  if (b) {
    m.flip();
  }
  return m;
}

template<u64 N, u64 M>
selection_mask<N>
to_mask(dtl::zone_mask<N, M> zm) {
  return dtl::zone_mask<N, M>::decode(zm.data);
}

template<u64 N>
selection_mask<N>
to_mask(selection_mask<N> m) {
  return m;
}



namespace detail {

template<typename T, u64 N>
inline void
select(const auto fn, const auto logical_op,
       const dtl::column_block<T, N>& col, const T val,
       dtl::selection_mask<N>& mask) noexcept {

  for (std::size_t i = 0; i < col.size(); i++) {
    if (typeid(logical_op) == typeid(std::logical_and<$u1>) && !mask[i]) continue; // avoid loading the actual column value, only in conjunctions
    mask[i] = logical_op(mask[i], fn(col[i], val));
  }
}


template<typename T, u64 N>
inline void
select_between(const auto fn, const auto logical_op,
               const T lower, const dtl::column_block<T, N>& col, const T upper,
               dtl::selection_mask<N>& mask) noexcept {

  for (std::size_t i = 0; i < col.size(); i++) {
    if (typeid(logical_op) == typeid(std::logical_and<$u1>) && !mask[i]) continue; // avoid loading the actual column value, only in conjunctions
    mask[i] = logical_op(mask[i], fn(lower, col[i], upper));
  }
}

/// evaluates a given predicate on the column and updates the selection mask
template<typename T, u64 N, typename LogicalOp>
inline void
select(const dtl::column_block<T, N>& col,
       const dtl::predicate& p, const LogicalOp logical_op, dtl::selection_mask<N>& m) {

  T value = *reinterpret_cast<T*>(p.value_ptr);
  T second_value; // in case of between predicates
  switch (p.comparison_operator) {
    case dtl::op::EQ:
      return detail::select(std::equal_to<T>(), logical_op, col, value, m);
    case dtl::op::LT:
      return detail::select(std::less<T>(), logical_op, col, value, m);
    case dtl::op::LE:
      return detail::select(std::less_equal<T>(), logical_op, col, value, m);
    case dtl::op::GT:
      return detail::select(std::greater<T>(), logical_op, col, value, m);
    case dtl::op::GE:
      return detail::select(std::greater_equal<T>(), logical_op, col, value, m);
    case dtl::op::BETWEEN:
      second_value = *reinterpret_cast<T*>(p.second_value_ptr);
      return detail::select_between(dtl::between<T>(), logical_op, value, col, second_value, m);
    case dtl::op::BETWEEN_LO:
      second_value = *reinterpret_cast<T*>(p.second_value_ptr);
      return detail::select_between(dtl::between_left_open<T>(), logical_op, value, col, second_value, m);
    case dtl::op::BETWEEN_RO:
      second_value = *reinterpret_cast<T*>(p.second_value_ptr);
      return detail::select_between(dtl::between_right_open<T>(), logical_op, value, col, second_value, m);
    case dtl::op::BETWEEN_O:
      second_value = *reinterpret_cast<T*>(p.second_value_ptr);
      return detail::select_between(dtl::between_open<T>(), logical_op, value, col, second_value, m);
  }
//  unreachable();
};

/// evaluates a given predicate on the column and updates the selection mask
template<typename T, u64 N>
inline void
select(const dtl::column_block<T, N>& col,
       const dtl::predicate& pred, const dtl::logical_op con, dtl::selection_mask<N>& mask) {

  switch (con) {
    case dtl::logical_op::AND:
      return detail::select(col, pred, std::logical_and<$u1>(), mask);
    case dtl::logical_op::OR:
      return detail::select(col, pred, std::logical_or<$u1>(), mask);
  }
//  unreachable();
};



} // namespace detail

/// evaluates a given predicate on the column and updates the selection mask
template<typename T, u64 N>
inline void
select(const dtl::column_block<T, N>& col,
       const dtl::predicate& pred, const dtl::logical_op con, dtl::selection_mask<N>& mask) {
  detail::select(col, pred, con, mask);
};


template<u64 N>
inline void
select(const dtl::column_block_base<N>& col, const dtl::rtt type,
       const dtl::predicate& pred, const dtl::logical_op con, dtl::selection_mask<N>& mask) {
  switch (type) {
#define DTL_GENERATE(T) \
    case dtl::rtt::T: {                                                       \
      using block_type = dtl::column_block<dtl::map<dtl::rtt::T>::type, N>;   \
      const block_type& b = static_cast<const block_type&>(col);              \
      detail::select(b, pred, con, mask);                                     \
      break;                                                                  \
    }
    DTL_GENERATE(u8)
    DTL_GENERATE(i8)
    DTL_GENERATE(u16)
    DTL_GENERATE(i16)
    DTL_GENERATE(u32)
    DTL_GENERATE(i32)
    DTL_GENERATE(u64)
    DTL_GENERATE(i64)
    DTL_GENERATE(str)
#undef DTL_GENERATE
  }
};


} // namespace dtl