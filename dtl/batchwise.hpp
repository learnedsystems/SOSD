#pragma once

#include <dtl/dtl.hpp>

namespace dtl {

//===----------------------------------------------------------------------===//
// The number of items processed in one go.
const size_t BATCH_SIZE = 1024;

template<std::size_t batch_size = BATCH_SIZE ,typename it, typename binary_fn>
void batch_wise(it begin, it end, binary_fn fn) {
  it i = begin;
  while (i + batch_size < end) {
    fn(i, i + batch_size);
    i += batch_size;
  }
  if (i < end) {
    fn(i, end);
  }
}
//===----------------------------------------------------------------------===//

} // namespace dtl

