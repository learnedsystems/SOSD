#pragma once

#ifndef _DTL_STORAGE_INCLUDED
#error "Never use <dtl/storage/column.hpp> directly; include <dtl/storage.hpp> instead."
#endif

#include <bitset>
#include <memory>

#include <dtl/dtl.hpp>
#include <dtl/storage/column_block.hpp>
#include <dtl/storage/types.hpp>


namespace dtl {


/// A naive column implementation which does not support NULL values and deletions
template<typename T>
struct column {

  static u64 block_size_bits = 16;
  static u64 block_size = 1ull << block_size_bits;

  using block = column_block<T, block_size>;

  /// references to all blocks of this column
  std::vector<std::unique_ptr<block>> blocks;

  /// the very last block (where new data is to be inserted)
  block* tail_block;

  /// c'tor
  column() {
    allocate_block();
  }

  inline void
  allocate_block() {
    blocks.push_back(std::make_unique<block>());
    tail_block = blocks[blocks.size() - 1].get();
  }

  inline void
  push_back(const T& val) noexcept {
    // painful branch + indirection !!!
    if (tail_block->size() == block_size) {
      allocate_block();
    }
    tail_block->push_back(val);
  }

  inline void
  push_back(T&& val) noexcept {
    // painful branch + indirection !!!
    if (tail_block->size() == block_size) {
      allocate_block();
    }
    tail_block->push_back(std::move(val));
  }

  inline T&
  operator[](u64 n) noexcept {
    block& b = *blocks[n >> block_size_bits].get();
    return b[n & ((1ull << block_size_bits) - 1)];
  }

  inline const T&
  operator[](u64 n) const noexcept {
    const block& b = *blocks[n >> block_size_bits].get();
    return b[n & ((1ull << block_size_bits) - 1)];
  }


  /// returns the total number of elements (linear runtime complexity; rarely used function)
  inline u64
  size() const noexcept {
    $u64 s = 0;
    for (auto& block_ptr : blocks) {
      s += block_ptr->size();
    }
    return s;
  }

  void
  print(std::ostream& os) const noexcept {
    std::cout << "[";
    for (auto& block_ptr : blocks) {
      print(block_ptr.get());
    }
    std::cout << "]";
  }

};


} // namespace dtl