#pragma once

#include <limits>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>


namespace dtl {


//===----------------------------------------------------------------------===//
/// The block addressing modes.
enum class block_addressing : u32 {
  /// The numbers of blocks is a power of two.
  POWER_OF_TWO,
  /// The numbers of blocks is restricted to 'cheap' magic numbers.
  MAGIC,
  /// Chooses either POWER_OF_TWO or MAGIC at runtime.
  DYNAMIC,
};
//===----------------------------------------------------------------------===//


struct block_addressing_logic_base {};

template<
    block_addressing          // the block _addressing mode
>
struct block_addressing_logic {};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Magic addressing
//===----------------------------------------------------------------------===//
template<>
struct block_addressing_logic<block_addressing::MAGIC>
    : block_addressing_logic_base {

  using size_t = $u32;
  using hash_value_t = $u32;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr std::size_t max_block_cnt = std::numeric_limits<hash_value_t>::max();

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt;      // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const size_t block_cnt_mask;
  const dtl::fast_divisor_u32_t fast_divisor;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Determines the actual block count.
  static size_t
  determine_block_cnt(const std::size_t desired_block_cnt) {
    auto actual_block_cnt = dtl::next_cheap_magic(desired_block_cnt).divisor;
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  block_addressing_logic(const std::size_t desired_block_cnt) noexcept
      : block_cnt(determine_block_cnt(desired_block_cnt)),
        block_cnt_log2(dtl::log_2(dtl::next_power_of_two(block_cnt))),
        block_cnt_mask(dtl::next_power_of_two(block_cnt) - 1),
        fast_divisor(dtl::next_cheap_magic(block_cnt)) {
    // sanity check
    if (desired_block_cnt > max_block_cnt
        || block_cnt < desired_block_cnt // overflow
        || dtl::next_power_of_two(block_cnt) > max_block_cnt) {
      throw std::logic_error("The block count must not exceed 2^32.");
    }
  }

  block_addressing_logic(const block_addressing_logic&) noexcept = default;

  block_addressing_logic(block_addressing_logic&&) noexcept = default;

  ~block_addressing_logic() noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the number of blocks.
  __forceinline__ __host__ __device__
  size_t
  get_block_cnt() const noexcept {
    return block_cnt;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto h = hash_value;
    const auto block_idx = dtl::fast_mod_u32(h, fast_divisor);
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto h = hash_value;
    const auto block_idx = dtl::fast_mod_u32(h, fast_divisor);
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return hash_value_bitlength;
  }
  //===----------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Power of two addressing
//===----------------------------------------------------------------------===//
template<>
struct block_addressing_logic<block_addressing::POWER_OF_TWO>
    : block_addressing_logic_base {

  using size_t = $u32;
  using hash_value_t = $u32;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr std::size_t max_block_cnt = std::numeric_limits<hash_value_t>::max();

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt;      // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const size_t block_cnt_mask;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Determines the actual block count.
  static size_t
  determine_block_cnt(const std::size_t desired_block_cnt) {
    auto actual_block_cnt = dtl::next_power_of_two(desired_block_cnt);
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  block_addressing_logic(const std::size_t desired_block_cnt) noexcept
      : block_cnt(determine_block_cnt(desired_block_cnt)),
        block_cnt_log2(dtl::log_2(block_cnt)),
        block_cnt_mask(u64(block_cnt) - 1) {
    // sanity check
    if (desired_block_cnt > max_block_cnt
        || block_cnt < desired_block_cnt // overflow
        || dtl::next_power_of_two(block_cnt) > max_block_cnt) {
      throw std::logic_error("The block count must not exceed 2^32.");
    }
  }

  block_addressing_logic(const block_addressing_logic&) noexcept = default;

  block_addressing_logic(block_addressing_logic&&) noexcept = default;

  ~block_addressing_logic() noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the number of blocks.
  __forceinline__ __host__ __device__
  size_t
  get_block_cnt() const noexcept {
    return block_cnt;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - get_required_addressing_bits()));
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - get_required_addressing_bits()));
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return block_cnt_log2;
  }
  //===----------------------------------------------------------------------===//

};


//===----------------------------------------------------------------------===//
// Dynamic addressing.
//
// Automatically determines the addressing mode (POW2 or MAGIC) at run time
// based on the desired block count.
//
// Alternatively, either POW2 or MAGIC can be enforced.
//===----------------------------------------------------------------------===//
template<>
struct block_addressing_logic<block_addressing::DYNAMIC>
    : block_addressing_logic_base {

  using size_t = $u32;
  using hash_value_t = $u32;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr std::size_t max_block_cnt = std::numeric_limits<hash_value_t>::max();


  //===----------------------------------------------------------------------===//
  /// Dynamically determines the addressing mode.
  /// Could either be POWER_OF_TWO or MAGIC.
  static block_addressing
  determine_addressing_mode(const std::size_t desired_block_cnt) {
    if (dtl::is_power_of_two(desired_block_cnt)) {
      return block_addressing::POWER_OF_TWO;
    }

    auto block_cnt_pow2  = dtl::next_power_of_two(desired_block_cnt);
    auto block_cnt_magic = dtl::next_cheap_magic(desired_block_cnt).divisor;
    if (block_cnt_magic > block_cnt_pow2) {
      // In some (rare) cases, the desired block count is slightly below a
      // power of 2 and the next cheap magic number is slightly higher.
      return block_addressing::POWER_OF_TWO;
    }

    return block_addressing::MAGIC;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  // the block addressing mode
  const block_addressing addr_mode;
  // the actual two possible block addressing instances
  const block_addressing_logic<block_addressing::MAGIC> magic_addr;
  const block_addressing_logic<block_addressing::POWER_OF_TWO> pow2_addr;
  //===----------------------------------------------------------------------===//


  block_addressing_logic(const std::size_t desired_block_cnt,
                         const block_addressing enforce_addr_mode = block_addressing::DYNAMIC) noexcept
      : addr_mode(enforce_addr_mode == block_addressing::DYNAMIC ? determine_addressing_mode(desired_block_cnt) : enforce_addr_mode),
        magic_addr(desired_block_cnt),
        pow2_addr(desired_block_cnt) {
    // sanity check
    if (desired_block_cnt > max_block_cnt) {
      throw std::logic_error("The block count must not exceed 2^32.");
    }
    switch (addr_mode) {
      case block_addressing::POWER_OF_TWO:
        if (pow2_addr.block_cnt < desired_block_cnt
            || dtl::next_power_of_two(pow2_addr.block_cnt) > max_block_cnt) {
          throw std::logic_error("The block count must not exceed 2^32.");
        }
      case block_addressing::MAGIC:
        if (magic_addr.block_cnt < desired_block_cnt
            || dtl::next_power_of_two(magic_addr.block_cnt) > max_block_cnt) {
          throw std::logic_error("The block count must not exceed 2^32.");
        }
      case block_addressing::DYNAMIC: // must not happen
        break;
    }
  }

  block_addressing_logic(const block_addressing_logic&) noexcept = default;

  block_addressing_logic(block_addressing_logic&&) noexcept = default;

  ~block_addressing_logic() noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the number of blocks.
  __forceinline__ __host__ __device__
  size_t
  get_block_cnt() const noexcept {
    switch (addr_mode) {
      case block_addressing::POWER_OF_TWO:
        return pow2_addr.get_block_cnt();
      case block_addressing::MAGIC:
        return magic_addr.get_block_cnt();
      case block_addressing::DYNAMIC: // must not happen
        break;
    }
    __builtin_unreachable();
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    switch (addr_mode) {
      case block_addressing::POWER_OF_TWO:
        return pow2_addr.get_block_idx(hash_value);
      case block_addressing::MAGIC:
        return magic_addr.get_block_idx(hash_value);
      case block_addressing::DYNAMIC: // must not happen
        break;
    }
    __builtin_unreachable();
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    switch (addr_mode) {
      case block_addressing::POWER_OF_TWO:
        return pow2_addr.get_block_idxs(hash_value);
      case block_addressing::MAGIC:
        return magic_addr.get_block_idxs(hash_value);
      case block_addressing::DYNAMIC: // must not happen
        break;
    }
    __builtin_unreachable();
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    switch (addr_mode) {
      case block_addressing::POWER_OF_TWO:
        return pow2_addr.get_required_addressing_bits();
      case block_addressing::MAGIC:
        return magic_addr.get_required_addressing_bits();
      case block_addressing::DYNAMIC: // must not happen
        break;
    }
    __builtin_unreachable();
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


} // namespace dtl
