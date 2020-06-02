#pragma once

#ifndef _DTL_BITSET_INCLUDED
#error "Never use <dtl/bitset_util.hpp> directly; include <dtl/bitset.hpp> instead."
#endif

namespace dtl {

/// converts a byte sequence of length LEN/8 into a bitset of length LEN
template<u64 LEN, typename T>
dtl::bitset<LEN>
to_bitset(const T* ptr) {
  static_assert(LEN > 0, "Template parameter LEN must be greater than 0.");
  static_assert(LEN % 8 == 0, "Template parameter LEN must be a multiple of 8.");
  dtl::bitset<LEN> bits;
  u8* bytes = reinterpret_cast<u8*>(ptr); // TODO use larger words
  u64 byte_cnt = LEN / 8;
  for ($u64 i = 0; i < byte_cnt; i++) {
    $u32 bitmask = bytes[i];
    for ($u64 m = dtl::bits::pop_count(bitmask); m > 0; m--) {
      u64 bit_pos = dtl::bits::tz_count(bitmask);
      bits.set(i * 8 + bit_pos);
//      bitmask = _blsr_u32(bitmask);
      bitmask = dtl::bits::blsr_u32(bitmask);
    }
  }
  return bits;
};


/// iterates over the positions of "on" bits in the given bitset
template<u64 N>
class on_bits_iterator: public std::iterator<
    std::input_iterator_tag,   // iterator_category
    $u64,                      // value_type
    $u64,                      // difference_type
    u64*,                      // pointer
    $u64> {                    // reference

  const std::bitset<N>& bits;
  $u64 bit_pos;
public:
  explicit
  on_bits_iterator(const std::bitset<N>& bits)
      : bits(bits), bit_pos(bits._Find_first()) {}

  on_bits_iterator(const std::bitset<N>& bits, $u64 bit_pos)
      : bits(bits), bit_pos(bit_pos) {}

  inline on_bits_iterator&
  operator++() {
    bit_pos = bits._Find_next(bit_pos);
    return *this;
  }

  inline on_bits_iterator
  operator++(int) {
    on_bits_iterator ret_val = *this;
    ++(*this);
    return ret_val;
  }

  inline bool
  operator==(on_bits_iterator other) const {
    return bit_pos == other.bit_pos;
  }

  inline bool
  operator!=(on_bits_iterator other) const {
    return !(*this == other);
  }

  reference operator*() const {
    return bit_pos;
  }

};


/// iterates over the positions of "on" bits in the given bitset
template<u64 N>
on_bits_iterator<N>
on_bits_begin(const std::bitset<N>& bits) {
  return on_bits_iterator<N>(bits);
}

template<u64 N>
on_bits_iterator<N>
on_bits_end(const std::bitset<N>& bits) {
  return on_bits_iterator<N>(bits, bits.size());
}

template<u64 N>
typename dtl::bitset<N>::on_bits_iterator
on_bits_begin(const dtl::bitset<N>& bits) {
  return bits.on_bits_begin();
}

template<u64 N>
typename dtl::bitset<N>::on_bits_iterator
on_bits_end(const dtl::bitset<N>& bits) {
  return bits.on_bits_end();
}


} // namespace dtl