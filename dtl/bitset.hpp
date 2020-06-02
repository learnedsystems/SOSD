#pragma once

#include <array>
#include <cstring>
#include <sstream>
#include <string>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

namespace dtl {


template<u64 N>
struct bitset {

  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two."); // TODO remove restriction
  static_assert(N > 0, "Template parameter 'N' must be greater than zero.");

  static constexpr u64 length = N;

  using word_t = $u64;
  static constexpr u64 word_bitlength = sizeof(word_t) * 8;
  static constexpr u64 word_bitlength_log = ct::log_2_u64<word_bitlength>::value;
  static constexpr u64 word_mask = word_bitlength - 1;
  static constexpr u64 word_cnt = (N + (word_bitlength - 1)) / word_bitlength;

  // the data
  alignas(64) std::array<word_t, word_cnt> words;

  // Proxy for a single bit of the bitset (allows modifications)
  struct reference {

    word_t* word;
    u64 in_word_bit_pos;
    const word_t mask;

    reference(bitset& bits, u64 bit_pos)
        : word(bits.get_word(bit_pos)),
          in_word_bit_pos(bits.in_word_index(bit_pos)),
          mask(word_t(1) << in_word_bit_pos) {};

    inline u1
    is_on() const noexcept {
      return (*word & mask) != word_t(0);
    }

    inline
    operator bool() const noexcept{
      return is_on();
    };

    inline bool
    operator~() const noexcept {
      return !is_on();
    };

    inline reference&
    flip() noexcept {
      if (is_on()) {
        *word ^= mask;
      }
      else {
        *word |= mask;
      }
      return *this;
    };

    inline reference&
    operator=(u1 x) noexcept {
      if (is_on() != x) {
        flip();
      }
      return *this;
    };

    inline reference&
    operator=(const reference& other) noexcept {
      if (is_on() != other.is_on()) {
        flip();
      }
      return *this;
    };
  };

  // --- c'tors ---

  bitset() {
    reset();
  }

  enum init_t {
    zero, one, uninitialized
  };

  bitset(init_t init) {
    switch (init) {
      case zero:
        reset();
        break;
      case one:
        std::memset(&words[0], ~0, word_cnt * sizeof(word_t));
        break;
      case uninitialized:
        break;
    }
  }

  bitset(word_t val) {
    reset();
    words[0] = val;
  }

  bitset(const bitset& other, const u1 flip = false) {
    if (flip) {
      for ($u64 i = 0; i < word_cnt; i++) {
        words[i] = ~other.words[i];
      }
    }
    else {
      std::memcpy(&words[0], &other.words[0], word_cnt * sizeof(word_t));
    }
  }

  // --- helper functions ---

  inline constexpr u64
  word_index(u64 bit_pos) const noexcept {
    return bit_pos >> word_bitlength_log;
  }

  inline constexpr u64
  in_word_index(u64 bit_pos) const noexcept {
    return bit_pos & word_mask;
  }

  inline word_t*
  get_word(u64 idx) noexcept {
    return &words[word_index(idx)];
  }


  // --- basic bit operations ---

  inline constexpr u1
  get(u64 idx) const noexcept {
    const word_t search_pattern = word_t(1) << in_word_index(idx);
    return (words[word_index(idx)] & search_pattern) == search_pattern;
  }

  inline void
  set(u64 idx) noexcept {
    words[word_index(idx)] |= word_t(1) << in_word_index(idx);
  }

  inline void
  reset(u64 idx) noexcept {
    words[word_index(idx)] &= ~(word_t(1) << in_word_index(idx));
  }

  inline void
  reset() noexcept {
    std::memset(&words[0], 0, word_cnt * sizeof(word_t));
  }

  inline void
  flip() noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      words[i] = ~words[i];
    }
  }

  inline void
  flip(u64 idx) noexcept {
    if (get(idx)) reset(idx); else set(idx);
  }

  inline void
  set(u64 idx, u1 value) noexcept {
    if (value) set(idx); else reset(idx);
  }


  // --- subscript operator ---

  inline constexpr u1
  operator[](u64 idx) const noexcept {
    return get(idx);
  }

  inline reference
  operator[](u64 idx) noexcept {
    return reference(*this, idx);
  }


  // --- comparison operations ---

  inline u1
  operator==(const bitset<N>& rhs) const noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != rhs.words[i]) return false;
    }
    return true;
  };

  inline u1
  operator!=(const bitset<N>& rhs) const noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != rhs.words[i]) return true;
    }
    return false;
  };


  inline u1
  all() const noexcept {
    const word_t m = ~word_t(0);
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != m) return false;
    }
    return true;
  };

  inline u1
  any() const noexcept {
    const word_t zero = word_t(0);
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != zero) return true;
    }
    return false;
  };

  inline u1
  none() const noexcept {
    const word_t zero = word_t(0);
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != zero) return false;
    }
    return true;
  };

  inline u64
  count() const noexcept {
    $u64 cnt = 0;
    for ($u64 i = 0; i < word_cnt; i++) {
      cnt += dtl::bits::pop_count(words[i]);
    }
    return cnt;
  };

  inline constexpr u64
  size() const noexcept {
    return N;
  };


  // --- bitwise operations ---

  inline bitset<N>&
  operator&=(const bitset<N>& other) noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      words[i] &= other.words[i];
    }
    return *this;
  };

  inline bitset<N>&
  operator|=(const bitset<N>& other) noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      words[i] |= other.words[i];
    }
    return *this;
  };

  inline bitset<N>&
  operator^=(const bitset<N>& other) noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      words[i] ^= other.words[i];
    }
    return *this;
  };

  inline bitset<N>
  operator~() const noexcept {
    bitset<N> cpy(*this, true);
    return cpy;
  };


  inline word_t
  extract_word(u64 bit_pos_begin) const noexcept {
    u64 bit_idx = in_word_index(bit_pos_begin);
    u64 first_word_idx = word_index(bit_pos_begin);
    u64 second_word_idx = first_word_idx + 1;
    const word_t first_word = words[first_word_idx];
    const word_t second_word = second_word_idx >= word_cnt ? word_t(0) : words[second_word_idx];
    word_t res = first_word >> bit_idx;
    res |= second_word << (word_bitlength - bit_idx);
    return res;
  }


  // --- shift ---

  inline bitset<N>
  operator<<(u64 pos) const noexcept {
    bitset ret_val(uninitialized);
    if ((pos % word_bitlength) == 0) {
      // special case
      u64 o = pos / word_bitlength;
      // copy words
      for ($u64 i = o; i < word_cnt; i++) {
        ret_val.words[i] = words[i - o];
      }
      // zero words at the beginning
      for ($u64 i = 0; i < o; i++) {
        ret_val.words[i] = 0;
      }
    }
    else if (pos >= N) {
      // special case
      ret_val.reset();
    }
    else {
      // common case
      u64 word_offset = pos / word_bitlength;
      u64 bit_offset = pos % word_bitlength;
      // zero words at the beginning
      for ($u64 i = 0; i < word_offset; i++) {
        ret_val.words[i] = 0;
      }
      // do the shift
      for ($u64 i = 0; i < (word_cnt - word_offset); i++) {
        const word_t w = words[i];
        ret_val.words[i + word_offset] |= w << bit_offset;
        ret_val.words[i + word_offset + 1] = w >> (word_bitlength - bit_offset);
      }
      // clear unused bits at the beginning
      ret_val.words[word_offset] &= ~word_t(0) << bit_offset;
      // clear unused bits at the ending
      u64 unused_bit_cnt = (N % word_bitlength) != 0 ? word_bitlength - (N % word_bitlength) : 0;
      if (unused_bit_cnt) {
        ret_val.words[word_cnt - 1] &= ~word_t(0) >> unused_bit_cnt;
      }
    }
    return ret_val;
  };

  inline bitset<N>&
  operator<<=(u64 pos) noexcept {
    if ((pos % word_bitlength) == 0) {
      // special case
      u64 o = pos / word_bitlength;
      // copy words
      for ($u64 i = o; i < word_cnt; i++) {
        words[i] = words[i - o];
      }
      // zero words at the beginning
      for ($u64 i = 0; i < o; i++) {
        words[i] = 0;
      }
    }
    else if (pos >= N) {
      // special case
      reset();
    }
    else {
      // common case
      u64 word_offset = pos / word_bitlength;
      u64 bit_offset = pos % word_bitlength;
      // zero words at the beginning
      for ($u64 i = 0; i < word_offset; i++) {
        words[i] = 0;
      }
      // do the shift
      for ($u64 i = 0; i < (word_cnt - word_offset); i++) {
        const word_t w = words[i];
        words[i + word_offset] |= w << bit_offset;
        words[i + word_offset + 1] = w >> (word_bitlength - bit_offset);
      }
      // clear unused bits at the beginning
      words[word_offset] &= ~word_t(0) << bit_offset;
      // clear unused bits
      u64 unused_bit_cnt = (N % word_bitlength) != 0 ? word_bitlength - (N % word_bitlength) : 0;
      if (unused_bit_cnt) {
        words[word_cnt - 1] &= ~word_t(0) >> unused_bit_cnt;
      }
    }
    return *this;
  };

  inline bitset<N>
  operator>>(u64 pos) const noexcept {
    bitset ret_val(uninitialized);
    if ((pos % word_bitlength) == 0) {
      // special case
      u64 o = pos / word_bitlength;
      // copy words
      for ($u64 i = 0; i < word_cnt - o; i++) {
        ret_val.words[i] = words[i + o];
      }
      // zero words at the ending
      for ($u64 i = word_cnt - o; i < word_cnt; i++) {
        ret_val.words[i] = 0;
      }
    }
    else if (pos >= N) {
      // special case
      ret_val.reset();
    }
    else {
      // common case
      u64 word_offset = pos / word_bitlength;
      u64 bit_offset = pos % word_bitlength;
      // do the shift
      for ($u64 i = 0; i < (word_cnt - word_offset); i++) {
        word_t w = words[i + word_offset] >> bit_offset;
        w |= words[i + word_offset + 1] << (word_bitlength - bit_offset);
        ret_val.words[i] = w;
      }
      // zero words at the ending
      for ($u64 i = (word_cnt - word_offset); i < word_cnt; i++) {
        ret_val.words[i] = 0;
      }
    }
    return ret_val;

  };

  inline bitset<N>
  operator>>=(u64 pos) noexcept {
    if ((pos % word_bitlength) == 0) {
      // special case
      u64 o = pos / word_bitlength;
      // copy words
      for ($u64 i = 0; i < word_cnt - o; i++) {
        words[i] = words[i + o];
      }
      // zero words at the ending
      for ($u64 i = word_cnt - o; i < word_cnt; i++) {
        words[i] = 0;
      }
    }
    else if (pos >= N) {
      // special case
      reset();
    }
    else {
      // common case
      u64 word_offset = pos / word_bitlength;
      u64 bit_offset = pos % word_bitlength;
      // do the shift
      for ($u64 i = 0; i < (word_cnt - word_offset); i++) {
        word_t w = words[i + word_offset] >> bit_offset;
        w |= words[i + word_offset + 1] << (word_bitlength - bit_offset);
        words[i] = w;
      }
      // zero words at the ending
      for ($u64 i = (word_cnt - word_offset); i < word_cnt; i++) {
        words[i] = 0;
      }
    }
    return *this;

  };


  // --- find "on" bits ---

  inline u64
  find_first() const noexcept {
    u64 word_idx = find_first_non_zero_word();
    u64 bit_idx = dtl::bits::tz_count(words[word_idx]);
    return word_idx * word_bitlength + bit_idx;
  }

  inline u64
  find_first_non_zero_word() const noexcept {
    for ($u64 i = 0; i < word_cnt; i++) {
      if (words[i] != 0) return i;
    }
    return length;
  }

  inline u64
  find_next($u64 pos) const noexcept {
    pos++;
    if (pos == length) return length;
    // check the current word for "on" bits
    $u64 word_idx = word_index(pos);
    const word_t mask = ~word_t(0) << in_word_index(pos);
    word_t word = words[word_idx] & mask;
    if (word != 0) {
      // there are remaining bits in the current word
      return word_idx * word_bitlength + dtl::bits::tz_count(word);
    }
    // find the next non zero word
    word_idx++;
    for (; word_idx < word_cnt; word_idx++) {
      if (words[word_idx] != 0) break;
    }
    if (word_idx == word_cnt) return length;
    return word_idx * word_bitlength + dtl::bits::tz_count(words[word_idx]);
  }


  // iterates over the positions of "on" bits in the given bitset
  class on_bits_iterator: public std::iterator<
      std::input_iterator_tag,   // iterator_category
      $u64,                      // value_type
      $u64,                      // difference_type
      u64*,                      // pointer
      $u64> {                    // reference

    const bitset& bits;
    $u64 bit_pos;

  public:
    explicit
    on_bits_iterator(const bitset& bits)
        : bits(bits), bit_pos(bits.find_first()) {}

    on_bits_iterator(const bitset& bits, u64 bit_pos)
        : bits(bits), bit_pos(bit_pos) {}

    inline on_bits_iterator&
    operator++() {
      bit_pos = bits.find_next(bit_pos);
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
  on_bits_iterator
  on_bits_begin() const {
    return on_bits_iterator(*this);
  }

  on_bits_iterator
  on_bits_end() const {
    return on_bits_iterator(*this, length);
  }


  // ---

  template<
      class CharT = char,
      class Traits = std::char_traits<CharT>,
      class Allocator = std::allocator<CharT>>
  inline std::basic_string<CharT, Traits, Allocator>
  to_string(CharT zero = CharT('0'), CharT one = CharT('1')) const {
    std::basic_string<CharT, Traits> str(length, zero);
    for (auto it = on_bits_begin(); it != on_bits_end(); it++) {
      str[length - 1 - *it] = one;
    }
    return str;
  }

};

} // namespace dtl


template<u64 N>
inline dtl::bitset<N>
operator&(const dtl::bitset<N>& lhs, const dtl::bitset<N>& rhs) noexcept {
  dtl::bitset<N> ret_val(dtl::bitset<N>::uninitialized);
  for ($u64 i = 0; i < dtl::bitset<N>::word_cnt; i++) {
    ret_val.words[i] = lhs.words[i] & rhs.words[i];
  }
  return ret_val;
};

template<u64 N>
inline dtl::bitset<N>
operator|(const dtl::bitset<N>& lhs, const dtl::bitset<N>& rhs) noexcept {
  dtl::bitset<N> ret_val(dtl::bitset<N>::uninitialized);
  for ($u64 i = 0; i < dtl::bitset<N>::word_cnt; i++) {
    ret_val.words[i] = lhs.words[i] | rhs.words[i];
  }
  return ret_val;
};

template<u64 N>
inline dtl::bitset<N>
operator^(const dtl::bitset<N>& lhs, const dtl::bitset<N>& rhs) noexcept {
  dtl::bitset<N> ret_val(dtl::bitset<N>::uninitialized);
  for ($u64 i = 0; i < dtl::bitset<N>::word_cnt; i++) {
    ret_val.words[i] = lhs.words[i] ^ rhs.words[i];
  }
  return ret_val;
};


template<class CharT, class Traits, u64 N>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const dtl::bitset<N>& x) {
  const std::ctype<CharT>& char_type = std::use_facet<std::ctype<CharT>>(os.getloc());
  auto str = x.to_string(char_type.widen('0'), char_type.widen('1'));
  os << str;
  return os;
};

template<class CharT, class Traits, u64 N>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, dtl::bitset<N>& x) {
  // TODO implement
};


// --- include utility functions for bitsets ---

// used to generate compiler errors if util headers are included directly
#ifndef _DTL_BITSET_INCLUDED
#define _DTL_BITSET_INCLUDED
#endif

#include <dtl/bitset_util.hpp>