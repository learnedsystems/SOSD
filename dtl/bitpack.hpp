#pragma once

#include <dtl/dtl.hpp>

#include <vector>

namespace dtl {

namespace {

using word_t = $u64;
const word_t word_bitlength = sizeof(word_t) * 8;
u64 header_size = 2; // words

} // anonymous namespace


/// truncates the given input values to k bits and stores them consecutively in the output vector
template<typename T>
std::vector<word_t>
bitpack_horizontal(u32 k, const std::vector<T> in) {
  assert(k > 0 && k < word_bitlength);
  assert(k < (sizeof(T) * 8));

  // allocate memory for the output
  u64 data_size = ((in.size() * k) / word_bitlength) + 1;
  std::vector<word_t> packed;
  packed.resize(header_size + data_size, 0);

  // write the number of codes
  packed[0] = in.size();
  // write the code size
  packed[1] = k;

  const word_t k_mask = k == word_bitlength ? ~word_t(0) : (word_t(1) << k) - 1;

  // write the codes starting at position 2
  word_t* writer = &packed[header_size];
  $u8 writer_bitpos_in_word = 0;

  for (T value : in) {
    // extract the k least significant bits
    const word_t v = value & k_mask;

    *writer |= v << writer_bitpos_in_word;
    writer_bitpos_in_word += k;

    if (writer_bitpos_in_word >= word_bitlength) {
      // word wrap
      writer++;
      writer_bitpos_in_word -= word_bitlength;
      // write remaining bits if any
      *writer |= v >> (k - writer_bitpos_in_word);
    }
  }
  return packed;
}


/// unpacks the given bit-packed input vector
template<typename T>
std::vector<T>
bitunpack_horizontal(const std::vector<word_t> packed) {
  // allocate memory for the output
  std::vector<T> out;
  u64 output_size = packed[0];
  out.resize(output_size, 0);

  u64 k = packed[1];
  assert(k > 0 && k < word_bitlength);
  const word_t k_mask = k == word_bitlength ? ~word_t(0) : (word_t(1) << k) - 1;

  // read the codes word by word starting at position 2
  T* writer = &out[0];
  const T* writer_end = &out[output_size];
  const word_t* reader = &packed[header_size];
  const word_t* reader_end = &packed[header_size + output_size];
  $u8 reader_bitpos_in_word = 0;

  while (reader != reader_end) {
    // extract next code
    T value = (*reader >> reader_bitpos_in_word) & k_mask;
    reader_bitpos_in_word += k;
    if (reader_bitpos_in_word >= word_bitlength) {
      // word wrap
      reader++;
      reader_bitpos_in_word -= word_bitlength;
      // read remaining bits if any
      value |= *reader << (k - reader_bitpos_in_word);
      value &= k_mask;
    }
    if (std::is_signed<T>::value) {
      // sign extend
      const T sign = ((T(1) << (k - 1)) & value) == 0 ? T(0) : ~T(0) << k;
      value |= sign;
    }
    *writer = value;
    writer++;
    if (writer == writer_end) break;
  }
  return out;
}


} // namespace dtl
