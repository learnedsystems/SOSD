#pragma once

#include <type_traits>

#include "./FST/include/fst.hpp"
#include "base.h"

template <class KeyType, int size_scale>
class FST : public Competitor {
 public:
  // assume that keys are unique and sorted in ascending order
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    data_size_ = data.size();
    min_key_ = data[0].key;

    // transform integer keys to strings
    std::vector<std::string> keys;
    keys.reserve(data.size());

    // we'll construct a `values` array, but it seems to be ignored by FST
    // (we always just get the index).
    std::vector<uint64_t> values;
    for (const KeyValue<KeyType>& kv : data) {
      if (size_scale > 1 && kv.value % size_scale != 0) continue;

      if (std::is_same<KeyType, std::uint64_t>::value) {
        uint64_t endian_swapped_word = __builtin_bswap64(kv.key);
        keys.emplace_back(std::string(
            reinterpret_cast<const char*>(&endian_swapped_word), 8));
      } else {
        uint32_t endian_swapped_word = __builtin_bswap32(kv.key);
        keys.emplace_back(std::string(
            reinterpret_cast<const char*>(&endian_swapped_word), 4));
      }

      max_key_ = kv.key;
      max_val_ = kv.value;
      values.push_back(kv.value);
    }

    // build fast succinct trie
    return util::timing(
        [&] { fst_ = std::make_unique<fst::FST>(keys, values); });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    std::string key;
    if (std::is_same<KeyType, std::uint64_t>::value) {
      uint64_t endian_swapped_word = __builtin_bswap64(lookup_key);
      key = std::string(reinterpret_cast<const char*>(&endian_swapped_word), 8);
    } else {
      uint32_t endian_swapped_word = __builtin_bswap32(lookup_key);
      key = std::string(reinterpret_cast<const char*>(&endian_swapped_word), 4);
    }

    uint64_t guess = 0;
    if (lookup_key >= max_key_) {
      // looking up a value greater than the largest value causes a segfault...
      return (SearchBound){max_val_, data_size_};
      std::cout << max_val_ << "!!!" << std::endl;
    }

    // faster codepath on size_scale == 1 (static if evaluated at compile time)
    if constexpr (size_scale == 1) {
      fst_->lookupKey(key, guess);
      return {guess, guess};
    } else {
      auto iter = fst_->moveToKeyGreaterThan(key, true);

      // sometimes we get back a bad iterator even though
      // we shouldn't...
      if (!iter.isValid()) return (SearchBound){0, data_size_};

      // multiply by size_scale here because getValue() returns an index
      guess = iter.getValue() * size_scale;
    }

    // expanding by error in both directions is faster than
    // recasting the string key and checking what side we
    // are on.
    const uint64_t error = size_scale;
    const uint64_t start = (guess < error ? 0 : guess - error);
    if (start > data_size_) {
      // happens on fb_200M_uint64 with 256 size_scale
      return (SearchBound){0, data_size_};
    }
    const uint64_t stop =
        (guess + error >= data_size_ ? data_size_ : guess + error + 1);

    return (SearchBound){start, stop};
  }

  std::string name() const { return "FST"; }

  std::size_t size() const {
    // return used memory in bytes
    return fst_->getMemoryUsage();
  }

  bool applicable(bool unique, const std::string& data_filename) {
    // FST only supports unique keys.
    return unique;
  }

  int variant() const { return size_scale; }

 private:
  std::unique_ptr<fst::FST> fst_;
  uint64_t data_size_;
  KeyType min_key_;
  KeyType max_key_;
  uint64_t max_val_;
};
