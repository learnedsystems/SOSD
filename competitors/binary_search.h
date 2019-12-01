#pragma once

#include "base.h"

template<class KeyType>
class BinarySearch : public Competitor {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_ = data;

    // Nothing else to do here as input data is already sorted.
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    size_t num_qualifying;
    return util::binary_search(data_, lookup_key, &num_qualifying);
  }

  std::string name() const {
    return "BinarySearch";
  }

  std::size_t size() const {
    return sizeof(*this) + data_.size()*sizeof(KeyValue<KeyType>);
  }

 private:
  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;
};
