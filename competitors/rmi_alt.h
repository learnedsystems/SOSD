#pragma once

#include <utility>

#include "./analysis-rmi/include/rmi/rmi.hpp"
#include "./analysis-rmi/include/rmi/models.hpp"
#include "base.h"

// Alternate implementation of RMI as described by Maltry and Dittrich
template <class KeyType, typename Layer1, typename Layer2, template <KeyType, Layer1, Layer2> typename RMIType, size_t layer2_size, int variant>
class RMIAlternate : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<std::pair<KeyType, uint64_t>> loading_data;
    loading_data.reserve(data.size());
    for (auto& itm : data) {
      uint64_t idx = itm.value;
      loading_data.push_back(std::make_pair(itm.key, itm.value));
    }

    data_size_ = data.size();
    return util::timing(
        [&] {
          rmi_ = RMIType<KeyType, Layer1, Layer2>(loading_data, layer2_size);
        });
  }

  std::string name() const { return "RMIAlternate"; }

  std::size_t size() const { return rmi_.size_in_bytes(); }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto it = rmi_.search(lookup_key);
    return (SearchBound){it.lo, it.hi};
  }

  int variant() const { return variant; }

 private:
  uint64_t data_size_ = 0;
  RMIType<KeyType, Layer1, Layer2> rmi_;
};