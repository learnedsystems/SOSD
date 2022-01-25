#pragma once

#include <utility>
#include <memory>

#include "./analysis-rmi/include/rmi/rmi.hpp"
#include "./analysis-rmi/include/rmi/models.hpp"
#include "base.h"

// Alternate implementation of RMI as described by Maltry and Dittrich
template <class KeyType, typename Layer1, typename Layer2, template <typename...> typename RMIType, size_t layer2_size, uint32_t variant_num>
class RMIAlternate : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<KeyType> loading_data;
    loading_data.reserve(data.size());
    for (auto& itm : data) {
      loading_data.push_back(itm.key);
    }

    return util::timing(
        [&] {
          auto new_rmi_ptr = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(loading_data, layer2_size);
          rmi_ = std::move(new_rmi_ptr);
        });
  }

  std::string name() const { return "RMIAlternate"; }

  std::size_t size() const {
    return rmi_->size_in_bytes();
//    RMIType<KeyType, Layer1, Layer2>& ref = const_cast<RMIType<KeyType, Layer1, Layer2>&>(rmi_);
//    return ref.size_in_bytes();
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto it = rmi_->search(lookup_key);
    return (SearchBound){it.lo, it.hi};
  }

  int variant() const { return variant_num; }

 private:
  std::unique_ptr<RMIType<KeyType, Layer1, Layer2>> rmi_;
};