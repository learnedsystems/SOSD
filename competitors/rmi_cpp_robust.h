#pragma once

#include <fstream>
#include <memory>
#include <utility>

#include "./analysis-rmi/include/rmi/models.hpp"
#include "./analysis-rmi/include/rmi/rmi_robust.hpp"
#include "base.h"

// Alternate implementation of RMI in C++ with robust outlier compensation
template <class KeyType, typename Layer1, typename Layer2,
          template <typename...> typename RMIType, size_t layer2_size,
          uint32_t variant_num>
class RMICppRobust : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<KeyType> loading_data;
    loading_data.reserve(data.size());
    for (auto& itm : data) {
      loading_data.push_back(itm.key);
    }

    return util::timing([&] {
      rmi_ = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(loading_data,
                                                                layer2_size, 0);
      if (auto rmi_ptr_one = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
              loading_data, layer2_size, 0.0001);
          rmi_ptr_one->mean_error() < rmi_->mean_error()) {
        rmi_ = std::move(rmi_ptr_one);
      }
      if (auto rmi_ptr_two = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
              loading_data, layer2_size, 0.0005);
          rmi_ptr_two->mean_error() < rmi_->mean_error()) {
        rmi_ = std::move(rmi_ptr_two);
      }
      if (auto rmi_ptr_three =
              std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
                  loading_data, layer2_size, 0.001);
          rmi_ptr_three->mean_error() < rmi_->mean_error()) {
        rmi_ = std::move(rmi_ptr_three);
      }
      if (auto rmi_ptr_four =
              std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
                  loading_data, layer2_size, 0.005);
          rmi_ptr_four->mean_error() < rmi_->mean_error()) {
        rmi_ = std::move(rmi_ptr_four);
      }
    });
  }

  std::string name() const { return "RMICppRobust"; }

  std::size_t size() const { return rmi_->size_in_bytes(); }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto it = rmi_->search(lookup_key);
    return (SearchBound){it.lo, it.hi};
  }

  int variant() const { return variant_num; }

 private:
  std::unique_ptr<RMIType<KeyType, Layer1, Layer2>> rmi_;
};
