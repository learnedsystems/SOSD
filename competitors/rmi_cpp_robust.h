#pragma once

#include <memory>
#include <utility>
#include <fstream>

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
      auto rmi_ptr_one = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0.0001);
      auto rmi_ptr_two = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0.0005);
      auto rmi_ptr_three = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0.001);
      auto rmi_ptr_four = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0.005);
      auto rmi_ptr_five = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0);
      rmi_ = std::move(rmi_ptr_one);
      if (rmi_ptr_two->max_error() < rmi_->max_error()) {
        rmi_ = std::move(rmi_ptr_two);
      }
      if (rmi_ptr_three->max_error() < rmi_->max_error()) {
        rmi_ = std::move(rmi_ptr_three);
      }
      if (rmi_ptr_four->max_error() < rmi_->max_error()) {
        rmi_ = std::move(rmi_ptr_four);
      }
      if (rmi_ptr_five->max_error() < rmi_->max_error()) {
        rmi_ = std::move(rmi_ptr_five);
      }
    });
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, const std::string dataset_name) {
    std::vector<KeyType> loading_data;
    loading_data.reserve(data.size());
    for (auto& itm : data) {
      loading_data.push_back(itm.key);
    }

    return util::timing([&] {
      auto new_rmi_ptr = std::make_unique<RMIType<KeyType, Layer1, Layer2>>(
          loading_data, layer2_size, 0.0001);
      rmi_ = std::move(new_rmi_ptr);
      PrintSegmentInformation(loading_data, dataset_name);
    });
  }

  std::string name() const { return "RMICppRobust"; }

  std::size_t size() const { return rmi_->size_in_bytes(); }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto it = rmi_->search(lookup_key);
    return (SearchBound){it.lo, it.hi};
  }

  int variant() const { return variant_num; }

  void PrintSegmentInformation(const std::vector<KeyType>& data, std::string dataset_name) {
    const std::string filename =
        "./results/" + dataset_name + "_bin_info.csv";

    std::ofstream fout(filename, std::ofstream::out | std::ofstream::app);

    if (!fout.is_open()) {
      std::cerr << "Failure to print CSV on " << filename << std::endl;
      return;
    }

    fout << dataset_name;
    for (auto &bin_size : rmi_->segments_per_bin(data, 10)) {
      fout << "," << bin_size;
    }
    fout << std::endl;
  }

 private:
  std::unique_ptr<RMIType<KeyType, Layer1, Layer2>> rmi_;
};
