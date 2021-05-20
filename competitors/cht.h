#pragma once

#include "../util.h"
#include "base.h"
#include "cht/builder.h"
#include "cht/cht.h"

template <class KeyType, int size_scale>
class CHT : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    if (!parameters_set_) util::fail("CHT parameters not set.");

    return util::timing([&] {
      auto min = std::numeric_limits<KeyType>::min();
      auto max = std::numeric_limits<KeyType>::max();
      if (data.size() > 0) {
        min = data.front().key;
        max = data.back().key;
      }
      cht::Builder<KeyType> chtb(min, max, num_bins_, max_error_,
                                 /*single_pass=*/false, /*use_cache=*/false);
      for (const auto& key_and_value : data) chtb.AddKey(key_and_value.key);
      cht_ = chtb.Finalize();
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    const cht::SearchBound sb = cht_.GetSearchBound(lookup_key);
    return {sb.begin, sb.end};
  }

  std::string name() const { return "CHT"; }

  std::size_t size() const { return cht_.GetSize(); }

  bool applicable(bool _unique, const std::string& data_filename) {
    // Remove prefix from filename.
    static constexpr const char* prefix = "data/";
    std::string dataset = data_filename.data();
    dataset.erase(dataset.begin(),
                  dataset.begin() + dataset.find(prefix) + strlen(prefix));

    if (dataset != "books_200M_uint64" && dataset != "fb_200M_uint64" &&
        dataset != "osm_cellids_200M_uint64") {
      return false;
    }

    // Set parameters based on the dataset.
    SetParameters(dataset);

    return true;
  }

  int variant() const { return size_scale; }

 private:
  // Returns <num_bins, max_error>.
  std::pair<size_t, size_t> GetConfig(const std::string& dataset) {
    assert(size_scale >= 1 && size_scale <= 10);

    using Configs = const std::vector<std::pair<size_t, size_t>>;

    if (dataset == "books_200M_uint64") {
      Configs configs = {{128, 512}, {128, 512}, {128, 512}, {128, 512},
                         {16, 512},  {256, 128}, {64, 64},   {512, 32},
                         {128, 16},  {1024, 16}};
      return configs[10 - size_scale];
    }

    if (dataset == "fb_200M_uint64") {
      Configs configs = {{64, 1024},  {64, 1024},  {64, 1024}, {64, 1024},
                         {256, 1024}, {1024, 512}, {64, 128},  {512, 128},
                         {256, 64},   {256, 32}};
      return configs[10 - size_scale];
    }

    if (dataset == "osm_cellids_200M_uint64") {
      Configs configs = {{32, 1024}, {32, 1024}, {32, 1024}, {32, 1024},
                         {32, 512},  {64, 256},  {64, 128},  {64, 32},
                         {64, 16},   {1024, 8}};
      return configs[10 - size_scale];
    }

    std::cerr << "No tuning config for this dataset" << std::endl;
    throw;
  }

  void SetParameters(const std::string& dataset) {
    const std::pair<size_t, size_t> config = GetConfig(dataset);
    num_bins_ = config.first;
    max_error_ = config.second;
    parameters_set_ = true;
  }

  cht::CompactHistTree<KeyType> cht_;
  size_t num_bins_ = 64;
  size_t max_error_;
  bool parameters_set_ = false;
};
