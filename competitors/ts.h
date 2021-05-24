#pragma once

#include "../util.h"
#include "base.h"
#include "ts/builder.h"
#include "ts/ts.h"

template <class KeyType, int size_scale>
class TS : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    if (!parameters_set_) util::fail("TS parameters not set.");

    return util::timing([&] {
      auto min = std::numeric_limits<KeyType>::min();
      auto max = std::numeric_limits<KeyType>::max();
      if (data.size() > 0) {
        min = data.front().key;
        max = data.back().key;
      }
      ts::Builder<KeyType> tsb(min, max, config_.spline_max_error,
                               config_.num_bins, config_.tree_max_error,
                               /*single_pass=*/false, /*use_cache=*/false);
      for (const auto& key_and_value : data) tsb.AddKey(key_and_value.key);
      ts_ = tsb.Finalize();
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    const ts::SearchBound sb = ts_.GetSearchBound(lookup_key);
    return {sb.begin, sb.end};
  }

  std::string name() const { return "TS"; }

  std::size_t size() const { return ts_.GetSize(); }

  bool applicable(bool _unique, const std::string& data_filename) {
    // Remove prefix from filename.
    static constexpr const char* prefix = "data/";
    std::string dataset = data_filename.data();
    dataset.erase(dataset.begin(),
                  dataset.begin() + dataset.find(prefix) + strlen(prefix));

    // Set parameters based on the dataset.
    return SetParameters(dataset);
  }

  int variant() const { return size_scale; }

 private:
  struct TSConfig {
    size_t spline_max_error;
    size_t num_bins;
    size_t tree_max_error;
  };

  bool SetParameters(const std::string& dataset) {
    assert(size_scale >= 1 && size_scale <= 10);
    std::vector<TSConfig> configs;

    if (dataset == "books_200M_uint64") {
      configs = {{512, 128, 2},  {256, 256, 2},  {128, 256, 16}, {64, 1024, 4},
                 {32, 1024, 16}, {16, 1024, 16}, {16, 1024, 8},  {4, 256, 8},
                 {2, 512, 8},    {2, 1024, 8}};
    } else if (dataset == "fb_200M_uint64") {
      configs = {{1024, 1024, 16}, {1024, 1024, 16}, {1024, 512, 8},
                 {256, 512, 8},    {128, 512, 8},    {16, 128, 16},
                 {16, 1024, 16},   {8, 1024, 16},    {4, 256, 16},
                 {2, 256, 16}};
    } else if (dataset == "osm_cellids_200M_uint64") {
      configs = {{1024, 32, 16}, {1024, 32, 16}, {512, 32, 16}, {128, 128, 16},
                 {64, 128, 16},  {16, 64, 16},   {8, 32, 16},   {8, 256, 16},
                 {2, 256, 16},   {2, 512, 16}};
    } else if (dataset == "wiki_ts_200M_uint64") {
      configs = {{1024, 128, 4}, {128, 128, 8}, {64, 256, 8}, {32, 1024, 8},
                 {16, 1024, 8},  {16, 1024, 4}, {4, 128, 16}, {8, 128, 2},
                 {2, 512, 8},    {2, 128, 2}};
    } else {
      // No config.
      return false;
    }

    config_ = configs[size_scale - 1];
    parameters_set_ = true;
    return true;
  }

  ts::TrieSpline<KeyType> ts_;
  TSConfig config_;
  bool parameters_set_ = false;
};
