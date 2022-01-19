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
      ts::Builder<KeyType> tsb(min, max, spline_max_error_);
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
  bool SetParameters(const std::string& dataset) {
    assert(size_scale >= 1 && size_scale <= 10);
    std::vector<size_t> configs;

    if (dataset == "books_200M_uint64") {
      configs = {500, 200, 150, 60, 50, 25, 25, 4, 2, 1};
    } else if (dataset == "fb_200M_uint64") {
      configs = {225, 225, 225, 225, 100, 32, 16, 8, 8, 2};
    } else if (dataset == "osm_cellids_200M_uint64") {
      configs = {150, 150, 150, 150, 80, 25, 8, 8, 4, 1};
    } else if (dataset == "wiki_ts_200M_uint64") {
      configs = {175, 175, 90, 32, 25, 16, 16, 4, 2, 1};
    } else {
      // No config.
      return false;
    }

    spline_max_error_ = configs[size_scale - 1];
    parameters_set_ = true;
    return true;
  }

  ts::TrieSpline<KeyType> ts_;
  size_t spline_max_error_;
  bool parameters_set_ = false;
};
