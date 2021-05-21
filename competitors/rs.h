#pragma once

#include "../util.h"
#include "base.h"
#include "rs/builder.h"
#include "rs/radix_spline.h"

template <class KeyType, int size_scale>
class RS : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    if (!parameters_set_) util::fail("RS parameters not set.");

    return util::timing([&] {
      auto min = std::numeric_limits<KeyType>::min();
      auto max = std::numeric_limits<KeyType>::max();
      if (data.size() > 0) {
        min = data.front().key;
        max = data.back().key;
      }
      rs::Builder<KeyType> rsb(min, max, num_radix_bits_, max_error_);
      for (const auto& key_and_value : data) rsb.AddKey(key_and_value.key);
      rs_ = rsb.Finalize();
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    const rs::SearchBound sb = rs_.GetSearchBound(lookup_key);
    return {sb.begin, sb.end};
  }

  std::string name() const { return "RS"; }

  std::size_t size() const { return rs_.GetSize(); }

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

    using Config = std::pair<size_t, size_t>;
    std::vector<Config> configs;

    if (dataset == "normal_200M_uint32") {
      configs = {{10, 6}, {15, 1}, {16, 1}, {18, 1}, {20, 1},
                 {21, 1}, {24, 1}, {25, 1}, {26, 1}, {26, 1}};
    } else if (dataset == "normal_200M_uint64") {
      configs = {{14, 2}, {16, 1}, {16, 1}, {20, 1}, {22, 1},
                 {24, 1}, {26, 1}, {26, 1}, {28, 1}, {28, 1}};
    } else if (dataset == "lognormal_200M_uint32") {
      configs = {{12, 20}, {16, 3}, {16, 2}, {18, 1}, {20, 1},
                 {22, 1},  {24, 1}, {24, 1}, {26, 1}, {28, 1}};
    } else if (dataset == "lognormal_200M_uint64") {
      configs = {{12, 3}, {18, 1}, {18, 1}, {20, 1}, {22, 1},
                 {24, 1}, {26, 1}, {26, 1}, {28, 1}, {28, 1}};
    } else if (dataset == "uniform_dense_200M_uint32") {
      configs = {{4, 2},  {16, 2}, {18, 1}, {20, 1}, {20, 1},
                 {22, 2}, {24, 1}, {26, 3}, {26, 3}, {28, 2}};
    } else if (dataset == "uniform_dense_200M_uint64") {
      configs = {{4, 2},  {16, 1}, {16, 1}, {20, 1}, {22, 1},
                 {24, 1}, {24, 1}, {26, 1}, {28, 1}, {28, 1}};
    } else if (dataset == "uniform_sparse_200M_uint32") {
      configs = {{12, 220}, {14, 100}, {14, 80}, {16, 30}, {18, 20},
                 {20, 10},  {20, 8},   {20, 5},  {24, 3},  {26, 1}};
    } else if (dataset == "uniform_sparse_200M_uint64") {
      configs = {{12, 150}, {14, 70}, {16, 50}, {18, 20}, {20, 20},
                 {20, 9},   {20, 5},  {24, 3},  {26, 2},  {28, 1}};
    } else if (dataset == "books_200M_uint32") {
      configs = {{14, 250}, {14, 250}, {16, 190}, {18, 80}, {18, 50},
                 {22, 20},  {22, 9},   {22, 8},   {24, 3},  {28, 2}};
    } else if (dataset == "books_200M_uint64") {
      configs = {{12, 380}, {16, 170}, {16, 110}, {20, 50}, {20, 30},
                 {22, 20},  {22, 10},  {24, 3},   {26, 3},  {28, 2}};
    } else if (dataset == "books_400M_uint64") {
      configs = {{16, 220}, {16, 220}, {18, 160}, {20, 60}, {20, 40},
                 {22, 20},  {22, 7},   {26, 3},   {28, 2},  {28, 1}};
    } else if (dataset == "books_600M_uint64") {
      configs = {{18, 330}, {18, 330}, {18, 190}, {20, 70}, {22, 50},
                 {22, 20},  {24, 7},   {26, 3},   {28, 2},  {28, 1}};
    } else if (dataset == "books_800M_uint64") {
      configs = {{18, 320}, {18, 320}, {18, 200}, {22, 80}, {22, 60},
                 {22, 20},  {24, 9},   {26, 3},   {28, 3},  {28, 3}};
    } else if (dataset == "fb_200M_uint64") {
      configs = {{8, 140}, {8, 140}, {8, 140}, {8, 140}, {10, 90},
                 {22, 90}, {24, 70}, {26, 80}, {26, 7},  {28, 80}};
    } else if (dataset == "osm_cellids_200M_uint64") {
      configs = {{20, 160}, {20, 160}, {20, 160}, {20, 160}, {20, 80},
                 {24, 40},  {24, 20},  {26, 8},   {26, 3},   {28, 2}};
    } else if (dataset == "osm_cellids_400M_uint64") {
      configs = {{20, 190}, {20, 190}, {20, 190}, {20, 190}, {22, 80},
                 {24, 20},  {26, 20},  {26, 10},  {28, 6},   {28, 2}};
    } else if (dataset == "osm_cellids_600M_uint64") {
      configs = {{20, 190}, {20, 190}, {20, 190}, {22, 180}, {22, 100},
                 {24, 20},  {26, 20},  {28, 7},   {28, 5},   {28, 2}};
    } else if (dataset == "osm_cellids_800M_uint64") {
      configs = {{22, 190}, {22, 190}, {22, 190}, {22, 190}, {24, 190},
                 {26, 30},  {26, 20},  {28, 7},   {28, 5},   {28, 1}};
    } else if (dataset == "wiki_ts_200M_uint64") {
      configs = {{14, 100}, {14, 100}, {16, 60}, {18, 20}, {20, 20},
                 {20, 9},   {20, 5},   {22, 3},  {26, 2},  {26, 1}};
    } else {
      // No config.
      return false;
    }

    const Config config = configs[size_scale - 1];
    num_radix_bits_ = config.first;
    max_error_ = config.second;
    parameters_set_ = true;
    return true;
  }

  rs::RadixSpline<KeyType> rs_;
  size_t num_radix_bits_;
  size_t max_error_;
  bool parameters_set_ = false;
};
