#pragma once

#include "base.h"
#include "fast64.h"

template <int size_scale>
class FAST64 : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<uint64_t>>& data) {
    // create key and value vectors
    std::vector<uint64_t> keys;
    std::vector<uint64_t> values;

    for (auto& itm : data) {
      uint64_t idx = itm.value;
      if (size_scale > 1 && idx % size_scale != 0) continue;

      keys.push_back(itm.key);
      values.push_back(itm.value);
    }

    data_size_ = data.size();
    return util::timing([&] {
      tree_ = create_fast64(&keys[0], keys.size(), &values[0], values.size());
    });
  }

  SearchBound EqualityLookup(const uint64_t lookup_key) const {
    uint64_t v1, v2;
    lookup_fast64(tree_, lookup_key, &v1, &v2);
    v2 = std::min(data_size_, v2);
    return (SearchBound){v1, v2};
  }

  std::string name() const { return "FAST"; }

  int variant() const { return size_scale; }

  std::size_t size() const { return size_fast64(tree_); }

  ~FAST64() {
    if (tree_) destroy_fast64(tree_);
  }

  bool applicable(bool unique, const std::string& _data_filename) const {
    return unique;
  }

 private:
  uint64_t data_size_;
  Fast64* tree_ = NULL;
};

#pragma once

#include "base.h"
#include "fast64.h"

template <int size_scale>
class FAST32 : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<uint32_t>>& data) {
    // create key and value vectors
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;

    for (auto& itm : data) {
      uint64_t idx = itm.value;
      if (size_scale > 1 && idx % size_scale != 0) continue;

      keys.push_back(itm.key);
      values.push_back(itm.value);
    }

    data_size_ = data.size();
    return util::timing([&] {
      tree_ = create_fast32(&keys[0], keys.size(), &values[0], values.size());
    });
  }

  SearchBound EqualityLookup(const uint32_t lookup_key) const {
    uint32_t v1, v2;
    lookup_fast32(tree_, lookup_key, &v1, &v2);
    v2 = (uint32_t)std::min(data_size_, (uint64_t)v2);
    return (SearchBound){v1, v2};
  }

  std::string name() const { return "FAST"; }

  int variant() const { return size_scale; }

  std::size_t size() const { return size_fast32(tree_); }

  ~FAST32() {
    if (tree_) destroy_fast32(tree_);
  }

 private:
  uint64_t data_size_;
  Fast32* tree_ = NULL;
};
