#pragma once

#include "base.h"

// Text book interpolation search.
template <class KeyType>
class InterpolationSearch : public Competitor {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_ = data;
    unique_ = util::is_unique(data);

    // Nothing else to do here as input data is already sorted.
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    const int pos = InterpSearch(lookup_key);
    if (pos == -1) util::fail("InterpolationSearch: key not found");
    // If the data is unique, we can immediately return.
    if (unique_) return data_[pos].value;
    // Otherwise, we need to compute a sum over their TIDs.
    return util::linear_search(data_, lookup_key, pos);
  }

  std::string name() const { return "InterpolationSearch"; }

  std::size_t size() const {
    return sizeof(*this) + data_.size() * sizeof(KeyValue<KeyType>);
  }

  bool applicable(bool unique, const std::string& data_filename) const {
    // Applicable to all datasets except lognormal (due to poor performance).
    return data_filename.find("lognormal") == std::string::npos;
  }

 private:
  int InterpSearch(const KeyType lookup_key) const {
    int start = 0;
    int end = data_.size() - 1;
    int idx_range, estimate;
    KeyType distance, val_range;
    double fraction;

    // Interpolation search.
    while (end - start > 1000 && lookup_key >= data_[start].key &&
           lookup_key <= data_[end].key) {
      distance = lookup_key - data_[start].key;
      val_range = data_[end].key - data_[start].key;
      fraction = static_cast<double>(distance) / val_range;
      idx_range = end - start;
      estimate = start + (fraction * idx_range);
      if (data_[estimate].key == lookup_key) {
        return estimate;
      }
      if (data_[estimate].key < lookup_key) {
        start = estimate + 1;
      } else {
        end = estimate - 1;
      }
    }

    // Find first occurrence of key using linear search.
    for (int i = start; i <= end; ++i) {
      if (data_[i].key == lookup_key) {
        return i;
      }
    }

    return -1;
  }

  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;
  bool unique_;
};
