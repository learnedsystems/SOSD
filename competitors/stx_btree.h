#pragma once

#include "base.h"
#include "../utils/tracking_allocator.h"

#include <stx/btree_multimap.h>

template<class KeyType>
class STXBTree : public Competitor {
 public:
  STXBTree() : btree_(TrackingAllocator<std::pair<KeyType, uint64_t>>(
      total_allocation_size)) {
  }

  void Build(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<std::pair<KeyType, uint64_t>> reformatted_data;
    reformatted_data.reserve(data.size());
    for (auto iter : data) {
      reformatted_data.emplace_back(KeyType(iter.key), uint64_t(iter.value));
    }
    btree_.bulk_load(reformatted_data.begin(), reformatted_data.end());
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    // Search for first occurrence of key.
    auto it = btree_.lower_bound(lookup_key);
    if (it==btree_.end() || it->first!=lookup_key)
      util::fail("STXBTree: key not found");
    // Sum over all values with that key.
    uint64_t result = it->second;
    while (++it!=btree_.end() && it->first==lookup_key) {
      result += it->second;
    }
    return result;
  }

  std::string name() const {
    return "stx::btree_multimap";
  }

  std::size_t size() const {
    return btree_.get_allocator().total_allocation_size + sizeof(*this);
  }

 private:
  // Using a multimap here since keys may contain duplicates.
  uint64_t total_allocation_size = 0;
  stx::btree_multimap<KeyType,
                      uint64_t,
                      std::less<KeyType>,
                      stx::btree_default_map_traits<KeyType, uint64_t>,
                      TrackingAllocator<std::pair<KeyType, uint64_t> >> btree_;
};
