#pragma once

#include <utility>
#include "../utils/tracking_allocator.h"
#include "base.h"
#include "./ALEX/src/core/alex.h"
#include "./ALEX/src/core/alex_base.h"

template<class KeyType, int size_scale>
class Alex : public Competitor {
public:
  Alex() : map_(TrackingAllocator<std::pair<KeyType, uint64_t>>(
      total_allocation_size)) {
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<std::pair<KeyType, uint64_t>> loading_data;
    loading_data.reserve(data.size());

    for (auto& itm: data) {
      uint64_t idx = itm.value;
      if (size_scale > 1 && idx % size_scale != 0)
        continue;
      loading_data.push_back(std::make_pair(itm.key, itm.value));
    }

    data_size_ = data.size();

    return util::timing([&] {
      map_.bulk_load(loading_data.data(), loading_data.size());
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto it = map_.lower_bound(lookup_key);

    uint64_t guess;
    if (it == map_.cend()) {
      guess = data_size_ - 1;
    } else {
      guess = it.payload();
    }

    const uint64_t error = size_scale - 1;

    const uint64_t start = guess < error ? 0 : guess - error;
    const uint64_t
        stop = guess + 1 > data_size_ ? data_size_ : guess + 1; // stop is exclusive (that's why +1)

    return (SearchBound){ start, stop };
  }

  std::string name() const {
    return "ALEX";
  }

  std::size_t size() const {
    return map_.get_allocator().total_allocation_size + sizeof(*this);
  }

  int variant() const { return size_scale; }

private:
  uint64_t total_allocation_size = 0;
  uint64_t data_size_ = 0;
  alex::Alex<KeyType,
	     uint64_t,
	     alex::AlexCompare,
	     TrackingAllocator<std::pair<KeyType, uint64_t>>,
	     true > map_;
};
