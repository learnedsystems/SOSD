#pragma once

#include <tsl/robin_map.h>

#include "base.h"

template <class KeyType>
class RobinHash : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    return util::timing([&] {
      // target a load factor of 0.75
      map_.reserve((4 * data.size()) / 3);
      for (auto& itm : data) {
        map_.insert({itm.key, itm.value});
      }
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto search = map_.find(lookup_key);
    if (search == map_.end())
      util::fail("Could not find lookup key in hashmap!");

    return (SearchBound){search.value(), search.value() + 1};
  }

  std::string name() const { return "RobinHash"; }

  std::size_t size() const {
    return map_.bucket_count() * (sizeof(KeyType) + sizeof(uint64_t));
  }

 private:
  tsl::robin_map<KeyType, uint64_t> map_;
};
