#pragma once

#include "base.h"

template <class KeyType>
class OracleSearch : public Competitor {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    // don't even copy the data, we're always going to return 0.
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const { return 0; }

  std::string name() const { return "Oracle"; }

  std::size_t size() const { return 0; }
};
