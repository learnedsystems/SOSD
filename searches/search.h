#pragma once
#include <cstdint>
#include <vector>

#include "../util.h"

template <typename KeyType>
class Search {
 public:
  // TODO change interface to [start, end) (excluding `end`) to match
  // std::lower_bound and update all callers & search implementations
  uint64_t search(const std::vector<Row<KeyType>>&, const KeyType, size_t*,
                  size_t, size_t) const;

  std::string name() const;
};
