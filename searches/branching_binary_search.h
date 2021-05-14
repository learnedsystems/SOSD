#pragma once
#include "search.h"

template <typename KeyType>
class BranchingBinarySearch : public Search<KeyType> {
 public:
  uint64_t search(const std::vector<Row<KeyType>>& data,
                  const KeyType lookup_key, size_t* num_qualifying,
                  size_t start, size_t end) const {
    // Returns the sum over all values with the given lookup key.
    // Caution: data has to be sorted.

    *num_qualifying = 0;
    // Search for first occurrence of key.
    auto it =
        std::lower_bound(data.begin() + start, data.begin() + end, lookup_key,
                         [](const Row<KeyType>& lhs, const KeyType lookup_key) {
                           return lhs.key < lookup_key;
                         });
    if (it == data.end() || it->key != lookup_key) {
      std::cerr << "key " << lookup_key << " not found between " << start
                << " and " << end << "\n";

      auto corr = std::lower_bound(
          data.begin(), data.end(), lookup_key,
          [](const Row<KeyType>& lhs, const KeyType lookup_key) {
            return lhs.key < lookup_key;
          });
      std::cerr << "correct index: " << std::distance(data.begin(), corr)
                << "\n";

      return 0;
    }

    // Sum over all values with that key.
    uint64_t result = it->data[0];
    ++(*num_qualifying);

    while (++it != data.end() && it->key == lookup_key) {
      result += it->data[0];
      ++(*num_qualifying);
    }
    return result;
  }

  std::string name() const { return "BinarySearch"; }
};
