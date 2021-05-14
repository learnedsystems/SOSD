#pragma once
#include "search.h"

template <typename KeyType>
class BranchlessBinarySearch : public Search<KeyType> {
 public:
  uint64_t search(const std::vector<Row<KeyType>>& data,
                  const KeyType lookup_key, size_t* num_qualifying,
                  size_t start, size_t end) const {
    // Returns the sum over all values with the given lookup key.
    // Caution: data has to be sorted.

    *num_qualifying = 0;

    // Search for first occurrence of key.
    int n = end - start + 1;  // `end` is inclusive.
    int lower = start;

    // Function adapted from
    // https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
    // Authored by Giorgio Vinciguerra.
    while (const int half = n / 2) {
      const int middle = lower + half;
      // Prefetch next two middles.
      __builtin_prefetch(&(data[lower + half / 2]), 0, 0);
      __builtin_prefetch(&(data[middle + half / 2]), 0, 0);
      lower = (data[middle].key <= lookup_key) ? middle : lower;
      n -= half;
    }
    // Scroll back to the first occurrence.
    while (lower > 0 && data[lower - 1].key == lookup_key) --lower;

    if (data[lower].key != lookup_key) {
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
    uint64_t result = 0;
    for (unsigned int i = lower; data[i].key == lookup_key && i < data.size();
         ++i) {
      result += data[i].data[0];
      ++(*num_qualifying);
    }
    return result;
  }

  std::string name() const { return "BranchlessBinarySearch"; }
};
