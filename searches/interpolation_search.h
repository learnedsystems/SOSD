#pragma once
#include "searches/search.h"

template <typename KeyType>
class InterpolationSearch : public Search<KeyType> {
 public:
  uint64_t search(const std::vector<Row<KeyType>>& data,
                  const KeyType lookup_key, size_t* num_qualifying,
                  size_t start, size_t end) const {
    *num_qualifying = 0;

    auto start_it = data.begin() + start;
    auto end_it = data.begin() + end;

    while (start_it < end_it) {
      if (std::distance(start_it, end_it) < 64) break;

      KeyType start_key = start_it->key;
      KeyType end_key = end_it->key;
      double rel_position =
          (double)(lookup_key - start_key) / (double)(end_key - start_key);
      size_t mid_offset =
          (size_t)(rel_position * (double)std::distance(start_it, end_it));

      auto mid = start_it + mid_offset;
      if (mid == start_it) break;

      KeyType mid_key = mid->key;

      if (lookup_key < mid_key) {
        end_it = mid;
      } else if (lookup_key > mid_key) {
        start_it = mid;
      } else {
        start_it = mid;
        end_it = mid + 1;

        // scroll back to the first occurrence
        while (start_it != data.begin() && (start_it - 1)->key == lookup_key)
          start_it--;

        break;
      }
    }

    return bbs.search(data, lookup_key, num_qualifying,
                      std::distance(data.begin(), start_it),
                      std::distance(data.begin(), end_it));
  }

  std::string name() const { return "InterpolationSearch"; }

 private:
  BranchingBinarySearch<KeyType> bbs;
};
