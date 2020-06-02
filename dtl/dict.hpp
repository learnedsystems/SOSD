#pragma once

#include <algorithm>
#include <set>
#include <vector>

namespace dtl {

/// a naively implemented string dictionary
template<typename T>
class dict {
public:
  std::vector<T> data;
  size_t size;

  dict(const std::vector<T>& values) noexcept {
    // sort values and eliminate duplicates
    const std::set<T> s(values.cbegin(), values.cend());
    size = s.size();
    for (auto it = s.cbegin(); it != s.cend(); it++) {
      data.push_back(*it);
    }
  }

  uint64_t lookup(const T& value) const noexcept {
    auto lower = std::lower_bound(data.begin(), data.end(), value);
    return lower - data.cbegin();
  }

  std::vector<uint64_t> map(const std::vector<T>& values) const noexcept {
    std::vector<uint64_t> ints;
    for (auto it = values.cbegin(); it != values.cend(); it++) {
      ints.push_back(lookup(*it));
    }
    return ints;
  }
  
};

} // namespace dtl
