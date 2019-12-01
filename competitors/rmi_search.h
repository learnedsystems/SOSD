#pragma once

#include "base.h"
#include "rmi/all_rmis.h"

#include <math.h>

//#define DEBUG_RMI

// RMI with linear search
template<class KeyType, uint64_t build_time,
    size_t rmi_size, const char* namespc, uint64_t (* RMI_FUNC)(uint64_t)>
class RMI_L {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_ = data;
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    uint64_t guess = RMI_FUNC(lookup_key);
    uint64_t result = util::linear_search(data_, lookup_key, guess);

#ifdef DEBUG_RMI
    std::cout << "key: " << lookup_key
              << "\tguess: " << guess
              << "\tresult: " << result
              << "\terror: " << (guess > result ? guess - result : result - guess)
              << "\n";
#endif
    return result;
  }

  std::string name() const {
    std::string str(namespc);
    return str.substr(str.length() - 3);
  }

  std::size_t size() const {
    return data_.size()*sizeof(KeyValue<KeyType>) + rmi_size;
  }

  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  uint64_t additional_build_time() const { return build_time; }

 private:
  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;
};

// RMI with binary search
template<class KeyType, uint64_t build_time, size_t rmi_size,
    const char* namespc,
    uint64_t (* RMI_FUNC)(uint64_t, size_t*)>
class RMI_B {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_ = data;
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    size_t error, num_qualifying;
    uint64_t guess = RMI_FUNC(lookup_key, &error);

    int64_t start = ((int64_t) guess - (int64_t) error) - 1;
    start = (start < 0 ? 0 : start);
    int64_t stop = guess + error + 1;
    stop = ((uint64_t) stop > data_.size() ? data_.size() : stop);

#ifdef RMI_DEBUG
    std::cout << "searching for key " << lookup_key << " from "
              << start << " to " << stop 
              << " (" << (stop - start) << " values)\n";

    std::cout << "    " << data_[start].key << " -- "
              << data_[stop].key << "\n";
#endif

    return util::binary_search(data_, lookup_key, &num_qualifying,
                               start, stop);
  }

  std::string name() const {
    std::string str(namespc);
    return str.substr(str.length() - 3);
  }

  std::size_t size() const {
    return data_.size()*sizeof(KeyValue<KeyType>) + rmi_size;
  }

  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  uint64_t additional_build_time() const { return build_time; }

 private:
  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;
};
