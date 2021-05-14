#pragma once

#include <math.h>

#include "base.h"
#include "rmi/all_rmis.h"

//#define DEBUG_RMI

// RMI with binary search
template <class KeyType, int rmi_variant, uint64_t build_time, size_t rmi_size,
          const char* namespc, uint64_t (*RMI_FUNC)(uint64_t, size_t*),
          bool (*RMI_LOAD)(char const*), void (*RMI_CLEANUP)()>
class RMI_B {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    data_size_ = data.size();
    ;

    const std::string rmi_path =
        (std::getenv("SOSD_RMI_PATH") == NULL ? "rmi_data"
                                              : std::getenv("SOSD_RMI_PATH"));
    if (!RMI_LOAD(rmi_path.c_str())) {
      util::fail(
          "Could not load RMI data from rmi_data/ -- either an allocation "
          "failed or the file could not be read.");
    }

    return build_time;
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    size_t error;
    uint64_t guess = RMI_FUNC(lookup_key, &error);

    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = (guess + error >= data_size_ ? data_size_ : guess + error);

    return (SearchBound){start, stop};
  }

  std::string name() const { return "RMI"; }

  std::size_t size() const { return rmi_size; }

  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  int variant() const { return rmi_variant; }

  ~RMI_B() { RMI_CLEANUP(); }

 private:
  uint64_t data_size_;
};
