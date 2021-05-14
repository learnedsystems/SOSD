#ifndef SOSDB_PGM_H
#define SOSDB_PGM_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../util.h"
#include "base.h"
#include "pgm_index.hpp"
//#include <functional>
//#include <boost/iterator/transform_iterator.hpp>
//#include <boost/range/adaptors.hpp>

template <class KeyType, int pgm_error>
class PGM : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    const auto extract_key = [](KeyValue<KeyType> kv) { return kv.key; };

    // This code uses a boost transform iterator to avoid a copy. It
    // seems to be much slower, however.
    /*
    auto it_begin = boost::make_transform_iterator(data.begin(), extract_key);
    auto it_end = boost::make_transform_iterator(data.end(), extract_key);
    pgm_ = PGMIndex<KeyType, pgm_error, 4>(it_begin, it_end);
    */

    // don't count the data copy time against the PGM build time
    std::vector<KeyType> keys;
    keys.reserve(data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(keys),
                   extract_key);

    uint64_t build_time =
        util::timing([&] { pgm_ = decltype(pgm_)(keys.begin(), keys.end()); });

    return build_time;
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    auto approx_range = pgm_.find_approximate_position(lookup_key);
    auto lo = approx_range.lo;
    auto hi = approx_range.hi;

    return (SearchBound){lo, hi + 1};
  }

  std::string name() const { return "PGM"; }

  std::size_t size() const { return pgm_.size_in_bytes(); }

  bool applicable(bool unique, const std::string& data_filename) const {
    return true;
  }

  int variant() const { return pgm_error; }

 private:
  PGMIndex<KeyType, pgm_error, 4> pgm_;
};

#endif  // SOSDB_PGM_H
