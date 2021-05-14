#pragma once

// Implementation of TIP by Ryan Marcus, 2019
// Based on "Efficiently Searching In-Memory Sorted Arrays:Revenge of the
// Interpolation Search?" by Peter Van Sandt, Yannis Chronis, Jignesh M. Patel

template <class KeyType>
class RMThreePointInterpolationSearch : public Competitor {
 public:
  constexpr static int64_t GUARD = 64;
  constexpr static int LOOP_UNROLL = 8;

  // This method should follow Eqn 8 from the paper -- used to determine
  // the first "expected" position. We use the implementation from the UW
  // code here, prefixing constants with "interp_r". See `build` for where
  // these are computed.
  inline int64_t interpolate_r(KeyType x) const {
    double dx = (double)x;
    return interp_r_d +
           (int64_t)(interp_r_d_a * ((double)interp_r_y_1 - dx) /
                     (interp_r_diff_scale - dx * (interp_r_a_0 + 1.0)));
  }

  inline KeyValue<KeyType> AtPadded(int64_t index) const {
    assert(index >= -LOOP_UNROLL);
    assert(index <= (int)data_.size());
    return data_[index + LOOP_UNROLL];
  }

  inline size_t SizePadded() const {
    return (int)data_.size() - LOOP_UNROLL * 2;
  }

  // Eqn 2 from the paper -- used to interpolate the next "expected"
  // position in the loop
  inline int64_t interpolate_jn(KeyType target, int64_t left, int64_t mid,
                                int64_t right) const {
    // These introduce additional casts to support subtraction of unsigned
    // numbers
    double y0 = (double)AtPadded(left).key - (double)target;
    double y1 = (double)AtPadded(mid).key - (double)target;
    // This does not do the additional cast because right >= target
    double y2 = AtPadded(right).key - target;

    int64_t x0 = left;
    int64_t x1 = mid;
    int64_t x2 = right;

    double numerator = y1 * static_cast<double>(x1 - x2) *
                       static_cast<double>(x1 - x0) * (y2 - y0);
    double denom = y2 * static_cast<double>(x1 - x2) * (y0 - y1) +
                   (y0 * static_cast<double>(x1 - x0) * (y1 - y2));
    return x1 + static_cast<int64_t>(numerator / denom);
  }

  // Build the index structure -- just copy the data over.
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_.reserve(data.size() + LOOP_UNROLL * 2);
    // we need to add LOOP_UNROLL sentinel values to either side of the array
    for (int i = 0; i < LOOP_UNROLL; i++)
      data_.push_back(KeyValue<KeyType>{0, 0});
    data_.insert(data_.end(), data.begin(), data.end());
    for (int i = 0; i < LOOP_UNROLL; i++)
      data_.push_back(
          KeyValue<KeyType>{std::numeric_limits<KeyType>::max(), 0});

    // From here on, data_ shouldn't be used directly because of the padding

    // these values and their computation lifted from the UW
    // ThreePointInterpolation constructor initializers
    double last_key = (double)AtPadded(SizePadded() - 1).key;
    interp_r_d = SizePadded() / 2;
    interp_r_y_1 = AtPadded(interp_r_d).key;
    double diff_y_01 = AtPadded(0).key - (double)interp_r_y_1;
    interp_r_a_0 = (diff_y_01 == ((double)interp_r_y_1 - last_key)
                        ? 0.99999999999999
                        : diff_y_01 / (interp_r_y_1 - last_key));
    interp_r_diff_scale = ((double)AtPadded(0).key - interp_r_a_0 * last_key);
    interp_r_d_a = (1.0 + interp_r_a_0) * interp_r_d;
  }

  __attribute__((optimize("unroll-loops"))) uint64_t unrolled_linear_search(
      const KeyType lookup_key, int64_t estimate) const {
    while (true) {
      for (int unroll_i = 0; unroll_i < LOOP_UNROLL; unroll_i++) {
        KeyType candidate = AtPadded(estimate + unroll_i).key;
        if (candidate >= lookup_key) return estimate + unroll_i;
      }
      estimate += LOOP_UNROLL;
    }
  }

  __attribute__((optimize("unroll-loops"))) uint64_t unrolled_linear_search_r(
      const KeyType lookup_key, int64_t estimate) const {
    while (true) {
      for (int unroll_i = 0; unroll_i < LOOP_UNROLL; unroll_i++) {
        KeyType candidate = AtPadded(estimate - unroll_i).key;
        if (candidate <= lookup_key) return estimate - unroll_i;
      }
      estimate -= LOOP_UNROLL;
    }
  }

  KeyType sum_reverse_unguarded(const KeyType lookup_key,
                                int64_t expected) const {
    uint64_t sum = 0;
    for (; AtPadded(expected).key == lookup_key; expected--) {
      sum += AtPadded(expected).value;
    }
    assert(AtPadded(expected).key < lookup_key);
    return sum;
  }

  KeyType sum_forward_unguarded(const KeyType lookup_key,
                                int64_t expected) const {
    uint64_t sum = 0;
    for (; AtPadded(expected).key == lookup_key; expected++) {
      sum += AtPadded(expected).value;
    }
    assert(AtPadded(expected).key > lookup_key);
    return sum;
  }

  __attribute__((noinline)) uint64_t sum_forward(const KeyType lookup_key,
                                                 int64_t expected) const {
    // Must not be in the middle of a set of matches
    assert(AtPadded(expected).key != lookup_key ||
           AtPadded(expected - 1).key != lookup_key);
    auto match = unrolled_linear_search(lookup_key, expected);
    return sum_forward_unguarded(lookup_key, match);
  }

  __attribute__((noinline)) uint64_t sum_reverse(const KeyType lookup_key,
                                                 int64_t expected) const {
    assert(AtPadded(expected).key != lookup_key ||
           AtPadded(expected + 1).key != lookup_key);
    auto match = unrolled_linear_search_r(lookup_key, expected);
    return sum_reverse_unguarded(lookup_key, match);
  }

  __attribute__((noinline)) uint64_t guard(const KeyType lookup_key,
                                           int64_t expected) const {
    if (AtPadded(expected).key == lookup_key) {
      return sum_reverse_unguarded(lookup_key, expected) +
             sum_forward_unguarded(lookup_key, expected + 1);
    } else if (AtPadded(expected).key < lookup_key) {
      return sum_forward(lookup_key, expected + 1);
    }
    return sum_reverse(lookup_key, expected - 1);
  }

  // following Algorithm 3 of the paper
  // Returns the sum of the values of the matches
  __attribute__((flatten)) __attribute__((always_inline)) uint64_t
  EqualityLookup(const KeyType lookup_key) const {
    int64_t left = 0;
    int64_t mid = SizePadded() / 2;
    int64_t right = SizePadded() - 1;

    int64_t expected = interpolate_r(lookup_key);
    while (true) {
      if (expected - mid <= GUARD && expected - mid >= -GUARD) {
        return guard(lookup_key, expected);
      }

      if (AtPadded(mid).key != AtPadded(expected).key) {
        if (mid < expected) {
          left = mid;
        } else {
          right = mid;
        }
        if (expected + GUARD >= right) {
          return sum_reverse(lookup_key, right);
        } else if (expected - GUARD <= left) {
          return sum_forward(lookup_key, left);
        }
      }
      mid = expected;
      expected = interpolate_jn(lookup_key, left, mid, right);
    }
  }

  std::string name() const { return "TIP"; }

  // size of the index structure -- we count the sentinal values and keys
  std::size_t size() const {
    return sizeof(*this) + data_.size() * sizeof(KeyValue<KeyType>);
  }

 protected:
  std::vector<KeyValue<KeyType>> data_;

  // these constants are used in the UW implementation of interpolate_r.
  // I'm simply trusting that their math works out. See `build` method
  // for details.
  int64_t interp_r_d = 0.0;
  double interp_r_d_a = 0.0;
  KeyType interp_r_y_1 = 0.0;
  double interp_r_diff_scale = 0.0;
  double interp_r_a_0 = 0.0;
};
