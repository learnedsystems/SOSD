#ifndef SOSDB_ADAPTIVE_RADIX_SPLINE_H
#define SOSDB_ADAPTIVE_RADIX_SPLINE_H

#include "../base.h"
#include "../../util.h"
#include "index.h"
#include "analytics.h"

#define Coord std::pair<KeyType, uint32_t>

// Adaptive Radix Spline (ARS)
// 'size_scale' is the memory limit in MiB

#define NOTIFY
// #define CUSTOM

namespace ars {
template<class KeyType, size_t size_scale>
class AdaptiveRadixSpline : public Competitor {
  public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {

    // Build the CDF of data
    data_ = data;
    dataSize = data.size();
    return util::timing([&] {
      cdf = buildCdf(data);

      // Get the best configuration from analytics or choose a custom one
      std::pair<State, std::pair<double, ArsMode>> configuration =
#ifdef CUSTOM
        // Custom configuration
        std::make_pair(std::make_pair(0, 0), std::make_pair(-1, ArsMode::Fusion));
#else
      // Best configuration from analytics
      analyzeData(data, cdf, size_scale);
#endif
      maxError = configuration.first.first;
      mode = configuration.second.second;

#ifdef NOTIFY
      // Notify about results
      std::cout << "size_scale=" << size_scale << " MiB";
      if (mode == ArsMode::Fusion)
        std::cout << ": Fusion with eps=" << configuration.first.first << " and rho=" << configuration.first.second << std::endl;
      else
        std::cout << ": Radix-only with rho=" << configuration.first.second << std::endl;
#endif

      // Build up the spline only if the fusion is better than a raw radix index
      min = data.front().key;
      if (mode == ArsMode::Fusion) {
        unsigned splineSize;
        buildSpline(cdf, spline, splineSize, maxError, true);
      }
      // We do not need the CDF anymore
      cdf.clear();

      // Depending on "mode", build up the corresponding radix index
      unsigned maxRadixBits = configuration.first.second;
      buildRadix(mode, maxRadixBits, data, spline, radixHint, shiftWith);

#ifdef NOTIFY
      // Finally, notify about the memory status
      if (size_scale * (1u << 20) >= size()) {
        double ratio = static_cast<double>(size_scale * (1u << 20) - size()) / (size_scale * (1u << 20));
        std::cout << "Memory restriction fulfilled! Remained with: " << ((size_scale * (1 << 20)) - size()) << " bytes, with ratio = " << ratio << std::endl;  
      } else {
        double ratio = static_cast<double>(size() - size_scale * (1 << 20)) / (size_scale * (1 << 20));  
        std::cout << "Memory restriction exceeded by: " << (size() - (size_scale * (1u << 20))) << " bytes, with ratio = " << ratio << std::endl;
      }
#endif
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    if (mode == ArsMode::Fusion) {
      uint64_t estimation = predict(lookup_key, spline, radixHint, min, shiftWith);
      uint64_t start = (estimation < maxError) ? 0 : (estimation - maxError);
      uint64_t stop = (estimation + maxError >= dataSize) ? dataSize : (estimation + maxError);
      return (SearchBound){start, stop};
    } else {
      return radixLookup(lookup_key, radixHint, min, shiftWith);
    }
  }

  std::string name() const {
    return "AdaptiveRadixSpline";
  }

  std::size_t size() const {
    return sizeof(*this)
        + spline.size() * sizeof(Coord)
        + radixHint.size() * sizeof(uint32_t);
  }

  bool applicable(__attribute__((unused)) bool unique_keys, const std::string& data_filename) {
    return true;
  }

 private:
  static constexpr double memTolerance = 0.01;
  std::vector<Coord> cdf;
  std::vector<Coord> spline;
  ArsMode mode;

  KeyType min;
  uint32_t shiftWith;
  std::vector<uint32_t> radixHint;

  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;
  uint64_t dataSize, maxError;
};

}

template<class KeyType, size_t size_scale>
using AdaptiveRadixSpline = ars::AdaptiveRadixSpline<KeyType, size_scale>;

#endif // SOSDB_ADAPTIVE_RADIX_SPLINE_H
