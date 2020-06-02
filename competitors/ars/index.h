#pragma once

#include <cassert>
#include "../../util.h"

#define Coord std::pair<KeyType, uint32_t> 

// Index: spline and radix index construction.

namespace ars {  
  // The mode of ARS
  static constexpr double precision = std::numeric_limits<double>::epsilon();
  enum class ArsMode : bool {Fusion, RadixOnly};
  enum class Orientation : unsigned {Collinear, Clockwise, CounterClockwise};

  // 'epsilon' specifies the maximum error bound of the approximation and, thus, also the width of the corridor.
  // Through 'lowerLimit' and 'upperLimit' we define the margins of the corridor.
  // If the current point, 'iter', is not inside it, then we add the previous point from the function.
  // Otherwise, we use the error margins of 'iter' ((iter_x, iter_y + eps) and (iter_x, iter_y - eps)) to narrow the corridor for the next point.
  template <class KeyType>
  static void buildSpline(const std::vector<Coord>& cdf, std::vector<Coord>& spline, uint32_t& splineSize, double epsilon, bool shouldSaveSpline = false) {
    // Init the variables
    spline.clear();
    splineSize = 0;
    
    // Empty cdf?
    if (cdf.empty()) return;
    
    // Generalized function to add a new knot to spline
    Coord lastAddedKnot;
    auto addNewKnot = [&](auto knot) {
      lastAddedKnot = knot;
      if (shouldSaveSpline)
        spline.push_back(knot);
      ++splineSize;
    };
   
    // Generalized computation for orientation of 3 points
    auto computeOrientation = [](double dx1, double dy1, double dx2, double dy2) -> Orientation {
      double expr = dy1 * dx2 - dy2 * dx1;
      if (expr > precision)
        return Orientation::Clockwise;
      else if (expr < -precision)
        return Orientation::CounterClockwise;
      return Orientation::Collinear;
    };

    // Add the first point and check if we are done
    addNewKnot(cdf.front());
    if (cdf.size() == 1) return;

    // Compress the 'cdf'
    auto iter = cdf.begin(), limit = cdf.end();
    // Precompute 'upperLimit' and 'lowerLimit' for the second point
    ++iter;
    Coord upperLimit = std::make_pair(iter->first, iter->second + epsilon);
    Coord lowerLimit = std::make_pair(iter->first, (epsilon > iter->second) ? 0 : (iter->second - epsilon));
    ++iter;
    
    // And compute
    for (Coord last; iter != limit; ++iter) {
      // Note that for the lower margin we exclude negative values (due to unsigned type).
      // However, this does not affect the algorithm itself.
      // Why? The only parameter affected by this change is 'lowerLimit'.
      // For its part, it is only used in 'computeOrientation', where 0 on the y-axis has the same impact as a negative value.
      double yPlus = iter->second + epsilon, yMinus = (epsilon > iter->second) ? 0 : (iter->second - epsilon);

      // Precomputation of divided differences
      // Invariant provided: lastAddedKnot_x < [upper, lower]Limit_x < iter_x
      double upperDx = upperLimit.first - lastAddedKnot.first,
             lowerDx = lowerLimit.first - lastAddedKnot.first,
             dx = iter->first - lastAddedKnot.first;
      // For the y-axis there are no invariants, so we perform careful operations with the unsigned type
      double inversed = -static_cast<double>(lastAddedKnot.second),
             upperDy = inversed + upperLimit.second,
             lowerDy = inversed + lowerLimit.second,
             dy = inversed + iter->second;

      // Do we cut the corridor?
      if ((last.first != lastAddedKnot.first) && ((computeOrientation(upperDx, upperDy, dx, dy) == Orientation::CounterClockwise) || (computeOrientation(lowerDx, lowerDy, dx, dy) == Orientation::Clockwise))) {
        // Add the previous point of the function
        addNewKnot(last);

        // And update the bounds of the corridor
        upperLimit = std::make_pair(iter->first, yPlus);
        lowerLimit = std::make_pair(iter->first, yMinus);
      } else {
        // No? Then update the bounds of the corridor. And since 'lastAddedKnot' did not change in this pass, we can reuse the previous computations.
        // Remark regarding the value (inversed + y[Plus, Minus]): the computation is done in a similiar manner as for 'dy',
        // where the y-coordinate of the last added knot is inversed, in order to respect the unsigned type.
        if (computeOrientation(upperDx, upperDy, dx, inversed + yPlus) == Orientation::Clockwise)
          upperLimit = std::make_pair(iter->first, yPlus);
        if (computeOrientation(lowerDx, lowerDy, dx, inversed + yMinus) == Orientation::CounterClockwise)
          lowerLimit = std::make_pair(iter->first, yMinus);
      }
      // And remember the current point
      last = *iter;
    }
    // Add the last point
    addNewKnot(cdf.back());
  }

  // Compute the number of bits to shift with when KeyType == uint64_t
  [[maybe_unused]] static uint32_t computeShiftBits(const uint32_t rho, const uint64_t val) {
    const uint32_t clzl = __builtin_clzl(val);
    if ((64 - clzl) < rho)
      return 0;
    return 64 - rho - clzl;
  }

  // Compute the number of bits to shift with when KeyType == uint32_t
  [[maybe_unused]] static uint32_t computeShiftBits(const uint32_t rho, const uint32_t val) {
    const uint32_t clz = __builtin_clz(val);
    if ((32 - clz) < rho)
      return 0;
    return 32 - rho - clz;
  }

  // Compute the prefix of 'val', after being shifted by 'shiftWith'
  template <class KeyType>
  static KeyType computePrefix(const KeyType min, const uint32_t shiftWith, const KeyType val) {
    return (val - min) >> shiftWith;
  }

  // Build the radix hints for the current spline 
  template <class KeyType>
  static void buildRadix(const ArsMode mode, const uint32_t rho, const std::vector<KeyValue<KeyType>>& data, const std::vector<Coord>& spline, std::vector<uint32_t>& radixHint, uint32_t& shiftWith) {
    // Alloc the memory for the hints
    assert(rho);
    radixHint.resize((1ull << rho) + 1);

    // Compute the number of bits to shift with
    uint32_t n;
    KeyType min, max;
    if (mode == ArsMode::Fusion) {
      n = spline.size();
      min = spline.front().first;
      max = spline.back().first;
    } else {
      n = data.size();
      min = data.front().key;
      max = data.back().key;
    }
    shiftWith = computeShiftBits(rho, max - min);

    // Compute the hints
    radixHint[0] = 0;
    KeyType prevPrefix = 0;
    for (uint32_t i = 0; i != n; ++i) {
      KeyType currPrefix = computePrefix(min, shiftWith, (mode == ArsMode::Fusion) ? spline[i].first : data[i].key);
      if (currPrefix != prevPrefix) {
        for (KeyType j = prevPrefix + 1; j <= currPrefix; ++j)
          radixHint[j] = i;
        prevPrefix = currPrefix;
      }
    }

    // Margin hint values
    for (; prevPrefix != (1ull << rho); ++prevPrefix)
      radixHint[prevPrefix + 1] = n;
  }

  // Predict position of 'key'
  template <class KeyType>
  static double predict(const KeyType key, const std::vector<Coord>& spline, const std::vector<uint32_t>& radixHint, const KeyType min, const uint32_t shiftBits) {
    // First find the segment onto which 'x' lies
    const KeyType p = (key - min) >> shiftBits;
    uint32_t index, begin = radixHint[p], end = radixHint[p + 1];

    switch (end - begin) {
      case 0:
        index = end;
        break;
      case 1: 
        index = (spline[begin].first >= key) ? begin : end;
        break;
      case 2:
        index = ((spline[begin].first >= key) ? begin : ((spline[begin + 1].first >= key) ? (begin + 1) : end));
        break;
      case 3:
        index = ((spline[begin].first >= key) ? begin : ((spline[begin + 1].first >= key) ? (begin + 1) : ((spline[begin + 2].first > key) ? (begin + 2) : end)));
        break;
      default:
        index = std::lower_bound(spline.begin() + begin,
                                 spline.begin() + end,
                                 key,
                                 [](const Coord& a,
                                    const KeyType lookup_key) {
                                   return a.first < lookup_key;
                                 }) - spline.begin();
        break;
    }

    // And then interpolate
    Coord down = spline[index - 1], up = spline[index];
    double slope = static_cast<double>(up.second - down.second) / (up.first - down.first);
    
    // Careful computation of prediction, since the intercept ('down.second') is of unsinged type
    // That's why we first carry out the multiplication
    return slope * (key - down.first) + down.second;
  }

  // The lookup of a raw radix index
  template <class KeyType>
  static SearchBound radixLookup(const KeyType key, const std::vector<uint32_t>& radixHint, const KeyType min, const uint32_t shiftBits) {
    const KeyType p = (key - min) >> shiftBits;
    return (SearchBound){radixHint[p], radixHint[p + 1]};
  }

  // Create the CDF of function 'data'
  // Assumes the function is already sorted after key
  template<class KeyType>
  static std::vector<Coord> buildCdf(const std::vector<KeyValue<KeyType>>& data) {
    std::vector<Coord> cdf;
    
    // Determine for each distinct key, where its first occurrence appeared
    uint32_t pos = 0, refresh = 0;
    KeyType last = data.front().key;
    for (auto d : data) {
      if (d.key != last) {
        cdf.push_back({last, refresh});
        refresh = pos;
        last = d.key;
      }
      pos++;
    }
    // And add the last point
    cdf.push_back({last, refresh});
    return cdf;
  }
}

