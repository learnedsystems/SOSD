#pragma once

#include <cmath>
#include <cassert>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include "index.h"
#include "../../util.h"

// Analyze data and return the best configuration for a given memory budget.

using namespace std::chrono;

#define Coord std::pair<KeyType, uint32_t> 
// #define TRANSLATION_VERBOSE
#define BENCHMARK_VERBOSE

namespace ars {
  static constexpr double tolerance = 0.01;
  using State = std::pair<uint32_t, uint32_t>;
  using Range = State;

  // Get the maximum possible number of radix bits 
  static uint32_t getMaxRadixBits(const size_t lambda, const size_t hintSize) {
    return static_cast<uint32_t>(log2(static_cast<double>(lambda * (1u << 20)) / hintSize));
  }

  // Compute the list [(splineSize, rho)], such that: knotSize * splineSize + hintSize * 2^rho <= memoryLimit
  static std::vector<State> computeStates(const size_t lambda, const size_t knotSize, const size_t hintSize) {
    assert((knotSize != 0) && (hintSize != 0));
    uint32_t maxRho = getMaxRadixBits(lambda, hintSize);
    // Transform MiB into Bytes
    double memoryLimit = lambda * (1u << 20);

    // And compute for each "rho" the corresponding "splineSize"
    std::vector<State> states;
    for (unsigned rho = 0; rho <= maxRho; ++rho) {
      uint32_t splineSize = static_cast<uint32_t>((memoryLimit - hintSize * (1ull << rho)) / knotSize);
      states.push_back(std::make_pair(splineSize, rho));
    }
    return states;
  }
 
  // Upgrade the states, by adding 1 to each remained 'eps' due to downcasting from double in 'solve'
  static std::vector<State> upgradeStates(const std::vector<State>& states) {
    std::vector<State> result(states.size());
    unsigned index = 0;
    for (auto [eps, rho] : states)
      result[index++] = std::make_pair(1 + eps, rho);
    return result;
  }

  // If the states [(eps, rho)] have eps-duplicates, take the one with the biggest rho.
  static std::vector<State> reduceStates(const std::vector<State>& states) {
    std::vector<State> result;
    for (auto state : states) {
      if (result.empty()) {
        result.push_back(state);
      } else if (result.back().first == state.first) {
        result[result.size() - 1] = state;
      } else {
        result.push_back(state);
      }
    }
    return result;
  }
  
  // Generate keys with reservoir sampling
  template<class KeyType, class Generator>
  std::vector<KeyType> sample(const std::vector<Coord>& cdf, size_t capacity, Generator& engine) {
    // Compute the reservoir sampling
    std::vector<KeyType> keys(capacity);
    for (unsigned index = 0; index != capacity; ++index)
      keys[index] = cdf[index].first;

    size_t expand = capacity + 1;
    for (unsigned index = capacity, limit = cdf.size(); index != limit; ++index) {
      size_t r = std::uniform_int_distribution<size_t>{0, expand++}(engine);
      if (r < capacity)
        keys[r] = cdf[index].first;
    }
    return keys;
  }

  // Generate keys for the current CDF
  template <class KeyType>
  static std::vector<KeyType> generateKeys(const std::vector<Coord>& cdf) {
    // The number of keys is proportional to the number of states, in order to differentiate between the best configurations
    unsigned capacity = static_cast<unsigned>(sqrt(cdf.size()));

    // And generate the random keys
    std::mt19937 engine{std::random_device{}()};
    std::vector<KeyType> keys = sample(cdf, capacity, engine);
    std::shuffle(keys.begin(), keys.end(), engine);
    return keys;
  }
  
  // Search in 'data' the 'lookup_key' between 'start' and 'stop'
  template <class KeyType>
  static uint64_t searchFunction(const std::vector<KeyValue<KeyType>>& data, const KeyType lookup_key, size_t start, size_t stop) {
    auto it = std::lower_bound(
        data.begin() + start,
        data.begin() + stop,
        lookup_key,
        [] (const KeyValue<KeyType>& lhs, const KeyType lookup_key) {
          return lhs.key < lookup_key;
        });
    uint64_t result = it->value;
    while ((++it != data.end()) && (it->key == lookup_key))
      result += it->value;
    return result;
  }

  // Measure the execution time for state := (eps, rho) in 'mode'
  template <class KeyType>
  static double measureExecutionTime(const ArsMode mode, const std::vector<KeyValue<KeyType>>& data, const std::vector<Coord>& cdf, State state, std::vector<KeyType>& keys) {
    // Is ARS in the fusion mode?
    if (mode == ArsMode::Fusion) {
      size_t eps = state.first;
      uint32_t rho = state.second, splineSize, shiftWith;
    
      // Build the spline and the radix hints
      std::vector<Coord> spline;
      buildSpline(cdf, spline, splineSize, eps, true);
      std::vector<uint32_t> radixHint;
      buildRadix<KeyType>(mode, rho, data, spline, radixHint, shiftWith);

      // And measure the execution time
      KeyType min = cdf.front().first;
      auto execute = [&]() -> double {
        uint64_t dataSize = data.size();
        auto start = high_resolution_clock::now();
        for (auto key : keys) {
          uint64_t estimation = predict(key, spline, radixHint, min, shiftWith);
          uint64_t start = (estimation < eps) ? 0 : (estimation - eps);
          uint64_t stop = (estimation + eps >= dataSize) ? dataSize : (estimation + eps);
          searchFunction(data, key, start, stop);
        }
        auto stop = high_resolution_clock::now();
        return 1.0 / keys.size() * duration_cast<nanoseconds>(stop - start).count();
      };
      return execute();
    } else {
      uint32_t rho = state.second, shiftWith;

      // Build the radix index on the data itself
      std::vector<Coord> dummySpline;
      std::vector<uint32_t> radixHint;
      buildRadix<KeyType>(mode, rho, data, dummySpline, radixHint, shiftWith);
      
      // And measure the execution time
      KeyType min = data.front().key;
      auto start = high_resolution_clock::now();
      for (auto key : keys) {
        SearchBound bound = radixLookup(key, radixHint, min, shiftWith);
        searchFunction(data, key, bound.start, bound.stop);
      }
      auto stop = high_resolution_clock::now();
      return 1.0 / keys.size() * duration_cast<nanoseconds>(stop - start).count();
    }
  }

  // Solve the translation from [(splineSize, rho)] ("states") to [(eps, rho)] ("result")
  // [a, b[ is the current range from "states" to be analyzed
  // Invariants: e1 < e2 and s1 > s2, because the function has a negative slope
  template <class KeyType>
  static void solve(
      const std::vector<Coord>& cdf,
      const std::vector<State>& states,
      Range eps,
      Range size,
      Range ptr,
      std::vector<State>& result) {
    // Is the range in 'states' empty?
    if (ptr.first == ptr.second)
      return;
    // Temporary store the points (e1, s2) and (e2, s2)
    unsigned e1 = eps.first, e2 = eps.second, s1 = size.first, s2 = size.second;
    
    // Can we fit a linear function?
    double ratio = static_cast<double>(s1 - s2) / (s1 + s2);
    if ((e2 - e1 <= 1) || (ratio < tolerance)) {
      // Careful operations on unsigned for computing the slope of the linear function
      double dx = s1 - s2, dy = e2 - e1, slope = -(dy / dx);

      // And interpolate the points in the range "ptr" 
      for (unsigned index = ptr.first; index != ptr.second; ++index)
        result[index].first = static_cast<unsigned>(slope * (states[index].first - s2) + e2);
      return;
    }

    // Compute the spline with eps := (e1 + e2) / 2, but do not save it
    unsigned mid = e1 + (e2 - e1) / 2, splineSize;
    std::vector<Coord> spline;
    buildSpline<KeyType>(cdf, spline, splineSize, mid);

    // Search where the split the range 'ptr', in regard to 'splineSize'
    unsigned pos = std::lower_bound(states.begin() + ptr.first, states.begin() + ptr.second, splineSize, [](auto l, unsigned x) {
      return l.first >= x;
    }) - states.begin() - ptr.first;

    // Recursively solve the translation, based on the splitted position
    solve(cdf, states, std::make_pair(e1, mid), std::make_pair(s1, splineSize), std::make_pair(ptr.first, ptr.first + pos), result);
    solve(cdf, states, std::make_pair(mid, e2), std::make_pair(splineSize, s2), std::make_pair(ptr.first + pos, ptr.second), result);
  }

  // Wrapper for the translation of states [lastPtr, currPtr[
  // 'lastSplineSize' corresponds to 'eps' / 2, whereas 'currSplineSize' directly to 'eps'
  template <class KeyType>
  static void solveWrapper(
      const std::vector<Coord>& cdf,
      const std::vector<State>& states,
      unsigned eps,
      unsigned lastSplineSize,
      unsigned currSplineSize,
      unsigned lastPtr,
      unsigned currPtr,
      std::vector<State>& result) {
    // Check if we can already make the translation (eps = 1 or eps = 2)
    if (eps <= 2) {
      for (unsigned index = lastPtr; index != currPtr; ++index)
        result[index].first = eps;
    } else if (lastPtr != currPtr) {
      solve(cdf, states, std::make_pair(eps >> 1, eps), std::make_pair(lastSplineSize, currSplineSize), std::make_pair(lastPtr, currPtr), result);
    }
  }

  // The actual translation of states, making use of exponential search
  template <class KeyType>
  static std::vector<State> epsTranslation(const std::vector<Coord>& cdf, const std::vector<State>& states) {
    // Get rid of spline sizes lower than 1
    // Instead, emulate a radix index at the end
    unsigned limit = states.size();
    while (limit && states[limit - 1].first <= 1)
      --limit;

    // Init the rho-values of the result, since they remain the same
    std::vector<State> result(limit);
    for (unsigned index = 0; index != limit; ++index)
      result[index].second = states[index].second;

    // Exponential search
    unsigned eps = 0, ptr = 0, lastSplineSize = 0, currSplineSize;
    while (ptr != limit) {
      // Build the spline for the upcoming eps
      eps = (!eps) ? 1 : (eps << 1);
      std::vector<Coord> tmpSpline;
      buildSpline<KeyType>(cdf, tmpSpline, currSplineSize, eps);
      
      // Forward the pointer
      unsigned save = ptr;
      while ((ptr != limit) && (states[ptr].first >= currSplineSize))
        ++ptr;
      
      // And solve the translation for the current range [save, ptr[
      solveWrapper(cdf, states, eps, lastSplineSize, currSplineSize, save, ptr, result);
      lastSplineSize = currSplineSize;
    }
#ifdef TRANSLATION_VERBOSE
    std::cout << "Final states (spline size -> eps, rho): " << std::endl;
    for (unsigned index = 0; index != limit; ++index)
      std::cout << index << ": (spline size=" << states[index].first << " -> eps=" << result[index].first << ", rho=" << result[index].second << ")" << std::endl;
#endif
    return result;
  }

  // Return the best state (eps, rho) for the current data
  template <class KeyType>
  static std::pair<State, std::pair<double, ArsMode>> analyzeData(const std::vector<KeyValue<KeyType>>& data, const std::vector<Coord>& cdf, const size_t lambda) {
    // Compute the states (eps, rho)
    std::vector<State> states = computeStates(lambda, sizeof(Coord), sizeof(uint32_t));
    std::cerr << "Finding candidates for the best configuration.." << std::endl;
    states = epsTranslation(cdf, states);
    states = reduceStates(states);
    states = upgradeStates(states);

    // Benchmark the states and take the best one
    std::vector<KeyType> keys = generateKeys(cdf);
    std::cerr << "Benchmarking them.." << std::endl;
    std::pair<State, std::pair<double, ArsMode>> configuration = std::make_pair(std::make_pair(0, 0), std::make_pair(-1, ArsMode::Fusion));
    for (State state : states) {
      double curr = measureExecutionTime<KeyType>(ArsMode::Fusion, data, cdf, state, keys);
#ifdef BENCHMARK_VERBOSE
      std::cout << "(eps=" << state.first << ", rho=" << state.second << ") took " << curr << "ns" << std::endl;
#endif
      if ((configuration.second.first == -1) || (curr < configuration.second.first))
        configuration = std::make_pair(state, std::make_pair(curr, ArsMode::Fusion));
    }

    // Finally benchmark the radix index
    State noSpline = std::make_pair(0, getMaxRadixBits(lambda, sizeof(uint32_t))); 
    double curr = measureExecutionTime<KeyType>(ArsMode::RadixOnly, data, cdf, noSpline, keys);
#ifdef BENCHMARK_VERBOSE
    std::cout << "Radix-only (rho=" << noSpline.second << ") took " << curr << "ns" << std::endl;
#endif
    if ((configuration.second.first == -1) || (curr < configuration.second.first))
      configuration = std::make_pair(noSpline, std::make_pair(curr, ArsMode::RadixOnly));
    return configuration;
  }
}

