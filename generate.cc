#include <iostream>
#include <random>
#include <vector>

#include "searches/branching_binary_search.h"
#include "util.h"

using namespace std;

// We ensure that there are no more qualifying entries for a given lookup
// than specified here.
// Setting a low value (e.g., 100) here ensures that the checksum computation
// doesn't dominate performance.
constexpr size_t max_num_qualifying = 100;

// Maximum number of retries to find a lookup key that has at most
// `max_num_qualifying entries`.
constexpr size_t max_num_retries = 100;

// Generates `num_lookups` lookups such that `negative_lookup_ratio` lookups are
// negative.
template <class KeyType, class T>
vector<EqualityLookup<KeyType>> generate_equality_lookups(
    const vector<Row<KeyType>>& data, const vector<T>& unique_keys,
    const size_t num_lookups, const double negative_lookup_ratio) {
  vector<EqualityLookup<KeyType>> lookups;
  lookups.reserve(num_lookups);
  util::FastRandom ranny(42);
  size_t num_generated = 0;
  size_t num_retries = 0;
  BranchingBinarySearch<KeyType> bbs;

  // Required to generate negative lookups within data domain.
  const KeyType min_key = data.front().key;
  const KeyType max_key = data.back().key;

  while (num_generated < num_lookups) {
    if (num_retries == 0 && negative_lookup_ratio > 0.0 &&
        ranny.ScaleFactor() < negative_lookup_ratio) {
      // Generate negative lookup.
      KeyType negative_lookup;
      size_t num_qualifying = 1;
      while (num_qualifying > 0) {
        // Draw lookup key from within data domain.
        negative_lookup = (ranny.ScaleFactor() * (max_key - min_key)) + min_key;
        bbs.search(data, negative_lookup, &num_qualifying, 0, data.size());
      }
      lookups.push_back({negative_lookup, util::NOT_FOUND});
      ++num_generated;
      continue;
    }

    // Generate positive lookup.

    // Draw lookup key from unique keys.
    const uint64_t offset = ranny.RandUint32(0, unique_keys.size() - 1);
    const KeyType lookup_key = unique_keys[offset];

    // Perform binary search on original keys.
    size_t num_qualifying;
    const uint64_t result =
        bbs.search(data, lookup_key, &num_qualifying, 0, data.size());

    if (num_qualifying > max_num_qualifying) {
      // Too many qualifying entries.
      ++num_retries;
      if (num_retries > max_num_retries)
        util::fail("generate_equality_lookups: exceeded max number of retries");
      // Try a different lookup key.
      continue;
    }
    lookups.push_back({lookup_key, result});
    ++num_generated;
    num_retries = 0;
  }
  return lookups;
}

template <class KeyType>
vector<EqualityLookup<KeyType>> generate_equality_lookups_from_trace(
    const vector<Row<KeyType>>& data, const vector<KeyType> keys) {
  vector<EqualityLookup<KeyType>> lookups;
  BranchingBinarySearch<KeyType> bbs;

  size_t nq;
  for (KeyType key : keys) {
    lookups.push_back({key, bbs.search(data, key, &nq, 0, keys.size())});
  }

  return lookups;
}

const string to_nice_number(uint64_t num) {
  const uint64_t THOUSAND = 1000;
  const uint64_t MILLION = 1000 * THOUSAND;
  const uint64_t BILLION = 1000 * MILLION;

  if (num >= BILLION && (num / BILLION) * BILLION == num) {
    return to_string(num / BILLION) + "B";
  }
  if (num >= MILLION && (num / MILLION) * MILLION == num) {
    return to_string(num / MILLION) + "M";
  }
  if (num >= THOUSAND && (num / THOUSAND) * THOUSAND == num) {
    return to_string(num / THOUSAND) + "K";
  }
  return to_string(num);
}

template <class KeyType>
void print_equality_lookup_stats(
    const vector<EqualityLookup<KeyType>>& lookups) {
  size_t negative_count = 0;
  for (const auto& lookup : lookups)
    if (lookup.result == util::NOT_FOUND) ++negative_count;
  std::cout << "negative lookup ratio: "
            << static_cast<double>(negative_count) / lookups.size()
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3)
    util::fail(
        "usage: ./generate <data file> <num lookups> [negative lookup ratio]");

  const string filename = argv[1];
  const DataType type = util::resolve_type(filename);
  size_t num_lookups = stoull(argv[2]);
  double negative_lookup_ratio = 0.0;
  if (argc >= 4) negative_lookup_ratio = stod(argv[3]);
  if (negative_lookup_ratio < 0 || negative_lookup_ratio > 1) {
    util::fail("negative lookup ratio must be between 0 and 1.");
  }

  switch (type) {
    case DataType::UINT32: {
      // Load data.
      const vector<uint32_t> keys = util::load_data<uint32_t>(filename);

      if (!is_sorted(keys.begin(), keys.end()))
        util::fail("keys have to be sorted (read 32-bit keys)");

      // Get duplicate-free copy: we draw lookup keys from unique keys.
      vector<uint32_t> unique_keys = util::remove_duplicates(keys);

      // Add artificial values to original keys.
      vector<Row<uint32_t>> data = util::add_values(keys);

      // Generate benchmarks.
      vector<EqualityLookup<uint32_t>> equality_lookups;
      if (filename.find("corp") != std::string::npos) {
        // load the precomputed trace
        std::cout << "generating lookups from trace" << std::endl;
        std::string trace_name = filename;
        trace_name.replace(trace_name.find("corp_200M_uint32"),
                           sizeof("corp_200M_uint32") - 1, "corp_trace_uint32");

        vector<uint32_t> keys = util::load_data<uint32_t>(trace_name);
        num_lookups = keys.size();
        equality_lookups = generate_equality_lookups_from_trace(data, keys);
      } else {
        equality_lookups = generate_equality_lookups(
            data, unique_keys, num_lookups, negative_lookup_ratio);
      }

      print_equality_lookup_stats(equality_lookups);
      util::write_data(equality_lookups, filename + "_equality_lookups_" +
                                             to_nice_number(num_lookups));

      break;
    }
    case DataType::UINT64: {
      // Load data.
      const vector<uint64_t> keys = util::load_data<uint64_t>(filename);

      if (!is_sorted(keys.begin(), keys.end()))
        util::fail("keys have to be sorted (read 64-bit keys)");

      // Get duplicate-free copy: we draw lookup keys from unique keys.
      vector<uint64_t> unique_keys = util::remove_duplicates(keys);

      // Add artificial values to original keys.
      vector<Row<uint64_t>> data = util::add_values(keys);

      // Generate benchmarks.
      vector<EqualityLookup<uint64_t>> equality_lookups;
      if (filename.find("corp") != std::string::npos) {
        // load the precomputed trace
        std::cout << "generating lookups from trace" << std::endl;
        std::string trace_name = filename;
        trace_name.replace(trace_name.find("corp_200M_uint64"),
                           sizeof("corp_200M_uint64") - 1, "corp_trace_uint64");

        vector<uint64_t> keys = util::load_data<uint64_t>(trace_name);
        num_lookups = keys.size();
        equality_lookups = generate_equality_lookups_from_trace(data, keys);
      } else {
        equality_lookups = generate_equality_lookups(
            data, unique_keys, num_lookups, negative_lookup_ratio);
      }

      print_equality_lookup_stats(equality_lookups);
      util::write_data(equality_lookups, filename + "_equality_lookups_" +
                                             to_nice_number(num_lookups));

      break;
    }
  }

  return 0;
}
