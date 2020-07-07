#ifndef SOSDB_RADIX_SPLINE_SEARCH_H
#define SOSDB_RADIX_SPLINE_SEARCH_H

#include "../util.h"
#include "../polynomial/spline_util.h"
#include "../polynomial/polynomial_util.h"
#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>
#include <string.h>
#include "base.h"

namespace polynomial_spline{

template<class KeyType, int size_scale>
class RadixSpline : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>> &data)
  {
    if (!tuning_set_) {
      util::fail("RS build called without calling set_tuning first!");
    }
    
    data_ = data;
    
    data_size = data_.size();
    max_key_ = data_.back().key;

    return util::timing([&] {
      cdf = spline_util::buildCdf<KeyType>(data_);
      
      // Compute the spline (use the maxError from applicable)
      spline = spline_util::tautString(cdf, cdf.back().second, max_error);
      spline_size = spline.size();
      
      // Store the spline x-coordinates in a radix index
      buildRadix();
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const
  {    
    uint64_t est_error = max_error + extra_fp_error;
    
    uint64_t estimate = segmentInterpolation(process(lookup_key), lookup_key);
    uint64_t left = (estimate > est_error) ? (estimate - est_error) : 0;
    uint64_t right = (estimate + est_error < data_size) ? (estimate + est_error) : data_size;
    return (SearchBound) { left, right };
  }
  

  std::string name() const
  {
    return "RS";
  }

  std::size_t size() const
  {
    return sizeof(*this) 
      + spline.size() * sizeof(Coord) // for spline
      + ((1ull << num_radix_bits_) + 1) * sizeof(uint32_t); // for radix_hint
  }

  int variant() const { return size_scale; }

  // Choose the most appropiate pair (max_error, num_radix_bits_) by the filename
  bool applicable(__attribute__((unused)) bool unique_keys, const std::string &data_filename) 
  { 
    std::string cut = data_filename.data();
    extra_fp_error = 0;
    
    // Cut the prefix of the filename
    cut.erase(cut.begin(), cut.begin() + cut.find(prefix) + strlen(prefix));

    // TODO segfault
    /*if (cut == "books_800M_uint64") return false;
    if (cut == "osm_cellids_800M_uint64") return false;
    if (cut == "osm_cellids_600M_uint64") return false;
    if (cut == "normal_800M_uint64") return false;*/
    
    // Normal
    if (cut == "normal_200M_uint64"
        || cut == "normal_400M_uint64"
        || cut == "normal_600M_uint64"
        || cut == "normal_800M_uint64"
        || cut == "normal_200M_uint32") {
      if (size_scale == 1) set_tuning(32, 18);
      if (size_scale == 2) set_tuning(16, 18);
      if (size_scale == 3) set_tuning(8, 18);
      if (size_scale == 4) set_tuning(4, 18);
      if (size_scale == 5) set_tuning(1, 18);
      if (size_scale == 6) set_tuning(1, 17);
      if (size_scale == 7) set_tuning(1, 16);
      if (size_scale == 8) set_tuning(1, 15);
      if (size_scale == 9) set_tuning(1, 14);
      if (size_scale == 10) set_tuning(1, 13);
    }

    // Lognormal
    if (cut == "lognormal_200M_uint32") set_tuning(1, 20);
    if (cut == "lognormal_200M_uint64") set_tuning(1, 25);

    // Uniform dense
    if (cut == "uniform_dense_200M_uint32") set_tuning(0, 15);
    if (cut == "uniform_dense_200M_uint64") set_tuning(0, 15);

    // Uniform sparse
    if (cut == "uniform_sparse_200M_uint32") set_tuning(6, 24);
    if (cut == "uniform_sparse_200M_uint64") set_tuning(5, 25);

    // Osm
    if (cut == "osm_cellids_200M_uint64"
        || cut == "osm_cellids_400M_uint64"
        || cut == "osm_cellids_600M_uint64"
        || cut == "osm_cellids_800M_uint64") {
      extra_fp_error = 32;
      if (size_scale == 1) set_tuning(13, 25);
      if (size_scale == 2) set_tuning(26, 23);
      if (size_scale == 3) set_tuning(32, 19);
      if (size_scale == 4) set_tuning(64, 18);
      if (size_scale == 5) set_tuning(128, 18);
      if (size_scale == 6) set_tuning(256, 16);
      if (size_scale == 7) set_tuning(512, 15);
      if (size_scale == 8) set_tuning(2*1024, 14);
      if (size_scale == 9) set_tuning(2*2048, 3);
      if (size_scale == 10) set_tuning(2*4096, 3);
    }

    // Wiki
    if (cut == "wiki_ts_200M_uint64") {
      if (size_scale >= 3) { extra_fp_error = 64; }
      else { extra_fp_error = 128; }
      
      
      if (size_scale == 1) set_tuning(9, 21);
      if (size_scale == 2) set_tuning(10, 18);
      if (size_scale == 3) set_tuning(32, 18);
      if (size_scale == 4) set_tuning(48, 18);
      if (size_scale == 5) set_tuning(84, 18);
      if (size_scale == 6) set_tuning(256, 16);
      if (size_scale == 7) set_tuning(512, 15);
      if (size_scale == 8) set_tuning(2*1024, 14);
      if (size_scale == 9) set_tuning(2*2048, 3);
      if (size_scale == 10) set_tuning(2*4096, 3);
    }

    // Books (or amazon in the paper)
    if (cut == "books_200M_uint32") set_tuning(14, 20);
    if (cut == "books_200M_uint64"
        || cut == "books_400M_uint64"
        || cut == "books_600M_uint64"
        || cut == "books_800M_uint64"
        || cut == "books_200M_uint32") {

      // TODO this is the original optimal config, but it gives wrong results.
      // if (size_scale == 1) set_tuning(11, 22);
      if (size_scale == 1) set_tuning(64, 18);
      if (size_scale == 2) set_tuning(82, 18);
      if (size_scale == 3) set_tuning(98, 18);
      if (size_scale == 4) set_tuning(256, 18);
      if (size_scale == 5) set_tuning(512, 16);
      if (size_scale == 6) set_tuning(1024, 14);
      if (size_scale == 7) set_tuning(1024, 12);
      if (size_scale == 8) set_tuning(2*1024, 10);
      if (size_scale == 9) set_tuning(2*2048, 3);
      if (size_scale == 10) set_tuning(2*4096, 3);

    }
    
    if (cut == "books_400M_uint64") {
      extra_fp_error = 64;
    }

    if (cut == "books_600M_uint64") {
      extra_fp_error = 128;
    }
    
    if (cut == "books_800M_uint64") {
      extra_fp_error = 128;
    }
    
    // Fb
    if (cut == "fb_200M_uint64" || cut == "fb_200M_uint32") {
      if (size_scale == 1) set_tuning(2, 1);
      if (size_scale == 2) set_tuning(4, 1);
      if (size_scale == 3) set_tuning(10, 1);
      if (size_scale == 4) set_tuning(32, 1);
      if (size_scale == 5) set_tuning(128, 1);
      if (size_scale == 6) set_tuning(512, 1);
      if (size_scale == 7) set_tuning(1024, 1);
      if (size_scale == 8) set_tuning(2*1024, 1);
      if (size_scale == 9) set_tuning(2*2048, 1);
      if (size_scale == 10) set_tuning(2*4096, 1);
      
    }
    return true;
  }
    
 private:
  std::vector<Coord> cdf;
  std::vector<Coord> spline;
  uint64_t spline_size, data_size;
  uint64_t max_error;
  uint64_t extra_fp_error;
  bool use_errors;
  KeyType max_key_;
  

  static constexpr const char* prefix = "data/";
  
  // Radix precomputing for spline x-coordinates
  uint32_t num_radix_bits_ = 0x0;
  uint64_t n_;
  KeyType min_;
  KeyType max_;
  KeyType shift_bits_;
  bool tuning_set_ = false;
  std::vector<uint32_t> radix_hint_; // is allocated afterwards 

  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;

  void set_tuning(uint64_t maxError, uint32_t radixBits, bool useErrors = false)
  // set the variables of the tuning: (max_error, num_radix_bits_, use_errors)
  {
    max_error = maxError;
    num_radix_bits_ = radixBits;
    use_errors = useErrors; // this is not even used (but remains for consistency)
    tuning_set_ = true;
  }

  inline uint64_t shift_bits(const uint64_t val) 
  // it's used only when keyType == uint64_t
  {
    const uint32_t clz = __builtin_clzl(val);
    if ((64 - clz) < num_radix_bits_)
      return 0;
    else
      return 64 - num_radix_bits_ - clz;
  }

  inline uint32_t shift_bits(const uint32_t val) 
  // it's used only when keyType == uint64_t
  {
    const uint32_t clz = __builtin_clz(val);
    if ((32 - clz) < num_radix_bits_)
      return 0;
    else
      return 32 - num_radix_bits_ - clz;
  }
  
  void buildRadix()
  // create a radix index for the spline knots
  {
    assert(num_radix_bits_);
    
    // Alloc the memory for the hints
    radix_hint_.resize((1ull << num_radix_bits_) + 1, 0);
    
    // Compute the number of bits to shift with
    n_ = spline.size();
    min_ = spline.front().first;
    max_ = spline.back().first;
    shift_bits_ = shift_bits(max_ - min_);

    // Compute the hints
    radix_hint_[0] = 0;
    uint64_t prev_prefix = 0;
    for (uint64_t i = 0; i < n_; ++i) {
      uint64_t tmp = static_cast<uint64_t>(spline[i].first);
      uint64_t curr_prefix = (tmp - min_) >> shift_bits_;
      if (curr_prefix != prev_prefix) {
        for (uint64_t j = prev_prefix + 1; j <= curr_prefix; ++j)
          radix_hint_[j] = i;
        prev_prefix = curr_prefix;
      }
    }
    
    // Margin hint values
    for (; prev_prefix < (1ull << num_radix_bits_); ++prev_prefix)
      radix_hint_[prev_prefix + 1] = n_;

  }
  
  inline uint32_t process(uint64_t x) const 
  // find on which spline segment "x" lies
  {
    // Compute index.
    uint32_t index;
    const uint64_t p = (x - min_) >> shift_bits_;
    uint32_t begin = radix_hint_[p];
    uint32_t end = radix_hint_[p + 1];

    // Note from Ryan:
    // Coord (spline[begin].first) is a double, whereas key is a uint64_t.
    // Comparison is not linear under casting. Example:
    // 35207349327993288 >= 35207349327993289 is false
    // (double)35207349327993288 >= (double)35207349327993289 is true.
    // Unclear what, if any, of the below code is unsound here,
    // but at least the condition above makes "case 1" return 0 on the OSM dataset.
    // added a std::max() in the return to address this temporarily
    
    // Return the index of the segment
    switch (end - begin) {
      case 0: index = end;
        break;
      case 1: index = (spline[begin].first >= x) ? begin : end;
        break;
      case 2:
        index = ((spline[begin].first >= x) ? begin : ((spline[begin
            + 1].first >= x) ? (begin + 1) : end));
        break;
      case 3:
        index = ((spline[begin].first >= x) ? begin : ((spline[begin
            + 1].first >= x) ? (begin + 1) : ((spline[begin + 2].first
            > x) ? (begin + 2) : end)));
        break;
      default:
        index = std::lower_bound(spline.begin() + begin,
                                 spline.begin() + end,
                                 x,
                                 [](const Coord& a,
                                    const uint64_t lookup_key) {
                                   return a.first < lookup_key;
                                 }) - spline.begin();
        break;
    }
    // Go a position back
    return std::max(index, (uint32_t) 1) - 1;
  }
  
  double segmentInterpolation(uint64_t segment, const double x) const
  // get f(x) at segment
  {
    Coord down = spline[segment], up = spline[segment + 1];
    double slope = (down.second - up.second) / (down.first - up.first);
    return down.second + (x - down.first) * slope; 
  }
};

}

template<class KeyType, int size_scale>
using RadixSpline = polynomial_spline::RadixSpline<KeyType, size_scale>;

#endif //SOSDB_RADIX_SPLINE_SEARCH_H
