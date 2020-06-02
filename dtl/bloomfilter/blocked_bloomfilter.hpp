#pragma once

#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_tune.hpp>
#include <dtl/bloomfilter/hash_family.hpp>

#include <boost/math/common_factor.hpp>


namespace dtl {

//namespace internal {
//
////===----------------------------------------------------------------------===//
//// Check whether the given blocked bloom filter configuration is valid.
//template<u32 w, u32 s, u32 k>
//struct is_valid_bbf_config {
//  static constexpr u1 value = (k % s) == 0;
//};
////===----------------------------------------------------------------------===//
//
//
//////===----------------------------------------------------------------------===//
////// Signals an invalid BBF configuration.
////struct null_bbf {};
//////===----------------------------------------------------------------------===//
//
//
////===----------------------------------------------------------------------===//
//// Checks whether the given BBF configuration is valid and resolves the
//// corresponding BBF type.
//// If the configuration is invalid this resolves to 'null_bbf'.
//template<
//    typename key_t, typename word_t,
//    u32 w, u32 s, u32 k, dtl::block_addressing a, u1 early_out,
//    typename enable = void
//>
//struct bbf_switch;
//
//template<
//    typename key_t, typename word_t,
//    u32 w, u32 s, u32 k, dtl::block_addressing a, u1 early_out
//>
//struct bbf_switch<key_t, word_t, w, s, k, a, early_out,
//                  typename std::enable_if<is_valid_bbf_config<w, s, k>::value>::type> {
//  using type = dtl::blocked_bloomfilter_logic<key_t, dtl::hash::stat::mul32, word_t, w, s, k, a, early_out>;
//};
//
//template<
//    typename key_t, typename word_t,
//    u32 w, u32 s, u32 k, dtl::block_addressing a, u1 early_out
//>
//struct bbf_switch<key_t, word_t, w, s, k, a, early_out,
//                  typename std::enable_if<!is_valid_bbf_config<w, s, k>::value>::type> {
////  using type = null_bbf; // invalid configuration
//  using type = dtl::blocked_bloomfilter_logic<key_t, dtl::hash::stat::mul32, word_t, 1, 1, 1, dtl::block_addressing::POWER_OF_TWO, false>;
//};
////===----------------------------------------------------------------------===//
//
//
//} // namespace internal


//===----------------------------------------------------------------------===//
template<typename Tw = $u32>
struct blocked_bloomfilter {

  using key_t = $u32;
  using hash_value_t = $u32;
  using word_t = Tw;

  static constexpr u1 early_out = false; // TODO make configurable and also adaptive


  // The operations used for dynamic dispatching.
  enum class op_t {
    CONSTRUCT,  // constructs the Bloom filter logic
    BIND,       // assigns the function pointers of the Bloom filter API
    DESTRUCT,   // destructs the Bloom filter logic
  };

  template<u32 word_cnt, u32 sector_cnt, u32 k, dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO, u1 early_out = false>
  using bbf = dtl::blocked_bloomfilter_logic<key_t, hasher, word_t, word_cnt, sector_cnt, k, addr, early_out>;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The (desired) bit length of the Bloom filter.
  $u64 m;
  /// The (actual) bit length of the Bloom filter.
  $u64 m_actual;
  /// The number of bits set per entry.
  $u32 k;
  /// The number of words per block.
  $u32 word_cnt_per_block;
  /// The number of sectors per block.
  $u32 sector_cnt;
  /// Pointer to the Bloom filter logic instance.
  void* instance = nullptr;
  /// A container for the (hardware dependent) tuning parameters.
  const blocked_bloomfilter_tune& tune;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The API functions. (Function pointers to the actual implementation.)
  //===----------------------------------------------------------------------===//
  std::function<void(__restrict word_t* /*filter data*/, const key_t /*key*/)>
  insert;

  std::function<void(__restrict word_t* /*filter data*/, const key_t* /*keys*/, u32 /*key_cnt*/)>
  batch_insert;

  std::function<$u1(const __restrict word_t* /*filter data*/, const key_t /*key*/)>
  contains;

  std::function<$u64(const __restrict word_t* /*filter data*/,
                     const key_t* /*keys*/, u32 /*key_cnt*/,
                     $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter(const size_t m, u32 k, u32 word_cnt_per_block, u32 sector_cnt,
                      const blocked_bloomfilter_tune& tune)
      : m(m), k(k), word_cnt_per_block(word_cnt_per_block), sector_cnt(sector_cnt), tune(tune) {

    // Construct the Bloom filter logic instance.
    dispatch(*this, op_t::CONSTRUCT);

    // Bind the API functions.
    dispatch(*this, op_t::BIND);

    // Check whether the constructed filter matches the given arguments.
    if (this->k != k
        || this->word_cnt_per_block != word_cnt_per_block
        || this->sector_cnt != sector_cnt) {
      dispatch(*this, op_t::DESTRUCT);
      throw std::invalid_argument("Invalid configuration: k=" + std::to_string(k)
                                  + ", w=" + std::to_string(word_cnt_per_block)
                                  + ", s=" + std::to_string(sector_cnt));
    }

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter(blocked_bloomfilter&& src)
      : m(src.m), m_actual(src.m_actual), k(src.k),
        word_cnt_per_block(src.word_cnt_per_block), sector_cnt(src.sector_cnt),
        instance(src.instance),
        insert(std::move(src.insert)),
        batch_insert(std::move(src.batch_insert)),
        contains(std::move(src.contains)),
        batch_contains(std::move(src.batch_contains)),
        tune(std::move(src.tune)) {
    // Invalidate pointer in src
    src.instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  ~blocked_bloomfilter() {
    // Destruct logic instance (if any).
    if (instance != nullptr) dispatch(*this, op_t::DESTRUCT);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter&
  operator=(blocked_bloomfilter&& src) {
    m = src.m;
    m_actual = src.m_actual;
    k = src.k;
    word_cnt_per_block = src.word_cnt_per_block;
    sector_cnt = src.sector_cnt;
    instance = src.instance;
    insert = std::move(src.insert);
    batch_insert = std::move(src.batch_insert);
    contains = std::move(src.contains);
    batch_contains = std::move(src.batch_contains);
    // invalidate pointers
    src.instance = nullptr;
    return *this;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Dynamic Dispatching
  //===----------------------------------------------------------------------===//
  //TODO make private
  static void
  dispatch(blocked_bloomfilter& instance, op_t op) {
      switch (instance.word_cnt_per_block) {
        case  1: _s< 1>(instance, op); break;
        case  2: _s< 2>(instance, op); break;
        case  4: _s< 4>(instance, op); break;
        case  8: _s< 8>(instance, op); break;
        case 16: _s<16>(instance, op); break;
        default:
          throw std::invalid_argument("The given 'word_cnt_per_block' is not supported.");
      }
  };


  static constexpr u32 max_sector_cnt_per_word = sizeof(word_t); // a sector is at least one byte in size

  template<u32 w>
  static void
  _s(blocked_bloomfilter& instance, op_t op) {
    switch (instance.sector_cnt) {
      case  1: _k<w,  1>(instance, op); break;
      case  2: _k<w, boost::static_unsigned_min<(w * max_sector_cnt_per_word),  2>::value>(instance, op); break;
      case  4: _k<w, boost::static_unsigned_min<(w * max_sector_cnt_per_word),  4>::value>(instance, op); break;
      case  8: _k<w, boost::static_unsigned_min<(w * max_sector_cnt_per_word),  8>::value>(instance, op); break;
      case 16: _k<w, boost::static_unsigned_min<(w * max_sector_cnt_per_word), 16>::value>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'sector_cnt' is not supported.");
    }
  }


  template<u32 w, u32 s>
  static void
  _k(blocked_bloomfilter& instance, op_t op) {
    switch (instance.k) {
      case  1: _a<w, s, boost::static_unsigned_max<1, s>::value>(instance, op); break;
      case  2: _a<w, s, boost::static_unsigned_max<( 2 % s == 0 ?  2 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  3: _a<w, s, boost::static_unsigned_max<( 3 % s == 0 ?  3 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  4: _a<w, s, boost::static_unsigned_max<( 4 % s == 0 ?  4 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  5: _a<w, s, boost::static_unsigned_max<( 5 % s == 0 ?  5 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  6: _a<w, s, boost::static_unsigned_max<( 6 % s == 0 ?  6 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  7: _a<w, s, boost::static_unsigned_max<( 7 % s == 0 ?  7 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  8: _a<w, s, boost::static_unsigned_max<( 8 % s == 0 ?  8 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  9: _a<w, s, boost::static_unsigned_max<( 9 % s == 0 ?  9 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 10: _a<w, s, boost::static_unsigned_max<(10 % s == 0 ? 10 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 11: _a<w, s, boost::static_unsigned_max<(11 % s == 0 ? 11 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 12: _a<w, s, boost::static_unsigned_max<(12 % s == 0 ? 12 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 13: _a<w, s, boost::static_unsigned_max<(13 % s == 0 ? 13 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 14: _a<w, s, boost::static_unsigned_max<(14 % s == 0 ? 14 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 15: _a<w, s, boost::static_unsigned_max<(15 % s == 0 ? 15 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 16: _a<w, s, boost::static_unsigned_max<16 , s>::value>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'k' is not supported.");
    }
  }


  template<u32 w, u32 s, u32 k>
  static void
  _a(blocked_bloomfilter& instance, op_t op) {
    dtl::block_addressing addr = dtl::is_power_of_two(instance.m)
                                 ? dtl::block_addressing::POWER_OF_TWO
                                 : dtl::block_addressing::MAGIC;
    switch (addr) {
      case dtl::block_addressing::POWER_OF_TWO: _u<w, s, k, dtl::block_addressing::POWER_OF_TWO>(instance, op); break;
      case dtl::block_addressing::MAGIC:        _u<w, s, k, dtl::block_addressing::MAGIC>(instance, op);        break;
      case dtl::block_addressing::DYNAMIC:      /* must not happen */                                           break;
    }
  }


  template<u32 w, u32 s, u32 k, dtl::block_addressing a>
  static void
  _u(blocked_bloomfilter& instance, op_t op) {
    switch (instance.tune.get_unroll_factor(k, sizeof(word_t), w, s, a)) {
      case  0: _o<w, s, k, a,  0>(instance, op); break;
      case  1: _o<w, s, k, a,  1>(instance, op); break;
      case  2: _o<w, s, k, a,  2>(instance, op); break;
      case  4: _o<w, s, k, a,  4>(instance, op); break;
      case  8: _o<w, s, k, a,  8>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'unroll_factor' is not supported.");
    }
  }


  template<u32 w, u32 s, u32 k, dtl::block_addressing a, u32 unroll_factor>
  static void
  _o(blocked_bloomfilter& instance, op_t op) {
//    using _t = typename internal::bbf_switch<key_t, word_t, w, s, k, a, early_out>::type;
    using _t = bbf<w, s, k, a, early_out>;
    switch (op) {
      case op_t::CONSTRUCT: instance._construct_logic<_t>();           break;
      case op_t::BIND:      instance._bind_logic<_t, unroll_factor>(); break;
      case op_t::DESTRUCT:  instance._destruct_logic<_t>();            break;
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Constructs a blocked Bloom filter logic instance.
  template<
      typename bf_t
  >
  void
  _construct_logic() {
    // Instantiate a Bloom filter logic.
    bf_t* bf = new bf_t(m);
    instance = bf;
    k = bf_t::k;
    word_cnt_per_block = bf_t::word_cnt_per_block;
    sector_cnt = bf_t::sector_cnt;

    // Get the actual size of the filter.
    m_actual = bf->get_length();
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Bind API functions to the (previously constructed) Bloom filter logic.
  template<
      typename bf_t,
      u32 unroll_factor = 1
  >
  void
  _bind_logic() {
    using namespace std::placeholders;
    auto* bf = static_cast<bf_t*>(instance);

    // Bind the API functions.
    insert = std::bind(&bf_t::insert, bf, _1, _2);
    batch_insert = std::bind(&bf_t::batch_insert, bf, _1, _2, _3);
    contains = std::bind(&bf_t::contains, bf, _1, _2);

    // SIMD vector length (0 = run scalar code)
    // TODO fix unrolling for word_t = u64
    static constexpr u64 vector_len = dtl::simd::lane_count<key_t> * unroll_factor;
    batch_contains = std::bind(&bf_t::template batch_contains<vector_len>, bf, _1, _2, _3, _4, _5);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Destructs the Bloom filter logic.
  template<typename bf_t>
  void
  _destruct_logic() {
    bf_t* bf = static_cast<bf_t*>(instance);
    delete bf;
    instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the block addressing mode.
  block_addressing get_addressing_mode() const {
    return dtl::is_power_of_two(m)
           ? block_addressing::POWER_OF_TWO
           : block_addressing::MAGIC;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (actual) size in bytes.
  std::size_t
  size_in_bytes() const noexcept {
    return (m_actual + 7) / 8;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (total) number of words.
  std::size_t
  size() const noexcept {
    constexpr u32 word_bitlength = sizeof(word_t) * 8;
    return (m_actual + (word_bitlength - 1)) / word_bitlength;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the name of the Bloom filter instance including the most
  /// important parameters (in JSON).
  std::string
  name() {
    return "{\"name\":\"blocked_bloom_multiword\",\"size\":" + std::to_string(size_in_bytes())
         + ",\"word_size\":" + std::to_string(sizeof(word_t))
         + ",\"k\":" + std::to_string(k)
         + ",\"w\":" + std::to_string(word_cnt_per_block)
         + ",\"s\":" + std::to_string(sector_cnt)
         + ",\"u\":" + std::to_string(tune.get_unroll_factor(k, sizeof(word_t), word_cnt_per_block, sector_cnt, get_addressing_mode()))
         + ",\"e\":" + std::to_string(early_out ? 1 : 0)
         + ",\"addr\":" + (get_addressing_mode() == dtl::block_addressing::POWER_OF_TWO ? "\"pow2\"" : "\"magic\"")
         + "}";
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
