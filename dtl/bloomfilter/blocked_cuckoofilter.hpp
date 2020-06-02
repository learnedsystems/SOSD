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

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_logic.hpp>
#include <dtl/bloomfilter/hash_family.hpp>

#include <boost/math/common_factor.hpp>


namespace dtl {

namespace internal {

//===----------------------------------------------------------------------===//
/// @see $u32& unroll_factor(u32, dtl::block_addressing, u32)
static constexpr u32 max_k = 16;

static
std::array<$u32, max_k * 5 /* different block sizes */ * 2 /* _addressing modes*/>
    unroll_factors_32 = {
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = magic
  };

static
std::array<$u32, max_k * 5 /* different block sizes */ * 2 /* _addressing modes*/>
    unroll_factors_64 = {
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = magic
  };
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// TODO remove
template<typename _word_src_t, typename _cuckoo_t>
struct blocked_cuckoofilter_api_adapter {

  using word_t = _word_src_t;

  using key_t = typename _cuckoo_t::key_t;
  using word_dst_t = typename _cuckoo_t::word_t;

  _cuckoo_t* filter;

  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  void
  insert(word_t* __restrict filter_data, const key_t& key) {
    return filter->insert(reinterpret_cast<word_dst_t*>(filter_data), key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__ __host__
  void
  batch_insert(word_t* __restrict filter_data, const key_t* keys, const uint32_t key_cnt) {
    filter->batch_insert(reinterpret_cast<word_dst_t*>(filter_data),
                         keys, key_cnt);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  bool
  contains(const word_t* __restrict filter_data, const key_t& key) const {
    return filter->contains(reinterpret_cast<const word_dst_t*>(filter_data), key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__  __host__
  uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return filter->batch_contains(reinterpret_cast<const word_dst_t*>(filter_data),
                                  keys, key_cnt, match_positions, match_offset);
  };
  //===----------------------------------------------------------------------===//
};
//===----------------------------------------------------------------------===//

} // namespace internal


struct blocked_cuckoofilter {

  using key_t = $u32;
  using hash_value_t = $u32;
  using word_t = $u32;


// TODO ?
//  template<
//      typename key_t,
//      $u32 hash_fn_no
//  >
//  using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;

  // The operations used for dynamic dispatching.
  enum class op_t {
    CONSTRUCT,  // constructs the filter logic
    BIND,       // assigns the function pointers of the filter API
    DESTRUCT,   // destructs the filter logic
  };

  static constexpr dtl::block_addressing power = dtl::block_addressing::POWER_OF_TWO;
  static constexpr dtl::block_addressing magic = dtl::block_addressing::MAGIC;

  template<u32 block_size_bytes, u32 tag_size_bits, u32 associativity, block_addressing addressing = power>
  using bcf = dtl::blocked_cuckoofilter_logic<block_size_bytes, tag_size_bits, associativity, addressing>;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The (desired) bit length of the filter.
  $u64 m;
  /// The (actual) bit length of the filter.
  $u64 m_actual;
  /// The block size in bytes.
  $u32 block_size_bytes;
  /// The size of a tag (aka signature) in bits.
  $u32 tag_size_bits;
  /// The number of slots per bucket.
  $u32 associativity;
  /// Pointer to the filter logic instance.
  void* instance = nullptr;
  // necessary because internally the word_t may differ
  void* adapter_instance = nullptr;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The API functions. (Function pointers to the actual implementation.)
  //===----------------------------------------------------------------------===//
  std::function<void(__restrict word_t* /*filter data*/, const key_t /*key*/)>
  insert;

  std::function<void(__restrict word_t* /*filter_data*/, const key_t* /*keys*/, u32 /*key_cnt*/)>
  batch_insert;

  std::function<$u1(const __restrict word_t* /*filter_data*/, const key_t /*key*/)>
  contains;

  std::function<$u64(const __restrict word_t* /*filter_ data*/,
                     const key_t* /*keys*/, u32 /*key_cnt*/,
                     $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_cuckoofilter(const size_t m, u32 block_size_bytes = 64, u32 tag_size_bits = 16, u32 associativity = 4)
      : m(m), block_size_bytes(block_size_bytes), tag_size_bits(tag_size_bits), associativity(associativity) {

    // Construct the filter logic instance.
    dispatch(*this, op_t::CONSTRUCT);

    // Bind the API functions.
    dispatch(*this, op_t::BIND);

    // Check whether the constructed filter matches the given arguments.
    if (this->block_size_bytes != block_size_bytes
        || this->tag_size_bits != tag_size_bits
        || this->associativity != associativity) {
      dispatch(*this, op_t::DESTRUCT);
      throw std::invalid_argument("Invalid configuration: block_size_bytes=" + std::to_string(block_size_bytes)
                                  + ", tag_size_bits=" + std::to_string(tag_size_bits)
                                  + ", associativity=" + std::to_string(associativity));
    }

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_cuckoofilter(blocked_cuckoofilter&& src)
      : m(src.m), m_actual(src.m_actual),
        block_size_bytes(src.block_size_bytes),
        tag_size_bits(src.tag_size_bits),
        associativity(src.associativity),
        instance(src.instance),
        insert(std::move(src.insert)),
        batch_insert(std::move(src.batch_insert)),
        contains(std::move(src.contains)),
        batch_contains(std::move(src.batch_contains)) {
    // Invalidate pointer in src
    src.instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  ~blocked_cuckoofilter() {
    // Destruct logic instance (if any).
    if (instance != nullptr) dispatch(*this, op_t::DESTRUCT);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_cuckoofilter&
  operator=(blocked_cuckoofilter&& src) {
    m = src.m;
    m_actual = src.m_actual;
    block_size_bytes = src.block_size_bytes;
    tag_size_bits = src.tag_size_bits;
    associativity = src.associativity;
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
  dispatch(blocked_cuckoofilter& instance, op_t op) {
      switch (instance.block_size_bytes) {
        case   8: _s<  8>(instance, op); break;
        case  32: _s< 32>(instance, op); break;
        case  64: _s< 64>(instance, op); break;
        case 128: _s<128>(instance, op); break;
        case 256: _s<256>(instance, op); break;
        case 512: _s<512>(instance, op); break;
        default:
          throw std::invalid_argument("The given 'block_size_bytes' is not supported.");
      }
  };


  template<u32 b>
  static void
  _s(blocked_cuckoofilter& instance, op_t op) {
    switch (instance.tag_size_bits) {
      case  8:
        switch (instance.associativity) {
          case  4: _a<b,  8, 4>(instance, op); return;
          case  8: _a<b,  8, 8>(instance, op); return;
        }
        break;
      case 10:
        switch (instance.associativity) {
          case  6: _a<b, 10, 6>(instance, op); return;
        }
        break;
      case 12:
        switch (instance.associativity) {
          case  4: _a<b, 12, 4>(instance, op); return;
        }
        break;
      case 16:
        switch (instance.associativity) {
          case  2: _a<b, 16, 2>(instance, op); return;
          case  4: _a<b, 16, 4>(instance, op); return;
        }
        break;
    }
    throw std::invalid_argument("The given 'tag_size_bits/associativity' is not supported.");
  }


  template<u32 b, u32 t, u32 s>
  static void
  _a(blocked_cuckoofilter& instance, op_t op) {
    dtl::block_addressing addr = dtl::is_power_of_two(instance.m)
                                 ? dtl::block_addressing::POWER_OF_TWO
                                 : dtl::block_addressing::MAGIC;
    switch (addr) {
      case dtl::block_addressing::POWER_OF_TWO: _o<b, t, s, dtl::block_addressing::POWER_OF_TWO>(instance, op); break;
      case dtl::block_addressing::MAGIC:        _o<b, t, s, dtl::block_addressing::MAGIC>(instance, op);        break;
      case dtl::block_addressing::DYNAMIC:      /* must not happen */                                           break;
    }
  }


//  template<u32 b, u32 t, u32 s, dtl::block_addressing a>
//  static void
//  _u(blocked_cuckoofilter& instance, op_t op) {
//    switch (unroll_factor(k, a, w)) {
//      case  0: _o<b, t, s, a,  0>(instance, op); break;
//      case  1: _o<b, t, s, a,  1>(instance, op); break;
//      case  2: _o<b, t, s, a,  2>(instance, op); break;
//      case  4: _o<b, t, s, a,  4>(instance, op); break;
////      case  8: _o<w, s, k, a,  8>(instance, op); break;
//      default:
//        throw std::invalid_argument("The given 'unroll_factor' is not supported.");
//    }
//  }


  template<u32 b, u32 t, u32 s, dtl::block_addressing a>
  static void
  _o(blocked_cuckoofilter& instance, op_t op) {
    using _t = bcf<b, t, s, a>;
    switch (op) {
      case op_t::CONSTRUCT: instance._construct_logic<_t>();           break;
//      case op_t::BIND:      instance._bind_logic<_t, unroll_factor>(); break;
      case op_t::BIND:      instance._bind_logic<_t>(); break;
      case op_t::DESTRUCT:  instance._destruct_logic<_t>();            break;
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Constructs a blocked filter logic instance.
  template<typename bcf_t>
  void
  _construct_logic() {
    // Instantiate a filter logic.
    bcf_t* bcf = new bcf_t(m);
    instance = bcf;
    block_size_bytes = bcf_t::block_size_bytes;
    tag_size_bits = bcf_t::tag_size_bits;
    associativity = bcf_t::associativity;

    // Get the actual size of the filter.
    m_actual = bcf->get_length();

    using adapter_t = internal::blocked_cuckoofilter_api_adapter<word_t, bcf_t>;
    adapter_t* adapter = new adapter_t { bcf };
    adapter_instance = adapter;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Bind API functions to the (previously constructed) filter logic.
  template<typename bcf_t>
  void
  _bind_logic() {
    using namespace std::placeholders;
    using adapter_t = internal::blocked_cuckoofilter_api_adapter<word_t, bcf_t>;
//    auto* bcf = static_cast<bcf_t*>(instance);
    auto* bcf_adapter = static_cast<adapter_t*>(adapter_instance);

    // Bind the API functions.
    insert = std::bind(&adapter_t::insert, bcf_adapter, _1, _2);
    batch_insert = std::bind(&adapter_t::batch_insert, bcf_adapter, _1, _2, _3);
    contains = std::bind(&adapter_t::contains, bcf_adapter, _1, _2);
    batch_contains = std::bind(&adapter_t::batch_contains, bcf_adapter, _1, _2, _3, _4, _5);

//    // Bind the API functions.
//    insert = std::bind(&bcf_t::insert, bcf, _1, _2);
////    batch_insert = std::bind(&bcf_t::batch_insert, bcf, _1, _2, _3);
//    contains = std::bind(&bcf_t::contains, bcf, _1, _2);
//
//    // SIMD vector length (0 = run scalar code)
//    static constexpr u64 vector_len = dtl::simd::lane_count<key_t>; // * unroll_factor;
////    batch_contains = std::bind(&bcf_t::template batch_contains<vector_len>, bcf, _1, _2, _3, _4, _5);
//    batch_contains = std::bind(&bcf_t::batch_contains, bcf, _1, _2, _3, _4, _5);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Destructs the filter logic.
  template<typename bcf_t>
  void
  _destruct_logic() {
    using adapter_t = internal::blocked_cuckoofilter_api_adapter<word_t, bcf_t>;
    adapter_t* bcf_adapter = static_cast<adapter_t*>(adapter_instance);
    delete bcf_adapter;
    adapter_instance = nullptr;

    bcf_t* bcf = static_cast<bcf_t*>(instance);
    delete bcf;
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
  /// Returns the name of the filter instance including the most
  /// important parameters (in JSON format).
  std::string
  name() const {
    return "{\"name\":\"blocked_cuckoo\", \"size\":" + std::to_string(size_in_bytes())
        + ",\"block_size\":" + std::to_string(block_size_bytes)
        + ",\"tag_bits\":" + std::to_string(tag_size_bits)
        + ",\"associativity\":" + std::to_string(associativity)
        + ",\"addr\":" + (get_addressing_mode() == dtl::block_addressing::POWER_OF_TWO ? "\"pow2\"" : "\"magic\"")
        + "}";
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Runs the calibration code. Results are stored in global variables.
  // TODO memoization in a global file / tool to calibrate
  static void
  calibrate() __attribute__ ((noinline)) {
    std::cerr << "Running calibration..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    static constexpr u32 data_size = 4u*1024*8;
    std::vector<key_t> random_data;
    random_data.reserve(data_size);
    for (std::size_t i = 0; i < data_size; i++) {
      random_data.push_back(dis(gen));
    }

    static const u32 max_unroll_factor = 4;
    for ($u32 w = 1; w <= 16; w *= 2) {
      for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
        for ($u32 k = 1; k <= 16; k++) {
          try {
            std::cerr << "w = " << std::setw(2) << w << ", "
                      << "addr = " << std::setw(5) << (addr_mode == block_addressing::POWER_OF_TWO ? "pow2" : "magic") << ", "
                      << "k = " <<  std::setw(2) << k << ": " << std::flush;

            $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
            $u32 u_min = 1;

            std::size_t match_count = 0;
            uint32_t match_pos[dtl::BATCH_SIZE];

            // baselines
            $f64 cycles_per_lookup_u0 = 0.0;
            $f64 cycles_per_lookup_u1 = 0.0;
            for ($u32 u = 0; u <= max_unroll_factor; u = (u == 0) ? 1 : u*2) {
              std::cerr << std::setw(2) << "u(" << std::to_string(u) + ") = "<< std::flush;
              unroll_factor(k, addr_mode, w) = u;
              $u32 sector_cnt = w;
              try {
                // with sectorization
                blocked_cuckoofilter bbf(data_size + 128 * static_cast<u32>(addr_mode), k, w, sector_cnt); // word_cnt = sector_cnt
              }
              catch (...) {
                // fall back to 1 sector
                sector_cnt = 1;
              }
              blocked_cuckoofilter bbf(data_size + 128 * static_cast<u32>(addr_mode), k, w, sector_cnt);
              std::vector<word_t, dtl::mem::numa_allocator<word_t>> filter_data(bbf.size(), 0);

              $u64 rep_cntr = 0;
              auto start = std::chrono::high_resolution_clock::now();
              auto tsc_start = _rdtsc();
              while (true) {
                std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
                if (diff.count() > 0.25) break;
                dtl::batch_wise(random_data.begin(), random_data.end(), [&](const auto batch_begin, const auto batch_end) {
                  match_count += bbf.batch_contains(&filter_data[0], &batch_begin[0], batch_end - batch_begin, match_pos, 0);
                });
                rep_cntr++;
              }
              auto tsc_end = _rdtsc();
              auto cycles_per_lookup = (tsc_end - tsc_start) / (data_size * rep_cntr * 1.0);
              if (u == 0) cycles_per_lookup_u0 = cycles_per_lookup;
              if (u == 1) cycles_per_lookup_u1 = cycles_per_lookup;
              std::cerr << std::setprecision(3) << std::setw(4) << std::right << cycles_per_lookup << ", ";
              if (cycles_per_lookup < cycles_per_lookup_min) {
                cycles_per_lookup_min = cycles_per_lookup;
                u_min = u;
              }
            }
            unroll_factor(k, addr_mode, w) = u_min;
            std::cerr << " picked u = " << unroll_factor(k, addr_mode, w)
                      << ", speedup over u(0) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_u0 / cycles_per_lookup_min)
                      << ", speedup over u(1) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_u1 / cycles_per_lookup_min)
                      << " (chksum: " << match_count << ")" << std::endl;

          } catch (...) {
            std::cerr<< " -> Failed to calibrate for k = " << k << "." << std::endl;
          }
        }
      }
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the SIMD unrolling factor for the given k and addressing mode.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  static $u32&
  unroll_factor(u32 k, dtl::block_addressing addr_mode, u32 word_cnt_per_block) {
    auto& unroll_factors = sizeof(word_t) == 8
                           ? internal::unroll_factors_64
                           : internal::unroll_factors_32;
    return unroll_factors[
        internal::max_k * dtl::log_2(word_cnt_per_block)
        + (k - 1)
        + (static_cast<u32>(addr_mode) * internal::max_k * 5)
    ];
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Force unroll factor for all implementations (used for benchmarking)
  static void
  force_unroll_factor(u32 u) {
    for ($u32 w = 1; w <= 16; w *= 2) {
      for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
        for ($u32 k = 1; k <= 16; k++) {
          unroll_factor(k, addr_mode, w) = u;
        }
      }
    }
  }
  //===----------------------------------------------------------------------===//

};



} // namespace dtl
