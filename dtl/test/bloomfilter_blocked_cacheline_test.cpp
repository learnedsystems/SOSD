#include "gtest/gtest.h"

#include <algorithm>
#include <bitset>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/env.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>
#include <dtl/thread.hpp>

#include "helper.h"
#include "immintrin.h"

//#include "bloomfilter_util.hpp"

using namespace dtl;

namespace dtl {
namespace test {


// around 12-13% slower compared to 32-bit blocked version

struct bbf_cl {


  // --- helper functions ---
  __forceinline__
  static uint32_t
  extract_epi32(const __m256i a, uint32_t i) {
    assert(i >= 0);
    assert(i < 8);
    switch (i) {
      case 0: return _mm256_extract_epi32(a, 0);
      case 1: return _mm256_extract_epi32(a, 1);
      case 2: return _mm256_extract_epi32(a, 2);
      case 3: return _mm256_extract_epi32(a, 3);
      case 4: return _mm256_extract_epi32(a, 4);
      case 5: return _mm256_extract_epi32(a, 5);
      case 6: return _mm256_extract_epi32(a, 6);
      case 7: return _mm256_extract_epi32(a, 7);
    }
    assert(false);
    __builtin_unreachable();
  }


  /// [a,b,c,...] -> [a,a,b,b,c,c,...]
  __forceinline__
  static __m512i
  duplicate_epu32(const __m256i a) {
    // TODO: test if unpack is faster
    const __m512i b = _mm512_cvtepu32_epi64(a);
    return _mm512_or_si512(b, _mm512_slli_epi64(b, 32));
  }


  __forceinline__
  static uint64_t
  to_positions(const __mmask16 bitmask, uint32_t* positions, uint32_t offset) noexcept {
    if (bitmask == 0) return 0;
    static const __m512i sequence = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i seq = _mm512_add_epi64(sequence, _mm512_set1_epi32(offset));
    _mm512_mask_compressstoreu_epi32(positions, bitmask, seq);
    return dtl::bits::pop_count(static_cast<u32>(bitmask));
  }


  // --- ---------------- ---

  using key_t = uint32_t;
  using word_t = uint64_t;
  using hash_t = uint64_t;

//  static constexpr u32 PRIME0 = 2654435769u; // Knuth
//  static constexpr u32 PRIME1 = 1799596469u;
  static constexpr u32 PRIME0 = 596572387u; // Peter
  static constexpr u32 PRIME1 = 370248451u;
  static constexpr u64 PRIMES = (static_cast<u64>(PRIME1) << 32) | PRIME0 ;

  static constexpr uint32_t w = 64;
  static constexpr uint32_t B = 512;
  static constexpr uint32_t k = B / w;


  std::vector<__m512i, dtl::mem::numa_allocator<__m512i>> bit_vec;

  // block index = hash value >> block_bits_shift
  const std::size_t block_bits_shift;


  bbf_cl(const std::size_t m, dtl::mem::numa_allocator<__m512i> alloc)
    : bit_vec(dtl::next_power_of_two(m) / B, _mm512_setzero_si512(), alloc),
      block_bits_shift(w - dtl::log_2(dtl::next_power_of_two(m) / B)) {
    assert(dtl::next_power_of_two(m) / B > 0);
    std::cout << "block cnt: " << bit_vec.size() << std::endl;
    std::cout << "block bits shifts: " << block_bits_shift << std::endl;
  }

  bbf_cl(const std::size_t m) : bbf_cl(m, dtl::mem::numa_allocator<__m512i>()) {};

  __forceinline__
  void
  insert(const uint32_t key) noexcept {
    const uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(key * PRIME1)) << 32) | static_cast<uint32_t>(key * PRIME0);
    // identify the block
    const auto block_id = h >> block_bits_shift;
    // broadcast the hash value to all SIMD lanes
    const __m512i h_v = _mm512_set1_epi64(h);
    // extract the corresponding hash-bits in each lane
    const __m512i shift_cnt_v = _mm512_set_epi64(7*6,6*6,5*6,4*6,3*6,2*6,1*6,0*6);
    const __m512i mask_v =  _mm512_set1_epi64(0b111111);
    const __m512i bit_id_v = _mm512_and_epi64(_mm512_srlv_epi64(h_v, shift_cnt_v), mask_v);
    // set the corresponding bits (one bit per lane)
    const __m512i bits = _mm512_sllv_epi64(_mm512_set1_epi64(1), bit_id_v);
    // update the block
    bit_vec[block_id] = _mm512_or_si512(bit_vec[block_id], bits);
  }


  __forceinline__
  bool
  contains(const uint32_t key) const noexcept {
    const uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(key * PRIME1)) << 32) | static_cast<uint32_t>(key * PRIME0);
    // identify the block
    const auto block_id = h >> block_bits_shift;
    // broadcast the hash value to all SIMD lanes
    const __m512i h_v = _mm512_set1_epi64(h);
    // test the bits within the given block
    const __m512i block = _mm512_load_si512(&bit_vec[block_id]);
    const __mmask8 m = block_contains(block, h_v);
    return m == 0;
  }


  // 'hash_value_vec' must contain a single 64-bit hash value, replicated to all SIMD lane
  // the element is contained iff the returned value m == 0
  __forceinline__
  __mmask8
  block_contains(const __m512i block, const __m512i hash_value_vec) const noexcept {
    // extract the corresponding hash-bits in each lane
    const __m512i shift_cnts = _mm512_set_epi64(7*6,6*6,5*6,4*6,3*6,2*6,1*6,0*6);
    const __m512i block_mask =  _mm512_set1_epi64(0b111111);
    const __m512i bit_ids = _mm512_and_epi64(_mm512_srlv_epi64(hash_value_vec, shift_cnts), block_mask);
    // set the corresponding bits (one bit per lane)
    const __m512i search_mask = _mm512_sllv_epi64(_mm512_set1_epi64(1), bit_ids);
    // test the bits within the given block
    const __m512i a = _mm512_and_epi64(block, search_mask);
    const __mmask8 m = _mm512_cmpneq_epu64_mask(a, search_mask);
    return m;
  }


  /// contains check for eight 32-bit keys
  __mmask8
  contains(const __m256i keys_epi32) const noexcept __attribute__ ((hot,flatten,optimize("unroll-loops"),noinline)) {
    const std::size_t key_cnt = 8;
    // compute 64 hash bits using multiplicative hashing with two different 32-bit primes
    const __m512i keys = duplicate_epu32(keys_epi32); // [a,b,c,...] -> [a,a,b,b,c,c,...]
    const __m512i hash_values = _mm512_mullo_epi32(keys, _mm512_set1_epi64(PRIMES));

    // identify the blocks
    const __m512i block_ids_u64 = _mm512_srlv_epi64(hash_values, _mm512_set1_epi64(block_bits_shift));
    const __m256i block_ids = _mm512_cvtepi64_epi32(block_ids_u64);

    // prefetch blocks
    // Note: slli a YMM is slow on KNL
    const auto base_addr = const_cast<void*>(reinterpret_cast<const void*>(&bit_vec[0]));
    _mm512_mask_prefetch_i64gather_pd(_mm512_slli_epi64(block_ids_u64, 3), __mmask8(~0), base_addr, 8, _MM_HINT_T0);

    // extract the block id of the i-th key
    const uint32_t block_id_0 = extract_epi32(block_ids, 0);
    const uint32_t block_id_1 = extract_epi32(block_ids, 1);
    const uint32_t block_id_2 = extract_epi32(block_ids, 2);
    const uint32_t block_id_3 = extract_epi32(block_ids, 3);
    const uint32_t block_id_4 = extract_epi32(block_ids, 4);
    const uint32_t block_id_5 = extract_epi32(block_ids, 5);
    const uint32_t block_id_6 = extract_epi32(block_ids, 6);
    const uint32_t block_id_7 = extract_epi32(block_ids, 7);
    // replicate the hash value of the i-th key to all lanes
    const __m512i h_0 = _mm512_permutexvar_epi64(_mm512_set1_epi64(0), hash_values);
    const __m512i h_1 = _mm512_permutexvar_epi64(_mm512_set1_epi64(1), hash_values);
    const __m512i h_2 = _mm512_permutexvar_epi64(_mm512_set1_epi64(2), hash_values);
    const __m512i h_3 = _mm512_permutexvar_epi64(_mm512_set1_epi64(3), hash_values);
    const __m512i h_4 = _mm512_permutexvar_epi64(_mm512_set1_epi64(4), hash_values);
    const __m512i h_5 = _mm512_permutexvar_epi64(_mm512_set1_epi64(5), hash_values);
    const __m512i h_6 = _mm512_permutexvar_epi64(_mm512_set1_epi64(6), hash_values);
    const __m512i h_7 = _mm512_permutexvar_epi64(_mm512_set1_epi64(7), hash_values);
    // load the individual blocks
    const __m512i b_0 = _mm512_load_si512(&bit_vec[block_id_0]);
    const __m512i b_1 = _mm512_load_si512(&bit_vec[block_id_1]);
    const __m512i b_2 = _mm512_load_si512(&bit_vec[block_id_2]);
    const __m512i b_3 = _mm512_load_si512(&bit_vec[block_id_3]);
    const __m512i b_4 = _mm512_load_si512(&bit_vec[block_id_4]);
    const __m512i b_5 = _mm512_load_si512(&bit_vec[block_id_5]);
    const __m512i b_6 = _mm512_load_si512(&bit_vec[block_id_6]);
    const __m512i b_7 = _mm512_load_si512(&bit_vec[block_id_7]);

    const __mmask8 r_0 = block_contains(b_0, h_0);
    const __mmask8 r_1 = block_contains(b_1, h_1);
    const __mmask8 r_2 = block_contains(b_2, h_2);
    const __mmask8 r_3 = block_contains(b_3, h_3);
    const __mmask8 r_4 = block_contains(b_4, h_4);
    const __mmask8 r_5 = block_contains(b_5, h_5);
    const __mmask8 r_6 = block_contains(b_6, h_6);
    const __mmask8 r_7 = block_contains(b_7, h_7);
    // move the returned mask in a ZMM register
    __m512i interm_results;
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 0), _mm512_broadcastmb_epi64(r_0));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 1), _mm512_broadcastmb_epi64(r_1));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 2), _mm512_broadcastmb_epi64(r_2));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 3), _mm512_broadcastmb_epi64(r_3));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 4), _mm512_broadcastmb_epi64(r_4));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 5), _mm512_broadcastmb_epi64(r_5));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 6), _mm512_broadcastmb_epi64(r_6));
    interm_results = _mm512_mask_mov_epi64(interm_results, __mmask8(1 << 7), _mm512_broadcastmb_epi64(r_7));
    // return the results of the eight contains checks as an 8-bit mask
    return _mm512_cmpeq_epi64_mask(interm_results, _mm512_setzero_si512());
  }


  /// Performs a batch-probe
  __forceinline__
  uint64_t
  batch_contains(const uint32_t* keys, const uint32_t key_cnt, uint32_t* match_positions, const uint32_t match_offset) const noexcept {
    const uint32_t* reader = keys;
    uint32_t* match_writer = match_positions;

    // determine the number of keys that need to be probed sequentially, due to alignment
    const uint64_t required_alignment_bytes = 32;
    const uint64_t unaligned_key_cnt = dtl::mem::is_aligned(reader)
                                       ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t)
                                       : key_cnt;
    // process the unaligned keys sequentially
    uint64_t read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      bool is_match = contains(*reader);
      *match_writer = static_cast<uint32_t>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // process the aligned keys vectorized
    const uint64_t vector_len = 8;
    using vec_t = __m256i;
    using mask_t = __mmask8;
    const uint64_t aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      assert(dtl::mem::is_aligned(reader, 32));
      const mask_t mask = contains(*reinterpret_cast<const vec_t*>(reader));
      const uint64_t match_cnt = to_positions(mask, match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      bool is_match = contains(*reader);
      *match_writer = static_cast<uint32_t>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }


  void
  print() const noexcept {
    std::for_each(bit_vec.begin(), bit_vec.end(),
                  [](__m512i b) {
                    uint64_t* words = reinterpret_cast<uint64_t*>(&b);
                    for (std::size_t i = 0; i < 512/64; i++) {
                      std::cout << reinterpret_cast<std::bitset<64>&>(words[i]) << std::endl;
                    }
                  });
  }

};


using bf_t = bbf_cl;

TEST(bloomfilter_blocked_cacheline, foo) {
  bf_t bf(1024);
  bf.insert(42);
  bf.print();
  ASSERT_TRUE(bf.contains(42));
}


TEST(bloomfilter_blocked_cacheline, contained) {
  bf_t bf(1024);

  alignas(32) std::array<uint32_t, 8> in {1,2,3,4,5,6,7,8};
  alignas(32) std::array<uint32_t, 8> q {1,2,3,4,5,6,7,8};

  std::for_each(in.begin(), in.end(), [&](auto i) { bf.insert(i); });
  bf.print();

  auto r = bf.contains(reinterpret_cast<__m256i&>(q));
  ASSERT_EQ(0b11111111, r) << "expected: 11111111, got: " << std::bitset<8>(r);
}


TEST(bloomfilter_blocked_cacheline, output_mask) {
  bf_t bf(1024);

  alignas(32) std::array<uint32_t, 8> in {0,0,0,0,0,0,0,0};
  alignas(32) std::array<uint32_t, 8> q {0,0,0,0,0,0,0,0};

  std::for_each(in.begin(), in.end(), [&](auto i) { bf.insert(i); });

  for (std::size_t i = 0; i < 1<<8; i++) {
    std::bitset<8> m(i);
    for (std::size_t j = 0; j < 8; j++) { q[j] = m[j] ? 0 : 100; }
    auto r = bf.contains(reinterpret_cast<__m256i&>(q));
    ASSERT_TRUE(m == std::bitset<8>(r)) << "expected: " << m << ", got: " << std::bitset<8>(r);
  }
}


TEST(bloomfilter_blocked_cacheline, random) {
  bf_t bf(1<<20);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max());

  alignas(32) std::array<uint32_t, 1<<19> keys;
  std::generate(keys.begin(), keys.end(), [&]() { return dis(gen); });

  std::for_each(keys.begin(), keys.end(), [&](auto key){
    bf.insert(key);
    ASSERT_TRUE(bf.contains(key));
  });

  const __mmask8 exp = __mmask8(~0);
  for (std::size_t i = 0; i < keys.size(); i += 8) {
    auto act = bf.contains(reinterpret_cast<__m256i&>(keys[i]));
    ASSERT_EQ(exp, act) << "expected: " << std::bitset<8>(exp) << ", got: " << std::bitset<8>(act);
  }
}


// --- helpers ---

inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();


// --- runtime settings ---

// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// set the bloomfilter size: m in [2^lo, 2^hi]
static i32 bf_size_lo_exp = dtl::env<$i32>::get("BF_SIZE_LO", 11);
static i32 bf_size_hi_exp = dtl::env<$i32>::get("BF_SIZE_HI", 31);

// the number of hash functions to use
static i32 bf_k = dtl::env<$i32>::get("BF_K", 1);

// repeats the benchmark with different concurrency settings
static i32 thread_cnt_lo = dtl::env<$i32>::get("THREAD_CNT_LO", 1);
static i32 thread_cnt_hi = dtl::env<$i32>::get("THREAD_CNT_HI", std::thread::hardware_concurrency());

// 1 = linear, 2 = exponential
static i32 thread_step_mode = dtl::env<$i32>::get("THREAD_STEP_MODE", 1);
static i32 thread_step = dtl::env<$i32>::get("THREAD_STEP", 1);

// the number of keys to probe per thread
static u64 key_cnt_per_thread = 1ull << dtl::env<$i32>::get("KEY_CNT", 24);

// the number of repetitions
static u64 repeat_cnt = dtl::env<$i32>::get("REPEAT_CNT", 16);;


// place bloomfilter in HBM?
static u1 use_hbm = dtl::env<$i32>::get("HBM", 1);
// replicate bloomfilter in HBM?
static u1 replicate_bloomfilter = dtl::env<$i32>::get("REPL", 1);

static void
print_env_settings() {
  std::cout
      << "Configuration:\n"
      << "  BF_K=" << bf_k
      << ", BF_SIZE_LO=" << bf_size_lo_exp
      << ", BF_SIZE_HI=" << bf_size_hi_exp
      << "\n  GRAIN_SIZE=" << dtl::env<$i32>::get("GRAIN_SIZE", 16)
      << ", THREAD_CNT_LO=" << thread_cnt_lo
      << ", THREAD_CNT_HI=" << thread_cnt_hi
      << ", THREAD_STEP=" << thread_step
      << ", THREAD_STEP_MODE=" << thread_step_mode << " (1=linear, 2=exponential)"
      << ", KEY_CNT=" << dtl::env<$i32>::get("KEY_CNT", 24) << " (per thread)"
      << ", REPEAT_CNT=" << repeat_cnt
      << "\n  HBM=" << static_cast<u32>(use_hbm) << " (0=no, 1=yes)"
      << ", REPL=" << static_cast<u32>(replicate_bloomfilter) << " (0=interleaved, 1=replicate)"
      << std::endl;
}

static auto inc_thread_cnt = [&](u64 i) {
  if (thread_step_mode == 1) {
    // linear
    return i + thread_step;
  }
  else {
    // exponential
    auto step = thread_step > 1 ? thread_step : 2;
    return i * step;
  }
};




static void
run_filter_benchmark_in_parallel_vec(u32 k, u32 m, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;

  using key_alloc = dtl::mem::numa_allocator<uint32_t>;
  using block_alloc = dtl::mem::numa_allocator<__m512i>;
  block_alloc bf_cpu_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  block_alloc bf_hbm_interleaved_alloc(dtl::mem::allocator_config::interleave_hbm());

//  if (use_hbm) {
//    std::cout << "Using HBM for bloomfilter" << std::endl;
//  }
  // TODO  bf_rts_t bf(bf_size, use_hbm ? bf_hbm_interleaved_alloc : bf_cpu_interleaved_alloc);
  bf_t bf(m, bf_cpu_interleaved_alloc);

  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < m >> 4; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (key_cnt) / (duration / nano_to_sec);
  }

  // prepare the input (interleaved)
  key_alloc input_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  std::vector<uint32_t, key_alloc> keys(input_interleaved_alloc);
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // create replicas as requested (see env 'HBM' and 'HBM_REPL')
  std::vector<bf_t> bloomfilter_replicas;
  // maps node_id -> replica_id
  std::vector<$u64> bloomfilter_node_map;
  // insert the already existing bloomfilter (as a fallback when numa/hbm is not available)
  bloomfilter_replicas.push_back(std::move(bf));
  // initially, let all nodes refer to the first replica
  bloomfilter_node_map.resize(dtl::mem::get_node_count(), 0);

//  bloomfilter_replicas[0].print_info();

  if (replicate_bloomfilter) {
    // replicate the bloomfilter to all HBM nodes
    auto replica_nodes = (use_hbm && dtl::mem::hbm_available())
                         ? dtl::mem::get_hbm_nodes()
                         : dtl::mem::get_cpu_nodes();

    for (auto dst_node_id : replica_nodes) {
      // make a copy
      std::cout << "replicate bloomfilter to node " << dst_node_id << std::endl;
      auto alloc_config = dtl::mem::allocator_config::on_node(dst_node_id);
      block_alloc alloc(alloc_config);
      bloomfilter_replicas.emplace_back(m, alloc);

      auto& src = bloomfilter_replicas[0];
      auto& replica = bloomfilter_replicas.back();
      for (std::size_t i = 0; i < src.bit_vec.size(); i++) {
        replica.bit_vec[i] = src.bit_vec[i];
      }
      // update mapping
      bloomfilter_node_map[dst_node_id] = bloomfilter_replicas.size() - 1;
    }
  }

  // size of a work item (dispatched to a thread)
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$f64> avg_cycles_per_probe;
  avg_cycles_per_probe.resize(thread_cnt, 0.0);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  auto worker_fn = [&](u32 thread_id) {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
    const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);

    // obtain the local bloomfilter instance
    const bf_t& _bf = bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]];

    // allocate a match vector
    std::vector<$u32> match_pos;
    match_pos.resize(grain_size, 0);

    $u64 tsc = 0;
    $u64 found = 0;
    $u64 probe_cntr = 0;
    while (true) {
      u64 cntr = grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % key_cnt;
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (cntr >= key_cnt * repeat_cnt) break;
      u64 cnt = read_to - read_from;
      probe_cntr += cnt;
      __sync_synchronize();
      u64 tsc_begin = _rdtsc();
      auto match_cnt = _bf.batch_contains(&keys[read_from], cnt, &match_pos[0], 0);
      __sync_synchronize();
      u64 tsc_end = _rdtsc();
      tsc += tsc_end - tsc_begin;
      found += match_cnt;
    }
    matches_found[thread_id] = found;
    avg_cycles_per_probe[thread_id] = (tsc * 1.0) / probe_cntr;
  };


  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
  });

  duration /= repeat_cnt;

  $u64 found = 0;
  for ($u64 i = 0; i < thread_cnt; i++) {
    found += matches_found[i];
  }
  found /= repeat_cnt;

  f64 cycles_per_probe = std::accumulate(avg_cycles_per_probe.begin(), avg_cycles_per_probe.end(), 0.0) / thread_cnt;

  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (m / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s], "
            << "cycles/probe: " << cycles_per_probe
            << " (matchcnt: " << found << ")"
            << std::endl;
}


TEST(bloomfilter_blocked_cacheline, filter_performance_parallel_vec) {
  print_env_settings();
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 1) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel_vec(bf_k, bf_size, t);
    }
  }
}


} // namespace test
} // namespace dtl