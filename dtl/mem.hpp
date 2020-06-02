#pragma once

// require NUMA support (for now)
#if !defined(HAVE_NUMA)
#define HAVE_NUMA
#endif

#include "adept.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <stdlib.h>
#include <sys/mman.h>
#if defined(HAVE_NUMA)
#include <numa.h>
#include <numaif.h>
#endif

namespace dtl {
namespace mem {

/// the size of a cache line (in bytes)
constexpr u64 cacheline_size = 64;

/// compile-time constant to determine whether NUMA support is available
constexpr u1 has_numa_support =
#if defined(HAVE_NUMA)
                                true;
#else
                                false;
#endif

/// Checks, whether a pointer is aligned (to n bytes).
template<typename T>
inline u1
is_aligned(const T* const ptr, u64 n = alignof(T)) {
  return (reinterpret_cast<uintptr_t>(ptr) % n) == 0 ;
}

namespace detail {

/// encapsulates a pointer to a numa bitmask (supports move and copy construction, etc.)
struct bitmask_wrap {
  /// the wrapped pointer
  bitmask* ptr = nullptr;

  /// default c'tor
  bitmask_wrap() : ptr(nullptr) { };
  /// c'tor that takes the ownership
  explicit bitmask_wrap(bitmask* ptr) : ptr(ptr) { };
  /// move c'tor
  bitmask_wrap(bitmask_wrap&& other) : ptr(other.ptr) { other.ptr = nullptr; };
  /// copy c'tor
  bitmask_wrap(const bitmask_wrap& other) {
    if (other.ptr) {
      ptr = numa_bitmask_alloc(other.ptr->size);
      copy_bitmask_to_bitmask(other.ptr, ptr);
    }
  };
  bitmask_wrap& operator=(const bitmask_wrap& other) {
    if (ptr) {
      numa_bitmask_free(ptr);
      ptr = nullptr;
    }
    if (other.ptr) {
      ptr = numa_bitmask_alloc(other.ptr->size);
      copy_bitmask_to_bitmask(other.ptr, ptr);
    }
    return *this;
  }
  bitmask_wrap& operator=(bitmask_wrap&& other) {
    if (ptr) {
      numa_bitmask_free(ptr);
      ptr = nullptr;
    }
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }
  /// d'tor
  ~bitmask_wrap() { if (ptr) numa_bitmask_free(ptr); };
};

inline bitmask_wrap
get_hbm_nodemask() {
  i32 node_cnt = numa_num_configured_nodes();
  bitmask* mask = numa_bitmask_alloc(node_cnt);
  numa_bitmask_setall(mask);

  i32 cpu_cnt = numa_num_configured_cpus();
  for ($i32 i = 0; i < cpu_cnt; i++) {
    numa_bitmask_clearbit(mask, numa_node_of_cpu(i));
  }
  return bitmask_wrap(mask);
}

inline bitmask_wrap
get_cpu_nodemask() {
  i32 node_cnt = numa_num_configured_nodes();
  bitmask* mask = numa_bitmask_alloc(node_cnt);
  numa_bitmask_clearall(mask);

  i32 cpu_cnt = numa_num_configured_cpus();
  for ($i32 i = 0; i < cpu_cnt; i++) {
    numa_bitmask_setbit(mask, numa_node_of_cpu(i));
  }
  return bitmask_wrap(mask);
}

inline bitmask_wrap
get_all_nodemask() {
  i32 node_cnt = numa_num_configured_nodes();
  bitmask* mask = numa_bitmask_alloc(node_cnt);
  numa_bitmask_setall(mask);
  return bitmask_wrap(mask);
}


} // namespace detail


/// determine the NUMA node ids
inline std::vector<$i32>
get_nodes() {
  std::vector<$i32> nodes;
#if defined(HAVE_NUMA)
  i32 node_cnt = numa_num_configured_nodes();
  for ($i32 i = 0; i < node_cnt; i++) {
    nodes.push_back(i);
  }
#else
  nodes.push_back(0);
#endif
  return nodes;
}


/// determine the number of NUMA nodes
inline i32
get_node_count() {
#if defined(HAVE_NUMA)
  return numa_num_configured_nodes();
#else
  return 1;
#endif
}


/// determine the NUMA node id of the given CPU id
inline i32
get_node_of_cpu(i32 cpu_id) {
#if defined(HAVE_NUMA)
  i32 node_id = numa_node_of_cpu(cpu_id);
  return (node_id < 0) ? 0 : node_id;
#else
  return 0;
#endif
}


/// determine the node ids of HBM nodes
inline std::vector<$i32>
get_cpu_nodes() {
  std::vector<$i32> cpu_nodes;
#if defined(HAVE_NUMA)
  i32 node_cnt = numa_num_configured_nodes();
  std::vector<$i32> cpus_on_node_cnt;
  cpus_on_node_cnt.resize(node_cnt, 0);

  i32 cpu_cnt = numa_num_configured_cpus();
  for ($i32 i = 0; i < cpu_cnt; i++) {
    cpus_on_node_cnt[numa_node_of_cpu(i)]++;
  }
  for ($i32 i = 0; i < node_cnt; i++) {
    if (cpus_on_node_cnt[i] > 0) {
      // we assume that all *memory-only* NUMA nodes are HBM nodes.
      cpu_nodes.push_back(i);
    }
  }
#endif
  return cpu_nodes;
}


/// determine the node ids of HBM nodes
inline std::vector<$i32>
get_hbm_nodes() {
  std::vector<$i32> hbm_nodes;
#if defined(HAVE_NUMA)
  i32 node_cnt = numa_num_configured_nodes();
  std::vector<$i32> cpus_on_node_cnt;
  cpus_on_node_cnt.resize(node_cnt, 0);

  i32 cpu_cnt = numa_num_configured_cpus();
  for ($i32 i = 0; i < cpu_cnt; i++) {
    cpus_on_node_cnt[numa_node_of_cpu(i)]++;
  }
  for ($i32 i = 0; i < node_cnt; i++) {
    if (cpus_on_node_cnt[i] == 0) {
      // we assume that all *memory-only* NUMA nodes are HBM nodes.
      hbm_nodes.push_back(i);
    }
  }
#endif
  return hbm_nodes;
}


/// determine whether the system has HBM nodes
inline u1
hbm_available() {
  return get_hbm_nodes().size() > 0;
}

/// determine the HBM node that is nearest to the given node
/// if HBM is not available, the given node id is returned.
inline i32
get_nearest_hbm_node(i32 numa_node_id) {
#if defined(HAVE_NUMA)
  if (!hbm_available()) return numa_node_id;

  $i32 min_distance = std::numeric_limits<$i32>::max();
  $i32 nearest_node = numa_node_id;
  for (auto hbm_node_id : get_hbm_nodes()) {
    i32 distance = numa_distance(numa_node_id, hbm_node_id);
    if (distance < min_distance) {
      min_distance = distance;
      nearest_node = hbm_node_id;
    }
  }
  return nearest_node;
#else
  return 0;
#endif
}

/// Determine the CPU node that is nearest to the given HBM node.
/// If HBM is not available, the given node ID is returned.
inline i32
get_nearest_cpu_node(i32 hbm_numa_node_id) {
#if defined(HAVE_NUMA)
  if (!hbm_available()) return hbm_numa_node_id;

  $i32 min_distance = std::numeric_limits<$i32>::max();
  $i32 nearest_node = hbm_numa_node_id;
  for (auto cpu_node_id : get_cpu_nodes()) {
    i32 distance = numa_distance(hbm_numa_node_id, cpu_node_id);
    if (distance < min_distance) {
      min_distance = distance;
      nearest_node = cpu_node_id;
    }
  }
  return nearest_node;
#else
  return 0;
#endif
}

inline i32
get_node_of_address(const void* addr) {
#if defined(HAVE_NUMA)
  void* ptr_to_check = const_cast<void*>(addr); // TODO align ptr to page boundary
  $i32 status[1];
  status[0] = -1;

  $i32 ret_code = move_pages(0, 1, &ptr_to_check, nullptr, status, 0);
  if (ret_code != 0) {
    throw std::invalid_argument("Failed to determine NUMA node for the given address.");
  }
  return status[0];
#else
  return 0;
#endif
}


/// the different memory allocation policies
enum class allocation_policy {
  /// thread local allocations (default)
  local,
  /// allows to specify a node where to allocate memory
  on_node,
  /// allows for interleaved allocation across specified nodes
  interleaved,
};


/// parameters for the NUMA allocator instance
class allocator_config {

  template<class T>
  friend class numa_allocator;

private:
  allocation_policy policy;
  detail::bitmask_wrap node_mask;
  $u32 numa_node;

  /// c'tor for interleaved allocation
  allocator_config(detail::bitmask_wrap node_mask)
      : policy(allocation_policy::interleaved), node_mask(node_mask), numa_node(~u32(0)) { }

  /// c'tor for allocations on a specific node
  allocator_config(u32 numa_node)
      : policy(allocation_policy::on_node), node_mask(nullptr), numa_node(numa_node) { }

  /// c'tor for local allocations
  allocator_config()
      : policy(allocation_policy::local), node_mask(nullptr), numa_node(~u32(0)) { }


public:
  /// move c'tor
  allocator_config(allocator_config&& src) = default;
  /// copy c'tor
  allocator_config(const allocator_config& src) = default;

  allocator_config& operator=(const allocator_config& other) = default;

  allocator_config& operator=(allocator_config&& other) = default;


  /// interleaved memory allocation (also includes HBM nodes)
  static allocator_config
  interleave_all() {
    return allocator_config(detail::get_all_nodemask());
  }


  /// interleaved memory allocations on nodes which contain CPUs.
  static allocator_config
  interleave_cpu() {
    return allocator_config(detail::get_cpu_nodemask());
  }

  /// interleaved memory allocations on HMB nodes (Note: we assume all NUMA nodes that do not contain any CPUs as HBM nodes)
  static allocator_config
  interleave_hbm() {
    if (dtl::mem::hbm_available()) {
      return allocator_config(detail::get_hbm_nodemask());
    }
    else {
      // fallback to 'interleave_all'
      return allocator_config(detail::get_all_nodemask());
    }
  }

  /// allocate memory on the specified node only
  static allocator_config
  on_node(u32 numa_node) {
    return allocator_config(numa_node);
  }

  /// allocate memory on the thread local node (default behavior)
  static allocator_config
  local() {
    return allocator_config();
  }

  void
  print(std::ostream& os) const {
    switch (policy) {
      case allocation_policy::interleaved:
        os << "interleaved on nodes ";
        for (std::size_t i = 0; i < node_mask.ptr->size; i++) {
          if (numa_bitmask_isbitset(node_mask.ptr, i)) {
            os << i << " ";
          }
        }
        break;
      case allocation_policy::on_node:
        os << "on node " << numa_node;
        break;
      case allocation_policy::local:
        os << "local";
        break;
    }
  }

};

// TODO alignment
template<typename T>
struct numa_allocator {

  using value_type = T;
  using pointer = value_type*;
  using size_type = std::size_t;

  const allocator_config config;


  /// c'tor
  numa_allocator() { }

  /// c'tor (with user specified parameters)
  numa_allocator(const allocator_config& config)
      : config(config) { }

  /// c'tor (with user specified parameters)
  numa_allocator(allocator_config&& config)
      : config(std::move(config)) { }

  /// copy c'tor
  template<class U>
  numa_allocator(const numa_allocator<U>& other)
      : config(other.config) { }

  ~numa_allocator() { }

  pointer
  allocate(size_type n, const void* /* hint */ = nullptr) throw() {
    void* ptr = nullptr;
    size_type size = n * sizeof(T);

    if (n > std::numeric_limits<size_type>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    switch (config.policy) {
      case allocation_policy::interleaved:
        ptr = numa_alloc_interleaved_subset(size, config.node_mask.ptr);
        break;
      case allocation_policy::on_node:
        ptr = numa_alloc_onnode(size, config.numa_node);
        break;
      case allocation_policy::local:
        ptr = numa_alloc_local(size);
        break;
    }

    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<pointer>(ptr);
  }

  void
  deallocate(pointer ptr, size_type n) throw() {
    numa_free(ptr, n * sizeof(T));
  }

};


} // namespace mem



  template<typename T>
  static T* aligned_alloc(u64 alignment, u64 cnt) {
    void* ptr = ::aligned_alloc(alignment, cnt * sizeof(T));
    return reinterpret_cast<T*>(ptr);
  }

  template<typename T>
  static T* aligned_alloc(u64 alignment, u64 cnt, u32 init_value) {
    void* ptr = aligned_alloc<T>(alignment, cnt * sizeof(T));
    std::memset(ptr, init_value, cnt * sizeof(T));
    return reinterpret_cast<T*>(ptr);
  }

  template<typename T>
  static T* malloc_huge(u64 n) {
    u64 huge_page_size = 2ull * 1024 * 1024;
    $u64 byte_cnt = n * sizeof(T);
    if (byte_cnt < huge_page_size) {
      void* p = malloc(byte_cnt);
      return reinterpret_cast<T*>(p);
    } else {
      byte_cnt = std::max(byte_cnt, huge_page_size);
      void* p = mmap(nullptr, byte_cnt, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      madvise(p, byte_cnt, MADV_HUGEPAGE);
      return reinterpret_cast<T*>(p);
    }
  }

  template<typename T>
  static void free_huge(T* ptr, const size_t n) {
    const uint64_t huge_page_size = 2ull * 1024 * 1024;
    uint64_t byte_cnt = n * sizeof(T);
    if (byte_cnt < huge_page_size) {
      free(ptr);
    } else {
      byte_cnt = std::max(byte_cnt, huge_page_size);
      munmap(ptr, byte_cnt);
    }
  }


} // namespace dtl

template<typename T, typename U>
inline u1
operator==(const dtl::mem::numa_allocator<T>&, const dtl::mem::numa_allocator<U>&) {
  // TODO not sure about the consequences here
  return true;
}

template<typename T, typename U>
inline u1
operator!=(const dtl::mem::numa_allocator<T>& a, const dtl::mem::numa_allocator<U>& b) {
  return !(a == b);
}

