#pragma once

#include <iostream>

// CC4 licenced code
// https://howardhinnant.github.io/allocator_boilerplate.html

template<class T>
class TrackingAllocator {
 public:
  using value_type    = T;

  uint64_t& total_allocation_size;

  template<class U>
  struct rebind {
    typedef TrackingAllocator<U> other;
  };

  TrackingAllocator(uint64_t& total_allocation_size) noexcept
      : total_allocation_size(total_allocation_size) {}
  template<class U>
  TrackingAllocator(TrackingAllocator<U> const& other) noexcept
      : total_allocation_size(other.total_allocation_size) {}

  value_type*  // Use pointer if pointer is not a value_type*
  allocate(std::size_t n) {
    assert(n==1);
    total_allocation_size += sizeof(value_type);
    return static_cast<value_type*>(::operator new(n*sizeof(value_type)));
  }

  void deallocate(value_type* p,
                  std::size_t) noexcept  // Use pointer if pointer is not a value_type*
  {
    total_allocation_size -= sizeof(value_type);
    ::operator delete(p);
  }

  template<class U>
  void destroy(U* p) noexcept {
    p->~U();
  }
};
