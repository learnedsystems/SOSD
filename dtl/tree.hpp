#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <bitset>

namespace dtl {

  template<u64 N>
  class binary_tree_structure {
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");

  public:

    static constexpr u64 max_node_cnt = 2 * N - 1;
    static constexpr u64 height = ct::log_2<N>::value;

  private:

    std::bitset<max_node_cnt> _is_inner_node;

  public:

    binary_tree_structure() {
      // initialize a complete binary tree
      // ... all the inner nodes have two children
      for ($u64 i = 0; i < max_node_cnt / 2; i++) {
        _is_inner_node[i] = true;
      }
    }

    static inline u64
    parent_of(u64 node_idx) {
      return (node_idx - 1) / 2;
    }

    static inline u64
    left_child_of(u64 node_idx) {
      return 2 * node_idx + 1;
    }

    static inline u64
    right_child_of(u64 node_idx) {
      return 2 * node_idx + 2;
    }

    static inline u64
    level_of(u64 node_idx) {
      return log_2(node_idx + 1);
    }

    inline u1
    is_inner_node(u64 node_idx) const {
      return _is_inner_node[node_idx];
    }

    inline u1
    is_leaf_node(u64 node_idx) const {
      return ! _is_inner_node[node_idx];
    }

    inline void
    set_leaf(u64 node_idx) {
      _is_inner_node[node_idx] = false;
    }

    inline void
    set_inner(u64 node_idx) {
      _is_inner_node[node_idx] = true;
    }

    void
    dot() const {
      std::function<void(u64)> dot_recursively = [&](u64 idx) {
        using tree_t = decltype(*this);
        u1 is_inner = is_inner_node(idx);
        std::cout << "n" << idx << "[label=\"foo" << "\",style=filled,color=\"" << (true ? "#abd600" : "#b50000") << "\",width=0.2]" <<std::endl;
//          std::cout << "n" << idx << std::endl;

        u64 left_child_idx = left_child_of(idx);
        if (left_child_idx < max_node_cnt) {
          std::cout << "n" << idx << " -- n" << left_child_idx << std::endl; // << "[weight=0]"
          dot_recursively(left_child_idx);
        }

        u64 right_child_idx = right_child_of(idx);
        if (right_child_idx < max_node_cnt) {
          std::cout << "n" << idx << " -- n" << right_child_idx << std::endl;
          dot_recursively(right_child_idx);
        }
      };
      std::cout << "graph \"\" {" << std::endl;
      dot_recursively(0);
      std::cout << "}" << std::endl;
    }

    inline void
    print(std::ostream& os) const {
      os << _is_inner_node;
    }

  };



  template<u64 N>
  using full_binary_tree = binary_tree_structure<N>;


} // namespace dtl
