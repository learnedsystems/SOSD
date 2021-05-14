#pragma once

#include <inttypes.h>

#include <array>
#include <cstring>

#include "base.h"

namespace interpolation_btree {

enum struct PageType : uint16_t {
  kInvalid,
  kBTreeLeaf,
  kBTreeInner,
};

struct BTreeNodeBase {  // -> 16 Byte
  PageType page_type;
  uint16_t count;
};
static_assert(sizeof(BTreeNodeBase) == 4, "");

template <class KeyType, uint32_t MAX_ENTRIES_TEMPLATES>
class BTreeGenericInner : public BTreeNodeBase {
 public:
  constexpr static uint32_t MAX_ENTRIES = MAX_ENTRIES_TEMPLATES;

  ~BTreeGenericInner() {
    for (uint32_t i = 0; i < count; i++) {
      delete children[i];
    }
  }

  explicit BTreeGenericInner() {
    count = 0;
    page_type = PageType::kBTreeInner;
  }

  double slope;
  std::array<KeyType, MAX_ENTRIES> keys;
  std::array<BTreeNodeBase*, MAX_ENTRIES + 1> children;

  void Init() {
    KeyType first = keys[0];
    KeyType last = keys[count - 1];
    slope = count * 1.0 / (last - first);
  }

  uint32_t LowerBound(KeyType key) {
    // Linear search
    //      uint32_t idx = 0;
    //      while (idx<count && keys[idx]<key) {
    //         idx++;
    //      }
    //      return idx;

    // Binary search
    //      auto result = std::lower_bound(keys.begin(), keys.begin() + count,
    //      key, [](KeyType lhs, KeyType rhs) {
    //         return lhs<rhs;
    //      });
    //      return std::distance(keys.begin(), result);

    // Interpolation search
    if (key <= keys[0]) {
      return 0;
    }
    if (key > keys[count - 1]) {
      return count;
    }

    // KeyType first = keys[0];
    uint32_t idx = slope * (key - keys[0]);
    if (idx >= count) {
      idx = count - 1;
    }
    while (idx > 0 && keys[idx] >= key) {
      idx--;
    }
    while (idx < count && keys[idx] < key) {
      idx++;
    }

    return idx;
  }
};

template <class KeyType, class ValueType, uint32_t MAX_ENTRIES_TEMPLATES>
class BTreeGenericLeaf : public BTreeNodeBase {
 public:
  constexpr static uint32_t MAX_ENTRIES = MAX_ENTRIES_TEMPLATES;

  explicit BTreeGenericLeaf() {
    count = 0;
    page_type = PageType::kBTreeLeaf;
  }

  double slope;
  std::array<KeyType, MAX_ENTRIES> keys;
  std::array<ValueType, MAX_ENTRIES> values;

  void Init() {
    KeyType first = keys[0];
    KeyType last = keys[count - 1];
    slope = count * 1.0 / (last - first);
  }

  uint32_t LowerBound(KeyType key) {
    // Linear search
    //      uint32_t idx = 0;
    //      while (idx<count && keys[idx]<key) {
    //         idx++;
    //      }
    //      if (idx == count) {
    //         return count - 1;
    //      }
    //      return idx;

    // Binary search
    //      auto result = std::lower_bound(keys.begin(), keys.begin() + count,
    //      key, [](KeyType lhs, KeyType rhs) {
    //         return lhs<rhs;
    //      });
    //      return std::distance(keys.begin(), result);

    // Interpolation search
    if (key <= keys[0]) {
      return 0;
    }
    if (key > keys[count - 1]) {
      return count;
    }

    KeyType first = keys[0];
    uint32_t idx = slope * (key - first);
    if (idx >= count) {
      idx = count - 1;
    }
    while (idx > 0 && keys[idx] >= key) {
      idx--;
    }
    while (idx < count && keys[idx] < key) {
      idx++;
    }
    if (idx == count) {
      return count - 1;
    }
    return idx;
  }
};

template <class Key, class Value>
class BTree {
 public:
  BTree();
  ~BTree();

  uint64_t GetSize() const;

  Value Lookup(Key k) const;

  uint64_t FastLoadGenerate(std::function<bool(Key&, Value&)> generator);

 private:
  using MMBTreeLeaf = BTreeGenericLeaf<Key, Value, 256>;
  using MMBTreeInner = BTreeGenericInner<Key, 256>;

  BTreeNodeBase* root;

  uint32_t height = 0;

  uint64_t GetSizeRec(const MMBTreeInner* node) const;
};

template <class Key, class Value>
BTree<Key, Value>::BTree() : root(nullptr) {
  height = 0;
}

template <class Key, class Value>
BTree<Key, Value>::~BTree() {
  delete root;
}

template <class Key, class Value>
uint64_t BTree<Key, Value>::GetSize() const {
  if (root && root->page_type == PageType::kBTreeInner) {
    return GetSizeRec(reinterpret_cast<const MMBTreeInner*>(root));
  }
  if (root && root->page_type == PageType::kBTreeLeaf) {
    return sizeof(MMBTreeLeaf);
  }
  return 0;
}

template <class Key, class Value>
uint64_t BTree<Key, Value>::GetSizeRec(const MMBTreeInner* node) const {
  uint64_t result = sizeof(MMBTreeInner);
  for (uint32_t i = 0; i <= node->count; i++) {
    if (node->children[i]->page_type == PageType::kBTreeInner) {
      result +=
          GetSizeRec(reinterpret_cast<const MMBTreeInner*>(node->children[i]));
    }
    if (node->children[i]->page_type == PageType::kBTreeLeaf) {
      result += sizeof(MMBTreeLeaf);
    }
  }
  return result;
}

template <class Key, class Value>
Value BTree<Key, Value>::Lookup(Key key) const {
  BTreeNodeBase* current = root;

  // Decent the tree till we find a leaf
  for (uint32_t i = 0; i < height; i++) {
    auto innerPtr = reinterpret_cast<MMBTreeInner*>(current);
    uint32_t pos = innerPtr->LowerBound(key);
    current = (BTreeNodeBase*)innerPtr->children[pos];
  }

  // Try to locate the key in the child node
  auto leaf = reinterpret_cast<MMBTreeLeaf*>(current);
  assert(current->page_type == PageType::kBTreeLeaf);
  uint32_t pos = leaf->LowerBound(key);
  return leaf->values[pos];
}

template <class Key, class Value>
uint64_t BTree<Key, Value>::FastLoadGenerate(
    std::function<bool(Key&, Value&)> generator) {
  height = 0;

  uint64_t loaded_tuples = 0;
  assert(MMBTreeInner::MAX_ENTRIES > 1);

  std::vector<std::pair<Key, BTreeNodeBase*>> nextLayerKeys;
  std::vector<std::pair<Key, BTreeNodeBase*>> currentLayerKeys;

  // Build leaf layer
  {
    // MMBTreeLeaf *prev_node = nullptr;

    bool input_good = true;
    while (input_good) {
      // Allocate leaf node
      MMBTreeLeaf* leaf = new MMBTreeLeaf();

      // Build one leaf node
      uint64_t pos = 0;
      while (pos < MMBTreeLeaf::MAX_ENTRIES && input_good) {
        input_good = generator(leaf->keys[pos], leaf->values[pos]);
        if (input_good) {
          pos++;
          loaded_tuples++;
        }
      }
      if (pos == 0) {
        delete leaf;
        assert(!input_good);
        if (nextLayerKeys.empty()) {
          // Empty file -> nothing to change at the tree
          assert(loaded_tuples == 0);
          return loaded_tuples;
        }
        break;
      }

      leaf->page_type = PageType::kBTreeLeaf;
      leaf->count = (uint16_t)pos;
      leaf->Init();
      nextLayerKeys.push_back(std::make_pair(leaf->keys[pos - 1], leaf));

      // prev_node = leaf;
    }
  }

  // Build layer by layer up to the root
  while (nextLayerKeys.size() > 1) {
    height++;
    nextLayerKeys.swap(currentLayerKeys);

    // Build one layer
    uint64_t pos = 0;
    uint64_t requiredInnerNodes =
        currentLayerKeys.size() / MMBTreeInner::MAX_ENTRIES +
        (currentLayerKeys.size() % MMBTreeInner::MAX_ENTRIES == 0 ? 0 : 1);
    uint64_t entriesPerInnerNode =
        currentLayerKeys.size() / requiredInnerNodes +
        (currentLayerKeys.size() % MMBTreeInner::MAX_ENTRIES == 0 ? 0 : 1);
    assert(entriesPerInnerNode > 1);
    for (uint32_t innerNode = 0; innerNode < requiredInnerNodes; innerNode++) {
      if (innerNode * entriesPerInnerNode +
              (requiredInnerNodes - innerNode) * (entriesPerInnerNode - 1) ==
          currentLayerKeys.size()) {
        entriesPerInnerNode--;
      }

      // Build one inner node
      uint64_t entriesInThisNode = entriesPerInnerNode;
      auto* inner = new MMBTreeInner();

      // Assign values
      inner->page_type = PageType::kBTreeInner;
      inner->count = (uint16_t)(entriesInThisNode - 1);
      assert(inner->count > 0);
      for (uint64_t i = 0; i < entriesInThisNode; ++i) {
        if (i + 1 != entriesInThisNode) {
          inner->keys[i] = currentLayerKeys[pos + i].first;
        }
        inner->children[i] = currentLayerKeys[pos + i].second;
      }
      inner->Init();
      nextLayerKeys.push_back(std::make_pair(
          currentLayerKeys[pos + entriesInThisNode - 1].first, inner));
      pos += entriesInThisNode;
    }

    currentLayerKeys.clear();
  }

  assert(nextLayerKeys.size() == 1);
  root = nextLayerKeys[0].second;

  return loaded_tuples;
}

}  // namespace interpolation_btree

template <class KeyType, uint32_t size_scale>
class InterpolationBTree : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    return util::timing([&] {
      uint64_t pos = 0;
      btree_.FastLoadGenerate([&](KeyType& key, uint64_t& value) {
        if (pos >= data.size()) {
          return false;
        }
        while (pos < data.size() && pos % size_scale != 0) {
          pos++;
        }
        if (pos == data.size()) {  // always insert the last element
          pos--;
        }

        key = data[pos].key;
        value = data[pos].value;
        pos++;
        return true;
      });
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    uint64_t ub = btree_.Lookup(lookup_key);
    uint64_t lower = ub < size_scale ? 0 : ub - size_scale;
    return SearchBound{lower, ub};
  }

  std::string name() const { return "IBTree"; }

  std::size_t size() const { return sizeof(*this) + btree_.GetSize(); }

  int variant() const { return size_scale; }

 private:
  interpolation_btree::BTree<KeyType, uint64_t> btree_;
};
