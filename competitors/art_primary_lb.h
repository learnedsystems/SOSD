#pragma once

#include <assert.h>
#include <emmintrin.h>  // x86 SSE intrinsics
#include <stdint.h>     // integer types
#include <stdio.h>
#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <sys/time.h>  // gettime

#include <algorithm>  // std::random_shuffle
#include <map>
#include <utility>
#include <vector>

#include "../util.h"
#include "base.h"

// Contains a version of ART that supports lower & upper bound lookups.

// Uses ART as a non-clustered primary index that stores <key, offset> pairs.
// At lookup time, we retrieve a key's offset from ART and lookup its value in
// the data array.
template <int size_scale>
class ARTPrimaryLB : public Competitor {
 public:
  ~ARTPrimaryLB() { destructTree(tree_); }

  uint64_t Build(const std::vector<KeyValue<uint64_t>>& data) {
    allocated_byte_count = 0;
    data_ = &data;
    data_size_ = data.size();

    uint64_t last_key = std::numeric_limits<uint64_t>::max();
    return util::timing([&] {
      for (uint64_t i = 0; i < data.size(); ++i) {
        if (size_scale > 1 && i % size_scale != 0) continue;

        const uint64_t data_key = data[i].key;

        if (data_key == last_key) continue;

        uint8_t key[8];
        swapBytes(data_key, key);
        insert(tree_, &tree_, key, 0, i, 8);
        last_key = data_key;
      }
    });
  }

  SearchBound EqualityLookup(const uint64_t lookup_key) {
    uint8_t key[8];
    swapBytes(lookup_key, key);

    // Lower bound lookup.
    Iterator it;
    const bool found = bound(tree_, key, 8, it, 8, true);

    const uint64_t guess = found ? it.value : data_size_ - 1;
    const uint64_t error = size_scale - 1;

    const uint64_t start = guess < error ? 0 : guess - error;
    const uint64_t stop = guess + 1 > data_size_
                              ? data_size_
                              : guess + 1;  // stop is exclusive (that's why +1)

    return (SearchBound){start, stop};
  }

  std::string name() const { return "ART"; }

  std::size_t size() const { return sizeof(*this) + allocated_byte_count; }

  int variant() const { return size_scale; }

  bool applicable(bool unique, const std::string& _data_filename) const {
    return unique;
  }

 private:
  static uint64_t allocated_byte_count;  // track bytes allocated

  /*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de
 */

  // Constants for the node types
  static const int8_t NodeType4 = 0;
  static const int8_t NodeType16 = 1;
  static const int8_t NodeType48 = 2;
  static const int8_t NodeType256 = 3;

  // The maximum prefix length for compressed paths stored in the
  // header, if the path is longer it is loaded from the database on
  // demand
  static const unsigned maxPrefixLength = 9;

  // Shared header of all inner nodes
  struct Node {
    // length of the compressed path (prefix)
    uint32_t prefixLength;
    // number of non-null children
    uint16_t count;
    // node type
    int8_t type;
    // compressed path (prefix)
    uint8_t prefix[maxPrefixLength];

    Node(int8_t type) : prefixLength(0), count(0), type(type) {}

    ~Node() {
      switch (type) {
        case NodeType4: {
          allocated_byte_count -= sizeof(Node4);
          break;
        }
        case NodeType16: {
          allocated_byte_count -= sizeof(Node16);
          break;
        }
        case NodeType48: {
          allocated_byte_count -= sizeof(Node48);
          break;
        }
        case NodeType256: {
          allocated_byte_count -= sizeof(Node256);
          break;
        }
      }
    }
  };

  // Node with up to 4 children
  struct Node4 : Node {
    uint8_t key[4];
    Node* child[4];

    Node4() : Node(NodeType4) {
      memset(key, 0, sizeof(key));
      memset(child, 0, sizeof(child));
      allocated_byte_count += sizeof(*this);
    }
  };

  // Node with up to 16 children
  struct Node16 : Node {
    uint8_t key[16];
    Node* child[16];

    Node16() : Node(NodeType16) {
      memset(key, 0, sizeof(key));
      memset(child, 0, sizeof(child));
      allocated_byte_count += sizeof(*this);
    }
  };

  static const uint8_t emptyMarker = 48;

  // Node with up to 48 children
  struct Node48 : Node {
    uint8_t childIndex[256];
    Node* child[48];

    Node48() : Node(NodeType48) {
      memset(childIndex, emptyMarker, sizeof(childIndex));
      memset(child, 0, sizeof(child));
      allocated_byte_count += sizeof(*this);
    }
  };

  // Node with up to 256 children
  struct Node256 : Node {
    Node* child[256];

    Node256() : Node(NodeType256) {
      memset(child, 0, sizeof(child));
      allocated_byte_count += sizeof(*this);
    }
  };

  inline Node* makeLeaf(uintptr_t tid) {
    // Create a pseudo-leaf
    return reinterpret_cast<Node*>((tid << 1) | 1);
  }

  inline bool isLeaf(Node* node) {
    // Is the node a leaf?
    return reinterpret_cast<uintptr_t>(node) & 1;
  }

  inline uintptr_t getLeafValue(Node* node) {
    // The the value stored in the pseudo-leaf
    return reinterpret_cast<uintptr_t>(node) >> 1;
  }

  struct IteratorEntry {
    Node* node;
    int pos;
  };

  struct Iterator {
    /// The current value, valid if depth>0
    uint64_t value;
    /// The current depth
    uint32_t depth;
    /// Stack, actually the size is determined at runtime
    IteratorEntry stack[9];
  };

  static inline unsigned ctz(uint16_t x) {
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n = 1;
    if ((x & 0xFF) == 0) {
      n += 8;
      x = x >> 8;
    }
    if ((x & 0x0F) == 0) {
      n += 4;
      x = x >> 4;
    }
    if ((x & 0x03) == 0) {
      n += 2;
      x = x >> 2;
    }
    return n - (x & 1);
#endif
  }

  // This address is used to communicate that search failed
  Node* nullNode = NULL;

  Node** findChild(Node* n, uint8_t keyByte) {
    // Find the next child for the keyByte
    switch (n->type) {
      case NodeType4: {
        Node4* node = static_cast<Node4*>(n);
        for (unsigned i = 0; i < node->count; i++)
          if (node->key[i] == keyByte) return &node->child[i];
        return &nullNode;
      }
      case NodeType16: {
        Node16* node = static_cast<Node16*>(n);
        __m128i cmp = _mm_cmpeq_epi8(
            _mm_set1_epi8(keyByte),
            _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
        unsigned bitfield = _mm_movemask_epi8(cmp) & ((1 << node->count) - 1);
        if (bitfield)
          return &node->child[ctz(bitfield)];
        else
          return &nullNode;
      }
      case NodeType48: {
        Node48* node = static_cast<Node48*>(n);
        if (node->childIndex[keyByte] != emptyMarker)
          return &node->child[node->childIndex[keyByte]];
        else
          return &nullNode;
      }
      case NodeType256: {
        Node256* node = static_cast<Node256*>(n);
        return &(node->child[keyByte]);
      }
    }
    throw;  // Unreachable
  }

  bool iteratorNext(Iterator& iter) {
    // Skip leaf
    if ((iter.depth) && (isLeaf(iter.stack[iter.depth - 1].node))) iter.depth--;

    // Look for next leaf
    while (iter.depth) {
      Node* node = iter.stack[iter.depth - 1].node;

      // Leaf found
      if (isLeaf(node)) {
        iter.value = getLeafValue(node);
        return true;
      }

      // Find next node
      Node* next = nullptr;
      switch (node->type) {
        case NodeType4: {
          Node4* n = static_cast<Node4*>(node);
          if (iter.stack[iter.depth - 1].pos < node->count)
            next = n->child[iter.stack[iter.depth - 1].pos++];
          break;
        }
        case NodeType16: {
          Node16* n = static_cast<Node16*>(node);
          if (iter.stack[iter.depth - 1].pos < node->count)
            next = n->child[iter.stack[iter.depth - 1].pos++];
          break;
        }
        case NodeType48: {
          Node48* n = static_cast<Node48*>(node);
          unsigned depth = iter.depth - 1;
          for (; iter.stack[depth].pos < 256; iter.stack[depth].pos++)
            if (n->childIndex[iter.stack[depth].pos] != emptyMarker) {
              next = n->child[n->childIndex[iter.stack[depth].pos++]];
              break;
            }
          break;
        }
        case NodeType256: {
          Node256* n = static_cast<Node256*>(node);
          unsigned depth = iter.depth - 1;
          for (; iter.stack[depth].pos < 256; iter.stack[depth].pos++)
            if (n->child[iter.stack[depth].pos]) {
              next = n->child[iter.stack[depth].pos++];
              break;
            }
          break;
        }
      }

      if (next) {
        iter.stack[iter.depth].pos = 0;
        iter.stack[iter.depth].node = next;
        iter.depth++;
      } else
        iter.depth--;
    }

    return false;
  }

  bool iteratorPrev(Iterator& iter) {
    // Skip leaf
    if ((iter.depth) && (isLeaf(iter.stack[iter.depth - 1].node))) iter.depth--;

    // Look for next leaf
    while (iter.depth) {
      Node* node = iter.stack[iter.depth - 1].node;

      // Leaf found
      if (isLeaf(node)) {
        iter.value = getLeafValue(node);
        return true;
      }

      Node* next = nullptr;
      switch (node->type) {
        case NodeType4: {
          Node4* n = static_cast<Node4*>(node);
          if (iter.stack[iter.depth - 1].pos >= 0)
            next = n->child[iter.stack[iter.depth - 1].pos--];
          break;
        }
        case NodeType16: {
          Node16* n = static_cast<Node16*>(node);
          if (iter.stack[iter.depth - 1].pos >= 0)
            next = n->child[iter.stack[iter.depth - 1].pos--];
          break;
        }
        case NodeType48: {
          Node48* n = static_cast<Node48*>(node);
          unsigned depth = iter.depth - 1;
          for (; iter.stack[depth].pos >= 0; iter.stack[depth].pos--)
            if (n->childIndex[iter.stack[depth].pos] != emptyMarker) {
              next = n->child[n->childIndex[iter.stack[depth].pos--]];
              break;
            }
          break;
        }
        case NodeType256: {
          Node256* n = static_cast<Node256*>(node);
          unsigned depth = iter.depth - 1;
          for (; iter.stack[depth].pos >= 0; iter.stack[depth].pos--)
            if (n->child[iter.stack[depth].pos]) {
              next = n->child[iter.stack[depth].pos--];
              break;
            }
          break;
        }
      }

      if (next) {
        if (!isLeaf(next)) {
          switch (next->type) {
            // Find last valid position
            case NodeType4:
            case NodeType16:
              iter.stack[iter.depth].pos = next->count - 1;
              break;
            case NodeType48:
            case NodeType256:
              iter.stack[iter.depth].pos = 255;
              break;
          }
        }
        iter.stack[iter.depth].node = next;
        iter.depth++;
      } else
        iter.depth--;
    }

    return false;
  }

  void swapBytes(uint64_t org_key, uint8_t key[]) {
    reinterpret_cast<uint64_t*>(key)[0] = __builtin_bswap64(org_key);
  }

  void loadKey(uintptr_t tid, uint8_t key[]) {
    // Store the key of the tuple into the key vector
    // Implementation is database specific
    const uint64_t org_key = (*data_)[tid].key;
    swapBytes(org_key, key);
  }

  Node* minimum(Node* node) {
    // Find the leaf with smallest key
    if (!node) return NULL;

    if (isLeaf(node)) return node;

    switch (node->type) {
      case NodeType4: {
        Node4* n = static_cast<Node4*>(node);
        return minimum(n->child[0]);
      }
      case NodeType16: {
        Node16* n = static_cast<Node16*>(node);
        return minimum(n->child[0]);
      }
      case NodeType48: {
        Node48* n = static_cast<Node48*>(node);
        unsigned pos = 0;
        while (n->childIndex[pos] == emptyMarker) pos++;
        return minimum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
        Node256* n = static_cast<Node256*>(node);
        unsigned pos = 0;
        while (!n->child[pos]) pos++;
        return minimum(n->child[pos]);
      }
    }
    throw;  // Unreachable
  }

  Node* maximum(Node* node) {
    // Find the leaf with largest key
    if (!node) return NULL;

    if (isLeaf(node)) return node;

    switch (node->type) {
      case NodeType4: {
        Node4* n = static_cast<Node4*>(node);
        return maximum(n->child[n->count - 1]);
      }
      case NodeType16: {
        Node16* n = static_cast<Node16*>(node);
        return maximum(n->child[n->count - 1]);
      }
      case NodeType48: {
        Node48* n = static_cast<Node48*>(node);
        unsigned pos = 255;
        while (n->childIndex[pos] == emptyMarker) pos--;
        return maximum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
        Node256* n = static_cast<Node256*>(node);
        unsigned pos = 255;
        while (!n->child[pos]) pos--;
        return maximum(n->child[pos]);
      }
    }
    throw;  // Unreachable
  }

  bool leafMatches(Node* leaf, uint8_t key[], unsigned keyLength,
                   unsigned depth, unsigned maxKeyLength) {
    // Check if the key of the leaf is equal to the searched key
    if (depth != keyLength) {
      uint8_t leafKey[maxKeyLength];
      loadKey(getLeafValue(leaf), leafKey);
      for (unsigned i = depth; i < keyLength; i++)
        if (leafKey[i] != key[i]) return false;
    }
    return true;
  }

  unsigned prefixMismatch(Node* node, uint8_t key[], unsigned depth,
                          unsigned maxKeyLength) {
    // Compare the key with the prefix of the node, return the number matching
    // bytes
    unsigned pos;
    if (node->prefixLength > maxPrefixLength) {
      for (pos = 0; pos < maxPrefixLength; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;
      uint8_t minKey[maxKeyLength];
      loadKey(getLeafValue(minimum(node)), minKey);
      for (; pos < node->prefixLength; pos++)
        if (key[depth + pos] != minKey[depth + pos]) return pos;
    } else {
      for (pos = 0; pos < node->prefixLength; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;
    }
    return pos;
  }

  unsigned prefixMismatch(Node* node, uint8_t key[], unsigned depth,
                          unsigned keyLength, unsigned maxKeyLength)
  // Compare the the key with the prefix, return the number matching bytes
  {
    unsigned compBytes = std::min(keyLength - depth, node->prefixLength);
    unsigned pos = 0;
    if (compBytes > maxPrefixLength) {
      for (; pos < maxPrefixLength; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;

      // Load key from database
      uint8_t minKey[maxKeyLength];
      Node* minNode = minimum(node);
      loadKey(getLeafValue(minNode), minKey);
      for (; pos < compBytes; pos++)
        if (key[depth + pos] != minKey[depth + pos]) return pos;
    } else {
      for (; pos < compBytes; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;
    }
    return pos;
  }

  Node* lookupPrefix(Node* node, uint8_t* key, unsigned depth,
                     uint32_t keyLength, unsigned maxKeyLength) {
    if (!node) return nullptr;

    if (isLeaf(node)) {
      // Check if the leaf matches the key
      if (depth == keyLength) return node;

      // Get the key of the leaf
      uint8_t existingKey[maxKeyLength];
      loadKey(getLeafValue(node), existingKey);

      // Check if it matches
      for (unsigned pos = depth; pos < keyLength; pos++)
        if (existingKey[pos] != key[pos]) return nullptr;
      return node;
    }

    if (depth + node->prefixLength < keyLength) {
      // Recurse to next level
      Node* next = *findChild(node, key[depth]);
      return lookupPrefix(next, key, depth, keyLength, maxKeyLength);
    }

    // Special case: only need to check prefix
    if (prefixMismatch(node, key, depth, keyLength, maxKeyLength) !=
        keyLength - depth)
      return nullptr;
    return node;
  }

  bool lookupPrefixMin(Node* root, uint8_t* key, uint32_t keyLength,
                       Iterator& iterator, unsigned maxKeyLength) {
    Node* node = lookupPrefix(root, key, 0, keyLength, maxKeyLength);

    if (node) {
      iterator.depth = 1;
      iterator.stack[0].node = node;
      iterator.stack[0].pos = 0;
      // Descend to first leaf if necessary
      if (isLeaf(node))
        iterator.value = getLeafValue(node);
      else
        return iteratorNext(iterator);  // xxx
      return true;
    } else
      return false;
  }

  bool bound(Node* n, uint8_t key[], unsigned keyLength, Iterator& iterator,
             unsigned maxKeyLength, bool lower = true) {
    iterator.depth = 0;

    if (!n) return false;

    unsigned depth = 0;
    while (true) {
      iterator.stack[iterator.depth].node = n;
      int& pos = iterator.stack[iterator.depth].pos;
      iterator.depth++;

      if (isLeaf(n)) {
        iterator.value = getLeafValue(n);
        if (depth == keyLength) {
          // Equal
          if (lower)
            return true;
          else
            return iteratorNext(iterator);
        }

        uint8_t leafKey[maxKeyLength];
        loadKey(getLeafValue(n), leafKey);
        for (unsigned i = depth; i < keyLength; i++)
          if (leafKey[i] != key[i]) {
            if (leafKey[i] < key[i]) {
              // Less
              iterator.depth--;
              return iteratorNext(iterator);
            }
            // Greater
            return true;
          }

        // Equal
        if (lower)
          return true;
        else
          return iteratorNext(iterator);
      }

      unsigned mismatchPos =
          prefixMismatch(n, key, depth, keyLength, maxKeyLength);
      if (mismatchPos != n->prefixLength) {
        if (n->prefix[mismatchPos] < key[depth + mismatchPos]) {
          // Less
          iterator.depth--;
          return iteratorNext(iterator);
        }
        // Greater
        pos = 0;
        return iteratorNext(iterator);
      }
      depth += n->prefixLength;
      uint8_t keyByte = key[depth];

      Node* next = nullptr;
      switch (n->type) {
        case NodeType4: {
          Node4* node = static_cast<Node4*>(n);
          for (pos = 0; pos < node->count; pos++)
            if (node->key[pos] == keyByte) {
              next = node->child[pos];
              break;
            } else if (node->key[pos] > keyByte)
              break;
          break;
        }
        case NodeType16: {
          Node16* node = static_cast<Node16*>(n);
          for (pos = 0; pos < node->count; pos++)
            if (node->key[pos] == keyByte) {
              next = node->child[pos];
              break;
            } else if (node->key[pos] > keyByte)
              break;
          break;
        }
        case NodeType48: {
          Node48* node = static_cast<Node48*>(n);
          pos = keyByte;
          if (node->childIndex[keyByte] != emptyMarker) {
            next = node->child[node->childIndex[keyByte]];
            break;
          }
          break;
        }
        case NodeType256: {
          Node256* node = static_cast<Node256*>(n);
          pos = keyByte;
          next = node->child[keyByte];
          break;
        }
      }

      if (!next) return iteratorNext(iterator);

      pos++;
      n = next;
      depth++;
    }
  }

  Node* lookup(Node* node, uint8_t key[], unsigned keyLength, unsigned depth,
               unsigned maxKeyLength) {
    // Find the node with a matching key, optimistic version

    bool skippedPrefix =
        false;  // Did we optimistically skip some prefix without checking it?

    while (node != NULL) {
      if (isLeaf(node)) {
        if (!skippedPrefix && depth == keyLength)  // No check required
          return node;

        if (depth != keyLength) {
          // Check leaf
          uint8_t leafKey[maxKeyLength];
          loadKey(getLeafValue(node), leafKey);
          for (unsigned i = (skippedPrefix ? 0 : depth); i < keyLength; i++)
            if (leafKey[i] != key[i]) return NULL;
        }
        return node;
      }

      if (node->prefixLength) {
        if (node->prefixLength < maxPrefixLength) {
          for (unsigned pos = 0; pos < node->prefixLength; pos++)
            if (key[depth + pos] != node->prefix[pos]) return NULL;
        } else
          skippedPrefix = true;
        depth += node->prefixLength;
      }

      node = *findChild(node, key[depth]);
      depth++;
    }

    return NULL;
  }

  Node* lookupPessimistic(Node* node, uint8_t key[], unsigned keyLength,
                          unsigned depth, unsigned maxKeyLength) {
    // Find the node with a matching key, alternative pessimistic version

    while (node != NULL) {
      if (isLeaf(node)) {
        if (leafMatches(node, key, keyLength, depth, maxKeyLength)) return node;
        return NULL;
      }

      if (prefixMismatch(node, key, depth, maxKeyLength) != node->prefixLength)
        return NULL;
      else
        depth += node->prefixLength;

      node = *findChild(node, key[depth]);
      depth++;
    }

    return NULL;
  }

  // Forward references
  // void insertNode4(Node4* node, Node** nodeRef, uint8_t keyByte, Node*
  // child); void insertNode16(Node16* node, Node** nodeRef, uint8_t keyByte,
  // Node* child); void insertNode48(Node48* node, Node** nodeRef, uint8_t
  // keyByte, Node* child); void insertNode256(Node256* node, Node** nodeRef,
  // uint8_t keyByte, Node* child);

  unsigned min(unsigned a, unsigned b) {
    // Helper function
    return (a < b) ? a : b;
  }

  void copyPrefix(Node* src, Node* dst) {
    // Helper function that copies the prefix from the source to the destination
    // node
    dst->prefixLength = src->prefixLength;
    memcpy(dst->prefix, src->prefix, min(src->prefixLength, maxPrefixLength));
  }

  void insert(Node* node, Node** nodeRef, uint8_t key[], unsigned depth,
              uintptr_t value, unsigned maxKeyLength) {
    // Insert the leaf value into the tree

    if (node == NULL) {
      *nodeRef = makeLeaf(value);
      return;
    }

    if (isLeaf(node)) {
      // Replace leaf with Node4 and store both leaves in it
      uint8_t existingKey[maxKeyLength];
      loadKey(getLeafValue(node), existingKey);
      unsigned newPrefixLength = 0;
      while (existingKey[depth + newPrefixLength] ==
             key[depth + newPrefixLength])
        newPrefixLength++;

      Node4* newNode = new Node4();
      newNode->prefixLength = newPrefixLength;
      memcpy(newNode->prefix, key + depth,
             min(newPrefixLength, maxPrefixLength));
      *nodeRef = newNode;

      insertNode4(newNode, nodeRef, existingKey[depth + newPrefixLength], node);
      insertNode4(newNode, nodeRef, key[depth + newPrefixLength],
                  makeLeaf(value));
      return;
    }

    // Handle prefix of inner node
    if (node->prefixLength) {
      unsigned mismatchPos = prefixMismatch(node, key, depth, maxKeyLength);
      if (mismatchPos != node->prefixLength) {
        // Prefix differs, create new node
        Node4* newNode = new Node4();
        *nodeRef = newNode;
        newNode->prefixLength = mismatchPos;
        memcpy(newNode->prefix, node->prefix,
               min(mismatchPos, maxPrefixLength));
        // Break up prefix
        if (node->prefixLength < maxPrefixLength) {
          insertNode4(newNode, nodeRef, node->prefix[mismatchPos], node);
          node->prefixLength -= (mismatchPos + 1);
          memmove(node->prefix, node->prefix + mismatchPos + 1,
                  min(node->prefixLength, maxPrefixLength));
        } else {
          node->prefixLength -= (mismatchPos + 1);
          uint8_t minKey[maxKeyLength];
          loadKey(getLeafValue(minimum(node)), minKey);
          insertNode4(newNode, nodeRef, minKey[depth + mismatchPos], node);
          memmove(node->prefix, minKey + depth + mismatchPos + 1,
                  min(node->prefixLength, maxPrefixLength));
        }
        insertNode4(newNode, nodeRef, key[depth + mismatchPos],
                    makeLeaf(value));
        return;
      }
      depth += node->prefixLength;
    }

    // Recurse
    Node** child = findChild(node, key[depth]);
    if (*child) {
      insert(*child, child, key, depth + 1, value, maxKeyLength);
      return;
    }

    // Insert leaf into inner node
    Node* newNode = makeLeaf(value);
    switch (node->type) {
      case NodeType4:
        insertNode4(static_cast<Node4*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType16:
        insertNode16(static_cast<Node16*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType48:
        insertNode48(static_cast<Node48*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType256:
        insertNode256(static_cast<Node256*>(node), nodeRef, key[depth],
                      newNode);
        break;
    }
  }

  void insertNode4(Node4* node, Node** nodeRef, uint8_t keyByte, Node* child) {
    // Insert leaf into inner node
    if (node->count < 4) {
      // Insert element
      unsigned pos;
      for (pos = 0; (pos < node->count) && (node->key[pos] < keyByte); pos++)
        ;
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos,
              (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByte;
      node->child[pos] = child;
      node->count++;
    } else {
      // Grow to Node16
      Node16* newNode = new Node16();
      *nodeRef = newNode;
      newNode->count = 4;
      copyPrefix(node, newNode);
      for (unsigned i = 0; i < 4; i++) newNode->key[i] = node->key[i];
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      delete node;
      return insertNode16(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode16(Node16* node, Node** nodeRef, uint8_t keyByte,
                    Node* child) {
    // Insert leaf into inner node
    if (node->count < 16) {
      // Insert element
      unsigned pos = 0;
      while ((pos < node->count) && (node->key[pos] < keyByte)) pos++;
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos,
              (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByte;
      node->child[pos] = child;
      node->count++;
    } else {
      // Grow to Node48
      Node48* newNode = new Node48();
      *nodeRef = newNode;
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      for (unsigned i = 0; i < node->count; i++)
        newNode->childIndex[node->key[i]] = i;
      copyPrefix(node, newNode);
      newNode->count = node->count;
      delete node;
      return insertNode48(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode48(Node48* node, Node** nodeRef, uint8_t keyByte,
                    Node* child) {
    // Insert leaf into inner node
    if (node->count < 48) {
      // Insert element
      unsigned pos = node->count;
      if (node->child[pos])
        for (pos = 0; node->child[pos] != NULL; pos++)
          ;
      node->child[pos] = child;
      node->childIndex[keyByte] = pos;
      node->count++;
    } else {
      // Grow to Node256
      Node256* newNode = new Node256();
      for (unsigned i = 0; i < 256; i++)
        if (node->childIndex[i] != 48)
          newNode->child[i] = node->child[node->childIndex[i]];
      newNode->count = node->count;
      copyPrefix(node, newNode);
      *nodeRef = newNode;
      delete node;
      return insertNode256(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode256(Node256* node, Node** nodeRef, uint8_t keyByte,
                     Node* child) {
    // Insert leaf into inner node
    node->count++;
    node->child[keyByte] = child;
  }

  // Forward references
  // void eraseNode4(Node4* node, Node** nodeRef, Node** leafPlace);
  // void eraseNode16(Node16* node, Node** nodeRef, Node** leafPlace);
  // void eraseNode48(Node48* node, Node** nodeRef, uint8_t keyByte);
  // void eraseNode256(Node256* node, Node** nodeRef, uint8_t keyByte);

  void erase(Node* node, Node** nodeRef, uint8_t key[], unsigned keyLength,
             unsigned depth, unsigned maxKeyLength) {
    // Delete a leaf from a tree

    if (!node) return;

    if (isLeaf(node)) {
      // Make sure we have the right leaf
      if (leafMatches(node, key, keyLength, depth, maxKeyLength))
        *nodeRef = NULL;
      return;
    }

    // Handle prefix
    if (node->prefixLength) {
      if (prefixMismatch(node, key, depth, maxKeyLength) != node->prefixLength)
        return;
      depth += node->prefixLength;
    }

    Node** child = findChild(node, key[depth]);
    if (isLeaf(*child) &&
        leafMatches(*child, key, keyLength, depth, maxKeyLength)) {
      // Leaf found, delete it in inner node
      switch (node->type) {
        case NodeType4:
          eraseNode4(static_cast<Node4*>(node), nodeRef, child);
          break;
        case NodeType16:
          eraseNode16(static_cast<Node16*>(node), nodeRef, child);
          break;
        case NodeType48:
          eraseNode48(static_cast<Node48*>(node), nodeRef, key[depth]);
          break;
        case NodeType256:
          eraseNode256(static_cast<Node256*>(node), nodeRef, key[depth]);
          break;
      }
    } else {
      // Recurse
      erase(*child, child, key, keyLength, depth + 1, maxKeyLength);
    }
  }

  void eraseNode4(Node4* node, Node** nodeRef, Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos = leafPlace - node->child;
    memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
    memmove(node->child + pos, node->child + pos + 1,
            (node->count - pos - 1) * sizeof(uintptr_t));
    node->count--;

    if (node->count == 1) {
      // Get rid of one-way node
      Node* child = node->child[0];
      if (!isLeaf(child)) {
        // Concantenate prefixes
        unsigned l1 = node->prefixLength;
        if (l1 < maxPrefixLength) {
          node->prefix[l1] = node->key[0];
          l1++;
        }
        if (l1 < maxPrefixLength) {
          unsigned l2 = min(child->prefixLength, maxPrefixLength - l1);
          memcpy(node->prefix + l1, child->prefix, l2);
          l1 += l2;
        }
        // Store concantenated prefix
        memcpy(child->prefix, node->prefix, min(l1, maxPrefixLength));
        child->prefixLength += node->prefixLength + 1;
      }
      *nodeRef = child;
      delete node;
    }
  }

  void eraseNode16(Node16* node, Node** nodeRef, Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos = leafPlace - node->child;
    memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
    memmove(node->child + pos, node->child + pos + 1,
            (node->count - pos - 1) * sizeof(uintptr_t));
    node->count--;

    if (node->count == 3) {
      // Shrink to Node4
      Node4* newNode = new Node4();
      newNode->count = 4;
      copyPrefix(node, newNode);
      for (unsigned i = 0; i < 4; i++) newNode->key[i] = node->key[i];
      memcpy(newNode->child, node->child, sizeof(uintptr_t) * 4);
      *nodeRef = newNode;
      delete node;
    }
  }

  void eraseNode48(Node48* node, Node** nodeRef, uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[node->childIndex[keyByte]] = NULL;
    node->childIndex[keyByte] = emptyMarker;
    node->count--;

    if (node->count == 12) {
      // Shrink to Node16
      Node16* newNode = new Node16();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (unsigned b = 0; b < 256; b++) {
        if (node->childIndex[b] != emptyMarker) {
          newNode->key[newNode->count] = b;
          newNode->child[newNode->count] = node->child[node->childIndex[b]];
          newNode->count++;
        }
      }
      delete node;
    }
  }

  void eraseNode256(Node256* node, Node** nodeRef, uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[keyByte] = NULL;
    node->count--;

    if (node->count == 37) {
      // Shrink to Node48
      Node48* newNode = new Node48();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (unsigned b = 0; b < 256; b++) {
        if (node->child[b]) {
          newNode->childIndex[b] = newNode->count;
          newNode->child[newNode->count] = node->child[b];
          newNode->count++;
        }
      }
      delete node;
    }
  }

  void destructTree(Node* node) {
    if (!node) return;

    switch (node->type) {
      case NodeType4: {
        auto n4 = static_cast<Node4*>(node);
        for (auto i = 0; i < node->count; i++) {
          if (!isLeaf(n4->child[i])) {
            destructTree(n4->child[i]);
          }
        }
        delete n4;
        break;
      }
      case NodeType16: {
        auto n16 = static_cast<Node16*>(node);
        for (auto i = 0; i < node->count; i++) {
          if (!isLeaf(n16->child[i])) {
            destructTree(n16->child[i]);
          }
        }
        delete n16;
        break;
      }
      case NodeType48: {
        auto n48 = static_cast<Node48*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n48->childIndex[i] != emptyMarker &&
              !isLeaf(n48->child[n48->childIndex[i]])) {
            destructTree(n48->child[n48->childIndex[i]]);
          }
        }
        delete n48;
        break;
      }
      case NodeType256: {
        auto n256 = static_cast<Node256*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n256->child[i] != nullptr && !isLeaf(n256->child[i])) {
            destructTree(n256->child[i]);
          }
        }
        delete n256;
        break;
      }
    }
  }

  Node* tree_ = NULL;
  const std::vector<KeyValue<uint64_t>>* data_;
  uint64_t data_size_;
};

template <int size_scale>
uint64_t ARTPrimaryLB<size_scale>::allocated_byte_count;
