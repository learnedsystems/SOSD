#pragma once

#include <type_traits>

#include "base.h"
#include "wormhole/wh.h"

struct kv* kv_dup_in_count(const struct kv* const kv, void* const priv) {
  int64_t current_usage = *((int64_t*)priv);
  *((int64_t*)priv) = current_usage + kv_size(kv);
  return kv_dup(kv);
}

struct kv* kv_dup_out_count(const struct kv* const kv, struct kv* const out) {
  return kv_dup2(kv, out);
}

void kv_free_count(struct kv* const kv, void* const priv) {
  int64_t current_usage = *((int64_t*)priv);
  *((int64_t*)priv) = current_usage - kv_size(kv);
  free(kv);
}

template <class KeyType, int size_scale>
class Wormhole : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    data_size_ = data.size();

    kvmap_mm allocator = (kvmap_mm){kv_dup_in_count, kv_dup_out_count,
                                    kv_free_count, (void*)&usage_};
    index = wormhole_create(&allocator);

    size_t key_length =
        (std::is_same<KeyType, std::uint64_t>::value ? sizeof(uint64_t)
                                                     : sizeof(uint32_t));

    // wormhole seek finds the first key that's >= the search key.
    // we need to find the last key that's <= the search key.
    // unfortunately, we cannot move the wormhole iterator backwards,
    // so instead we will "shift" the value array up one:
    // the value associated with each key will be the index of *previous*
    // key.

    __in = kv_create(NULL, key_length, NULL, sizeof(uint64_t));
    __out = kv_create(NULL, key_length, NULL, sizeof(uint64_t));

    std::vector<uint64_t> keys;
    std::vector<uint64_t> values;

    for (unsigned int i = 0; i < data.size(); i++) {
      if (size_scale > 1 && i % size_scale != 0) continue;

      keys.push_back(data[i].key);
      values.push_back(data[i].value);
    }

    for (unsigned int i = 1; i < keys.size(); i++) {
      if (std::is_same<KeyType, std::uint64_t>::value) {
        uint64_t swappedKey = __builtin_bswap64(keys[i]);
        *(uint64_t*)kv_kptr(__in) = swappedKey;
      } else {
        uint64_t swappedKey = __builtin_bswap32(keys[i]);
        *(uint32_t*)kv_kptr(__in) = swappedKey;
      }

      uint64_t value = values[i - 1];
      *(uint64_t*)kv_vptr(__in) = value;
      kv_update_hash(__in);
      whunsafe_set(index, __in);
      max_key_ = keys[i];
      last_index_ = values[i];
    }

    return util::timing([&] {
      auto iter = whunsafe_iter_create(index);
      whunsafe_iter_peek(iter, __out);
      if (std::is_same<KeyType, std::uint64_t>::value) {
        min_key_ = __builtin_bswap64(*(uint64_t*)kv_kptr(__out));
      } else {
        min_key_ = __builtin_bswap32(*(uint32_t*)kv_kptr(__out));
      }
      whunsafe_iter_destroy(iter);
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    if (std::is_same<KeyType, std::uint64_t>::value) {
      *(uint64_t*)kv_kptr(__in) = __builtin_bswap64(lookup_key - 1);
    } else {
      *(uint32_t*)kv_kptr(__in) = __builtin_bswap32(lookup_key - 1);
    }

    kv_update_hash(__in);

    auto iter = whunsafe_iter_create(index);
    whunsafe_iter_seek(iter, __in);

    if (!whunsafe_iter_peek(iter, __out)) {
      // past the last key
      whunsafe_iter_destroy(iter);
      return (SearchBound){last_index_, data_size_};
    }
    uint64_t start = *(uint64_t*)kv_vptr(__out);

    whunsafe_iter_next(iter, __out);
    whunsafe_iter_peek(iter, __out);
    uint64_t stop = *(uint64_t*)kv_vptr(__out);

    stop = (start == stop ? data_size_ - 1 : stop);
    whunsafe_iter_destroy(iter);
    return (SearchBound){start, stop + 1};
  }

  std::string name() const { return "Wormhole"; }

  std::size_t size() const {
    // return used memory in bytes
    if (usage_ < 0) {
      util::fail("Wormhole memory usage was negative!");
    }

    return usage_;
  }

  bool applicable(bool unique, const std::string& data_filename) {
    // only supports unique keys.
    return unique;
  }

  int variant() const { return size_scale; }

  ~Wormhole() {
    if (index) wormhole_destroy(index);
  }

 private:
  struct wormhole* index = NULL;
  struct kv* __in;
  struct kv* __out;

  uint64_t data_size_;
  KeyType min_key_;
  KeyType max_key_;
  uint64_t last_index_;
  int64_t usage_ = 0;
};
