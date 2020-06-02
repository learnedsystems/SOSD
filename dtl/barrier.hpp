#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <dtl/dtl.hpp>

namespace dtl {

class barrier {
 private:
  std::mutex mutex;
  std::condition_variable conditionVar;
  const std::size_t threadCount;
  std::size_t cntr;
  std::atomic<bool> gogo;
 public:
  explicit barrier(std::size_t threadCount) :
      threadCount(threadCount), cntr(threadCount), gogo(false) {
  }
  void wait() {
    while(gogo) {
      // wait until barrier is ready for re-use
    }

    std::unique_lock<std::mutex> lock(mutex);
    cntr--;

    if (cntr == 0) {
      gogo = true;
      cntr++;
      conditionVar.notify_all();
    } else {
      conditionVar.wait(lock, [this] {return gogo.load();});
      cntr++;
      if (cntr == threadCount) {
        gogo = false;
      }
    }
  }
};


class busy_barrier {
 private:
  const std::size_t threadCount;
  std::atomic<std::size_t> cntr;
  std::atomic<bool> gogo;

 public:
  explicit busy_barrier(std::size_t threadCount) :
      threadCount(threadCount), cntr(threadCount), gogo(false) {
  }

  __forceinline__
  void wait() {
    while (gogo) {
      // wait until barrier is ready for re-use
    }
    const std::size_t t = cntr.fetch_sub(1);
    if (t == 1) {
      // the last thread arrived
      cntr++;
      gogo = true;
    }
    else {
      while (!gogo) {
        // busy wait until all threads arrived
      }
      std::size_t prevCntr = cntr.fetch_add(1);
      if (prevCntr + 1 == threadCount) {
        gogo = false;
      }
    }
  }
};


class busy_barrier_one_shot {
 private:
  const std::size_t threadCount;
  std::atomic<std::size_t> cntr;
  std::atomic<bool> gogo;

 public:
  explicit busy_barrier_one_shot(std::size_t threadCount) :
      threadCount(threadCount), cntr(threadCount), gogo(false) { }

  __forceinline__
  void wait() {
    const std::size_t t = cntr.fetch_sub(1);
    if (t == 1) {
      // the last thread arrived
      gogo = true;
    }
    else {
      while (!gogo) {
        // busy wait until all threads arrived
      }
    }
  }

  void reset() {
    cntr = threadCount;
    gogo = false;
  }

};


} // namespace dtl