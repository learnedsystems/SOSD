# Quick start instructions

## Clone + Build
```bash
git clone git@github.com:harald-lang/dtl-bloomfilter.git dtl
mkdir dtl/test/build
cd dtl/test/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make bloomfilter_rts_experiments_core-avx2
make bloomfilter_rts_experiments_knl
```

## Run performance experiment(s)
```bash
REPEAT_CNT=1000 GRAIN_SIZE=14 KEY_CNT=14 BF_SIZE_LO=10 BF_SIZE_HI=16 THREAD_CNT_LO=1 THREAD_STEP_MODE=2 HBM=1 REPL=1 BF_K=1 ./bloomfilter_rts_experiments_core-avx2 --gtest_filter=*performance*
```
Settings:
* `BF_K=[1..7]` k
* `BF_SIZE_LO=a`, `BF_SIZE_HI=b`: repeats the experiment with varying Bloom filter sizes: [2^a, 2^b] 
* `KEY_CNT=n`: specifies the input size per thread (2^n)
* `REPEAT_CNT=n`: repeats the experiment n times and report the avg.

Multi threading:
* `THREAD_CNT_LO=a`, `THREAD_CNT_HI=b`: specifies the number of parallel threads. 
    The experiment is first executed using *a* threads. 
    The experiment is repeated with an increasing number of threads, until *b* is reached.
    By default *b* is set to the hardware concurrency. 
* `THREAD_STEP_MODE=m`: specifies how the number of threads are increased (1 = linear, 2 = exponential)
* `GRAIN_SIZE=n`: the size of a batch (a work item) executed by a single thread, 2^n

Memory allocation:
* `HBM=0|1`: 0 = use RAM, 1 = use HBM memory nodes (default)
* `REPL=0|1`: 0 = use interleaved memory, 1 = replicate the Bloom filter to all memory nodes


Example output:
```
Note: Google Test filter = *performance*
[==========] Running 1 test from 1 test case.
[----------] Global test environment set-up.
[----------] 1 test from bloom
[ RUN      ] bloom.filter_performance_parallel_vec
Configuration:
  BF_K=3, BF_SIZE_LO=18, BF_SIZE_HI=18
  GRAIN_SIZE=14, THREAD_CNT_LO=64, THREAD_CNT_HI=256, THREAD_STEP=1, THREAD_STEP_MODE=2 (1=linear, 2=exponential), KEY_CNT=14 (per thread), REPEAT_CNT=100000
  HBM=1 (0=no, 1=yes), REPL=1 (0=interleaved, 1=replicate)
-- bloomfilter parameters --
static
  h:                    1
  k:                    3
  word bitlength:       32
  hash value bitlength: 32
  sectorized:           false
  sector count:         1
  sector bitlength:     32
  hash bits per sector: 5
  hash bits per word:   15
  hash bits wasted:     0
  remaining hash bits:  17
  max m:                4194304
  max size [MiB]:       0.5
dynamic
  m:                    262144
  size [KiB]:           32
  population count:     43454
  load factor:          0.165764
replicate bloomfilter to node 4
replicate bloomfilter to node 5
replicate bloomfilter to node 6
replicate bloomfilter to node 7
bf_size: 32768 [bytes], thread_cnt: 64, key_cnt: 1048576, grain_size: 16384, performance: 28287014204 [1/s], cycles/probe: 2.8631 (matchcnt: 28229)
-- bloomfilter parameters --
static
  h:                    1
  k:                    3
  word bitlength:       32
  hash value bitlength: 32
  sectorized:           false
  sector count:         1
  sector bitlength:     32
  hash bits per sector: 5
  hash bits per word:   15
  hash bits wasted:     0
  remaining hash bits:  17
  max m:                4194304
  max size [MiB]:       0.5
dynamic
  m:                    262144
  size [KiB]:           32
  population count:     43454
  load factor:          0.165764
replicate bloomfilter to node 4
replicate bloomfilter to node 5
replicate bloomfilter to node 6
replicate bloomfilter to node 7
bf_size: 32768 [bytes], thread_cnt: 128, key_cnt: 2097152, grain_size: 16384, performance: 34975390403 [1/s], cycles/probe: 4.4389 (matchcnt: 40037)
-- bloomfilter parameters --
static
  h:                    1
  k:                    3
  word bitlength:       32
  hash value bitlength: 32
  sectorized:           false
  sector count:         1
  sector bitlength:     32
  hash bits per sector: 5
  hash bits per word:   15
  hash bits wasted:     0
  remaining hash bits:  17
  max m:                4194304
  max size [MiB]:       0.5
dynamic
  m:                    262144
  size [KiB]:           32
  population count:     43454
  load factor:          0.165764
replicate bloomfilter to node 4
replicate bloomfilter to node 5
replicate bloomfilter to node 6
replicate bloomfilter to node 7
bf_size: 32768 [bytes], thread_cnt: 256, key_cnt: 4194304, grain_size: 16384, performance: 20000607603 [1/s], cycles/probe: 16.1042 (matchcnt: 64130)
[       OK ] bloom.filter_performance_parallel_vec (30997 ms)
[----------] 1 test from bloom (30997 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test case ran. (30998 ms total)
[  PASSED  ] 1 test.

```

## Run accuracy experiment (may run for hours)
```bash
./bloomfilter_rts_experiments_core-avx2 --gtest_filter=*accuracy*
```