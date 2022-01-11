```
   _____ ____  _____ ____ 
  / ___// __ \/ ___// __ \
  \__ \/ / / /\__ \/ / / /
 ___/ / /_/ /___/ / /_/ / 
/____/\____//____/_____/  
                          
```

Search on Sorted Data Benchmark
====

[![Build Status](https://drone.rm.cab/api/badges/learnedsystems/SOSD/status.svg)](https://drone.rm.cab/learnedsystems/SOSD)

[SOSD](https://learned.systems/papers/sosd.pdf) is a benchmark to compare (learned) index structures on equality lookup performance over densely packed, sorted data.
It comes with state-of-the-art baseline implementations to compare against and many datasets to compare on.
Each dataset consists of 200 million to 800 million 32-bit or 64-bit unsigned integers.

## Dependencies

On vanilla Ubuntu 20.04 LTS:
```
sudo apt -y update
sudo apt -y install zstd python3-pip m4 cmake clang libboost-all-dev
pip3 install --user numpy scipy
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Usage instructions

We provide a number of scripts to automate things. Each is located in the `scripts` directory, but should be executed from the repository root.

## Running the benchmark

- `./scripts/download.sh` downloads and stores required data from the Internet
- `./scripts/build_rmis.sh` compiles and builds the RMIs for each dataset. If you run into the error message `error: no override and no default toolchain set`, try running `rustup install stable`.
  - `./scripts/download_rmis.sh` will download pre-built RMIs instead, which may be faster. You'll need to run `build_rmis.sh` if you want to measure build times on your platform.
- `./scripts/prepare.sh` constructs query workloads and compiles the benchmark
- `./scripts/execute.sh` executes the benchmark on each workload, storing the results in `results`. You can use the `-c` flag to output a .csv file of results rather than a .txt.

Build times can be long, as we make aggressive use of templates to ensure we do not accidentally measure vtable lookup time. For development, this can be annoying: you can set `USE_FAST_MODE` in `config.h` to disable some features and get a faster build time.

## Cite

If you use this benchmark in your own work, please cite us:

```
@article{sosd-vldb,
  author    = {Ryan Marcus and
               Andreas Kipf and
               Alexander van Renen and
               Mihail Stoian and
               Sanchit Misra and
               Alfons Kemper and
               Thomas Neumann and
               Tim Kraska},
  title     = {Benchmarking Learned Indexes},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {14},
  number    = {1},
  pages     = {1--13},
  year      = {2020}
}

@article{sosd-neurips,
  title={SOSD: A Benchmark for Learned Indexes},
  author={Kipf, Andreas and Marcus, Ryan and van Renen, Alexander and Stoian, Mihail and Kemper, Alfons and Kraska, Tim and Neumann, Thomas},
  journal={NeurIPS Workshop on Machine Learning for Systems},
  year={2019}
}
```
