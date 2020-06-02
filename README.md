```
   _____ ____  _____ ____ 
  / ___// __ \/ ___// __ \
  \__ \/ / / /\__ \/ / / /
 ___/ / /_/ /___/ / /_/ / 
/____/\____//____/_____/  
                          
```

Search on Sorted Data Benchmark
====

[SOSD](https://learned.systems/papers/sosd.pdf) is a benchmark to compare (learned) index structures on equality lookup performance.
It comes with state-of-the-art baseline implementations to compare against and many datasets to compare on.
Each dataset consists of 200 million 32-bit or 64-bit unsigned integers.

## Usage instructions

We provide a number of scripts to automate things. Each is located in the `scripts` directory, but should be executed from the repository root.

## Running the benchmark

* `scripts/download.sh` downloads and stores required data from the Internet
* `scripts/build_rmis.sh` compiles and builds the RMIs for each dataset
* `scripts/prepare.sh` constructs query workloads and compiles the benchmark
* `scripts/execute.sh` executes the benchmark on each workload, storing the results in `results`

## Cite

If you use this benchmark in your own work, please cite our paper:

```
@article{sosd,
  title={SOSD: A Benchmark for Learned Indexes},
  author={Kipf, Andreas and Marcus, Ryan and van Renen, Alexander and Stoian, Mihail and Kemper, Alfons and Kraska, Tim and Neumann, Thomas},
  journal={NeurIPS Workshop on Machine Learning for Systems},
  year={2019}
}
```
