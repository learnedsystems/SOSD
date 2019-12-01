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

#### AWS setup

If you'd like to run the whole benchmark on AWS, provision a new `c5d.4xlarge` node using the Ubuntu 18 AMI, and then copy and paste this after logging in.

NOTE: The `scripts/setup_aws.sh` sets up an AWS `c5d.4xlarge` or `c5d.2xlarge` node for the rest of the benchmark scripts (it is *destructive*!). It can be run from your home directory, and will automatically clone this repository.

```bash
wget https://raw.githubusercontent.com/learnedsystems/SOSD/master/scripts/setup_aws.sh
chmod +x ./setup_aws.sh
. ./setup_aws.sh
cd /sosdata/SOSD
```

#### Local setup

If you'd like to run the benchmark locally, you need a machine with at least 16GiB of RAM and 50GiB of free disk space. We've included a `setup_anywhere.sh` script which installs the prerequistes on `apt` based systems (Debian, Ubuntu).
To set it up, follow these steps:

```bash
git clone git@github.com:learnedsystems/SOSD.git
cd SOSD
. ./scripts/setup_anywhere.sh
```

#### Running the benchmark

* `scripts/download.sh` downloads and stores required data from the Internet
* `scripts/build_rmis.sh` compiles and builds the RMIs for each dataset
* `scripts/prepare.sh` constructs query workloads and compiles the benchmark
* `scripts/execute.sh` executes the benchmark on each workload, storing the results in `results`
* `scripts/create_leaderboard.sh` gathers all results in `results` into an easy to read table (see below)

To run them all, execute `reproduce.sh`.

## Results

Here is the current ranking of index structures.
If you have any improvements to existing indexes or want to add new ones, feel free to contact us.
Provided that they fulfill the constraints outlined in the [paper](https://learned.systems/papers/sosd.pdf), we will update the "leaderboard" accordingly.
All measurements are performed on the target platform: `c5d.4xlarge`.
For obvious reasons, we retain the freedom to exclude any submissions at our discretion.
 
|               | ART       | B-tree    | BS        | FAST      | IS        | RBS       | RMI       | RS        | TIP       |
| ------------- | ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|
| amzn32        |       n/a |       529 |       773 |       244 |      4604 |       325 |       264 |       275 |       731 |
| face32        |       187 |       524 |       771 |       229 |      1285 |       312 |       274 |       386 |       964 |
| logn32        |       n/a |       522 |       765 |       294 |       n/a |       471 |      97.0 |       105 |       744 |
| norm32        |       191 |       522 |       771 |       229 |     10257 |       355 |      71.7 |      70.9 |       884 |
| uden32        |       102 |       521 |       771 |       228 |      39.8 |       333 |      54.2 |      64.2 |       176 |
| uspr32        |       n/a |       524 |       771 |       230 |       469 |       301 |       153 |       200 |       400 |
| amzn64        |       n/a |       601 |       804 |       n/a |      4736 |       387 |       266 |       288 |       759 |
| face64        |       391 |       592 |       784 |       n/a |      1893 |       337 |       334 |       461 |      1232 |
| logn64        |       309 |       597 |       784 |       n/a |       n/a |       753 |       179 |       120 |       454 |
| norm64        |       266 |       592 |       785 |       n/a |     10510 |       405 |      71.5 |      70.5 |       862 |
| osmc64        |       n/a |       599 |       785 |       n/a |     95076 |       492 |       402 |       437 |      7186 |
| uden64        |       112 |       592 |       784 |       n/a |      43.4 |       344 |      54.3 |      53.9 |       193 |
| uspr64        |       287 |       591 |       785 |       n/a |       449 |       313 |       169 |       214 |       428 |
| wiki64        |       n/a |       608 |       802 |       n/a |      7846 |       364 |       222 |       218 |      1019 |

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
