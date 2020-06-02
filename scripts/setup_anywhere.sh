#! /usr/bin/env bash

git submodule init
git submodule update

echo "Installing software..."
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y python3-pip cmake linux-tools-common linux-tools-aws libboost-all-dev m4 zstd
pip3 install --user numpy scipy scikit-learn jupyterlab pandas matplotlib
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup update

echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
. ~/.bashrc
