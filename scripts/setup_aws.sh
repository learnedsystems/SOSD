#! /usr/bin/env bash

echo "Formatting the attached NVME drive..."
sudo parted /dev/nvme0n1 mklabel -s gpt
sudo partprobe
sudo parted -a optimal /dev/nvme0n1 mkpart primary 0% 100%
sudo partprobe
sync
sudo mkfs.ext4 -F -F /dev/nvme0n1p1
sudo mkdir /sosdata
sync
sudo mount /dev/nvme0n1p1 /sosdata
sudo chown -R ubuntu /sosdata

cd /sosdata
rm -rf SOSD
echo "Cloning SOSD repository..."
echo "github.com,192.30.253.112 ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" >> ~/.ssh/known_hosts
git clone --recurse-submodules git@github.com:learnedsystems/SOSD.git
cd SOSD

echo "Installing software..."
export DEBIAN_FRONTEND=noninteractive
sudo apt update -y
sudo apt install -y python3-pip cmake linux-tools-common linux-tools-aws libboost-all-dev m4
pip3 install --user numpy scipy scikit-learn jupyterlab pandas matplotlib
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup update

echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
. ~/.bashrc
