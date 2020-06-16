#! /usr/bin/env bash

rm -rf rmi_data
rm -rf competitors/rmi

if [ -f rmi.tar.zst ]; then
    echo "Tarball already exists, using it."
else
    echo "Downloading tarball..."
    wget -O rmi.tar.zst https://www.dropbox.com/s/uuyknn0qeag93sj/rmi.tar.zst?dl=1
fi

echo "Decompressing..."
tar -I zstd -xf rmi.tar.zst
echo "Placing..."
mkdir -p competitors/rmi
touch competitors/rmi/.keep
mv rmi_competitors/* competitors/rmi
rm -r rmi_competitors
echo "Done!"
