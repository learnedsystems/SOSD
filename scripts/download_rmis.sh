#! /usr/bin/env bash

rm -rf rmi_data
rm -rf competitors/rmi

if [ -f rmi.tar.zst ]; then
    echo "Tarball already exists, using it."
else
    echo "Downloading tarball..."
    wget -O rmi.tar.zst https://s3.wasabisys.com/sosd/rmi.tar.zst
fi

echo "Decompressing..."
tar -I zstd -xf rmi.tar.zst
echo "Placing..."
mkdir -p competitors/rmi
touch competitors/rmi/.keep
mv rmi_competitors/* competitors/rmi
rm -r rmi_competitors
echo "Done!"
