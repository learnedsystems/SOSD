#! /usr/bin/env bash

rm -f competitors/rmi/*.h
rm -f competitors/rmi/*.cpp
rm -f rmi_data/*

scripts/build_rmis.sh
