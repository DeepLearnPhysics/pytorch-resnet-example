#!/bin/bash

home=$PWD
source /usr/local/root/6.16.00_python3/bin/thisroot.sh

cd ../larcv2
source configure.sh

cd ../larcvdataset
source setenv.sh

cd $home

export CUDA_VISIBLE_DEVICES=1
