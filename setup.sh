#!/bin/bash

home=$PWD
source ~/setup_root6.sh

cd ../larcv2
source configure.sh

cd ../torchlarcvdataset
source setenv.sh

cd $home

export CUDA_VISIBLE_DEVICES=1
