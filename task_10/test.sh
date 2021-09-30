#!/bin/bash

#for version in 0 1; do
#for size in 500 2000 10000 100000 ; do
#for size in 500000 2000000 50000000 150000000; do

#mpisubmit.pl --stdout gpu.out.$size.$version --stderr gpu.err.$size.$version --gpu prog_gpu -- $size $version 2

#done
#done

mpisubmit.pl --stdout gpu.nprof.out --stderr gpu.nprof.err --gpu ~/local_cuda/bin/nvprof -- -f -o timeline_short ./prog_gpu 26843000 0 1
