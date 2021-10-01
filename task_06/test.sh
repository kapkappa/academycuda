#!/bin/bash

#for version in old new; do
#for size in 1048576 268435456; do
#for size in 512 2048; do

#mpisubmit.pl --stdout gpu.out.$version --stderr gpu.err.$version --gpu prog_$version
#mpisubmit.pl --stdout gpu.nprof.out.$version --stderr gpu.nprof.err.$version --gpu ~/local_cuda/bin/nvprof -- --analysis-metrics -f -o short.$version ./prog_$version

#done
#done

mpisubmit.pl --stdout gpu.nprof.out --stderr gpu.nprof.err --gpu ~/local_cuda/bin/nvprof -- -f -o timeline_short ./prog_old 26843000

