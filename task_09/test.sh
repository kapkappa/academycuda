#!/bin/bash

for size in 500000 2000000 50000000 150000000; do
#for size in 500 2000 10000 100000 ; do

#mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu -- $size
echo ''
done

mpisubmit.pl --stdout gpu.nprof.out --stderr gpu.nprof.err --gpu ~/local_cuda/bin/nvprof -- --analysis-metrics -f -o short ./prog_gpu 268430000
