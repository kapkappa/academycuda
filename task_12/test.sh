#!/bin/bash

for size in 30 300 3000 20000 25000 30000; do
#for size in 500000 2000000 50000000 150000000; do

mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu $size

done
#done

#mpisubmit.pl --stdout gpu.nprof.out --stderr gpu.nprof.err --gpu ~/local_cuda/bin/nvprof -- -f -o timeline_short ./prog_gpu 26843000 0 1
