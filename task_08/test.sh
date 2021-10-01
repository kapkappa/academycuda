#!/bin/bash

#for version in 0 1; do
#for size in 1048576 268435456; do
#for size in 512 2048; do

#pisubmit.pl --stdout log.out.$version.$size --stderr gpu.err.$version.$size --gpu prog_gpu -- $size $version

#done
#done

mpisubmit.pl --stdout gpu.nvprof.out --stderr gpu.nvprof.err --gpu ~/local_cuda/bin/nvprof -- -f -o timeline_short ~/academycuda/task_08/prog_gpu 268430000 1
