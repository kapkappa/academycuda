#!/bin/bash

for version in old new; do
#for size in 1048576 268435456; do
#for size in 512 2048; do

mpisubmit.pl --stdout log.out.$version --stderr gpu.err.$version --gpu prog_$version

#done
done

#rm err.log out.log
#mpisubmit.pl --stdout out.log --stderr err.log --gpu ~/local_cuda/bin/nvprof -- --analysis-metrics -o short ~/academycuda/task_08/prog_gpu 2000000 1
