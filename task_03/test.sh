#!/bin/bash

for size in 10000 100000 1000000 10000000; do
#for size in 16 64 256 1024; do

mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu -- $size

done
