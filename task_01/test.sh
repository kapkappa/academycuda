#!/bin/bash

for size in 1000 10000 100000 1000000; do
#for size in 1000000; do

mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu -- $size

done
