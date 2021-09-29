#!/bin/bash

for size in 50 100 200 500; do
#for size in 1000 2000 5000 10000; do
#for size in 20000; do

mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu -- $size

done
