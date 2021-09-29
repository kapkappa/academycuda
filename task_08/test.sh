#!/bin/bash

#for size in 100 512 1024; do
for version in 0 1; do
for size in 1048576 268435456; do

mpisubmit.pl --stdout gpu.out.$version.$size --stderr gpu.err.$version.$size --gpu prog_gpu -- $size $version

done
done
