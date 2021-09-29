#!/bin/bash

for size in 100 512 1024; do

mpisubmit.pl --stdout gpu.out.$size --stderr gpu.err.$size --gpu prog_gpu -- $size

done
