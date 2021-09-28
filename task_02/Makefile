all: clean prog_cpu prog_gpu

clean:
	rm -f prog_* a.out cpu.* gpu.* core.*

prog_cpu:
	gcc -O3 vec_sum.c -lm -o prog_cpu

prog_gpu:
	nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 vec_sum.cu -o prog_gpu
