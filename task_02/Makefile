all: clean prog_gpu

clean:
	rm -f prog* a.out cpu.* gpu.* core.*

prog_cpu:
	gcc -O3 main.c -o prog

prog_gpu:
	nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 transpose.cu -o prog_gpu
