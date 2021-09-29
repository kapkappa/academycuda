all: clean prog_gpu

clean:
	rm -f prog_* a.out cpu.* gpu.* core.*

prog_gpu:
	nvcc -O0 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 vec_sum.cu -o prog_gpu
