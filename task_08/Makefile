all: clean prog_gpu

clean:
	rm -f prog* a.out cpu.* gpu.* core.*

prog_gpu:
	nvcc -O0 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 stancil.cu -o prog_gpu
