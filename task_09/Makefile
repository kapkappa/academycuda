all: clean prog_gpu

clean:
	rm -f prog* a.out cpu.* gpu.* core.* log.*

prog_gpu:
	nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 pinmem.cu -o prog_gpu
