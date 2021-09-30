all: clean prog_old prog_new

clean:
	rm -f prog* a.out cpu.* gpu.* core.* log.*

prog_old:
	nvcc --std c++11 -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 graph.cu -o prog_old

prog_new:
	nvcc --std c++11 -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 w_graph.cu -o prog_new
