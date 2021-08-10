CUDA ?= /usr/local/cuda-11.4
CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_INC := -I $(CUDA)/include

GDRAPI_INC := -I /packages/gdrcopy/include
GDRAPI_SRC := -L /packages/gdrcopy/src

test:
	gcc $(CUDA_INC) $(CUDA_LIB) $(GDRAPI_SRC) $(GDRAPI_INC) -o run test.c -lcudart -lcuda -lgdrapi

.PHONY : clean
clean:
	-rm -rf run
