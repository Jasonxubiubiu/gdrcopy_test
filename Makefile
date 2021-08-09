CUDA ?= /usr/local/cuda-11.4
CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_INC += -I $(CUDA)/include

LDFLAGS  := $(CUDA_LIB) -L $(CUDA)/lib64 -L $(GDRAPI_SRC)

COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lcuda -lpthread -ldl

test:
	gcc $(CUDA_INC) $(CUDA_LIB) -o run test.c -lcudart -lcuda

