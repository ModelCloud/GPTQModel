# ttbfp - CPU emulation kernels for Tenstorrent BFP8_b / BFP4_b
CXX      ?= g++
NVCC     ?= nvcc
PYTHON   ?= python3
CXXFLAGS ?= -O3 -std=c++17 -fPIC -fopenmp -Wall -Wextra -Iinclude
LDFLAGS  ?= -shared -fopenmp

LIB := libttbfp.so
CUDA_LIB := libttbfp_cuda.so
SRC := src/ttbfp_cpu.cpp
HDR := include/ttbfp.h

.PHONY: all lib cuda test bench clean

all: lib

lib: $(LIB)

$(LIB): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SRC)

cuda: $(CUDA_LIB)

$(CUDA_LIB): src/ttbfp_cuda.cu include/ttbfp_cuda.h $(HDR)
	$(NVCC) -O3 -std=c++17 -shared -Xcompiler -fPIC -Iinclude -o $@ src/ttbfp_cuda.cu

test: $(LIB)
	TTBFP_LIB=$(CURDIR)/$(LIB) PYTHONPATH=$(CURDIR)/python \
		$(PYTHON) -m pytest -q tests

bench: $(LIB)
	TTBFP_LIB=$(CURDIR)/$(LIB) PYTHONPATH=$(CURDIR)/python \
		$(PYTHON) bench/bench_matmul.py

clean:
	rm -f $(LIB) $(CUDA_LIB)
