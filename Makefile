CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude -Ibuild
OBJCXX   := -fobjc-arc
FRAMEWORKS := -framework Metal -framework Foundation

CORE   := src/reference.cpp src/circuits.cpp src/pauli.cpp src/plan.cpp
GPU    := src/metal/simulator.mm
GENHDR := build/kernels_source.h

.PHONY: all test test-cpu test-fusion test-gpu test-python lib bench clean

all: test

test: test-cpu test-fusion test-gpu test-python

test-cpu: build/test_reference
	./build/test_reference

test-fusion: build/test_fusion
	./build/test_fusion

test-gpu: build/test_gpu
	./build/test_gpu

# Shared library exposing the C ABI. The Python bindings load this with ctypes,
# so they need no compiler and no binding framework.
lib: build/libqmetal.dylib

build/libqmetal.dylib: $(CORE) src/capi.cpp $(GPU) $(GENHDR) | build
	$(CXX) $(CXXFLAGS) $(OBJCXX) $(FRAMEWORKS) -dynamiclib \
	    -install_name @rpath/libqmetal.dylib \
	    $(CORE) src/capi.cpp $(GPU) -o $@

test-python: build/libqmetal.dylib
	QMETAL_LIBRARY=build/libqmetal.dylib python3 tests/test_python.py

build/test_reference: $(CORE) tests/test_reference.cpp | build
	$(CXX) $(CXXFLAGS) $^ -o $@

build/test_fusion: $(CORE) tests/test_fusion.cpp | build
	$(CXX) $(CXXFLAGS) $^ -o $@

build/test_gpu: $(CORE) $(GPU) tests/test_gpu.mm $(GENHDR) | build
	$(CXX) $(CXXFLAGS) $(OBJCXX) $(FRAMEWORKS) $(CORE) $(GPU) tests/test_gpu.mm -o $@

# Embed the Metal source so the binary is self-contained: no runtime file
# lookup, and the shader stays a real .metal file for editor and compiler use.
$(GENHDR): src/metal/kernels.metal | build
	@printf '// Generated from src/metal/kernels.metal -- do not edit.\n' > $@
	@printf 'static const char *kKernelSource = R"METALSRC(\n' >> $@
	@cat $< >> $@
	@printf ')METALSRC";\n' >> $@

build:
	@mkdir -p build

bench:
	$(MAKE) -C bench

clean:
	rm -rf build
	$(MAKE) -C bench clean
