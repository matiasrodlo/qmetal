CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude -Ibuild
OBJCXX   := -fobjc-arc
FRAMEWORKS := -framework Metal -framework Foundation

CORE   := src/reference.cpp src/circuits.cpp
GPU    := src/metal/simulator.mm
GENHDR := build/kernels_source.h

.PHONY: all test test-cpu test-gpu bench clean

all: test

test: test-cpu test-gpu

test-cpu: build/test_reference
	./build/test_reference

test-gpu: build/test_gpu
	./build/test_gpu

build/test_reference: $(CORE) tests/test_reference.cpp | build
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
