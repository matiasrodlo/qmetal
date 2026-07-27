CXX        := clang++
CXXFLAGS   := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation

TOOLS := bw_probe reuse_probe zerocheck scale fusion_bench ablation

.PHONY: all clean

all: $(TOOLS)

CORE := ../src/reference.cpp ../src/circuits.cpp ../src/pauli.cpp ../src/plan.cpp \
        ../src/metal/simulator.mm

bw_probe reuse_probe zerocheck: %: %.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

# These drive the real simulator, so they link the library sources.
scale fusion_bench ablation: %: %.mm $(CORE)
	$(MAKE) -C .. build/kernels_source.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -I../include -I../build $^ -o $@

clean:
	rm -rf $(TOOLS) *.dSYM
