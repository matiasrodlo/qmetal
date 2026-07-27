CXX        := clang++
CXXFLAGS   := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation

TOOLS := bw_probe reuse_probe zerocheck scale

.PHONY: all clean

all: $(TOOLS)

CORE := ../src/reference.cpp ../src/circuits.cpp ../src/pauli.cpp \
        ../src/metal/simulator.mm

bw_probe reuse_probe zerocheck: %: %.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

# scale drives the real simulator, so it links the library sources.
scale: scale.mm $(CORE)
	$(MAKE) -C .. build/kernels_source.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -I../include -I../build $^ -o $@

clean:
	rm -rf $(TOOLS) *.dSYM
