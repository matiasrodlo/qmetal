CXX      := clang++
CXXFLAGS := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation

.PHONY: all clean run

all: bw_probe

bw_probe: bw_probe.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

# Default sweep goes to 33 qubits (64 GiB). Pass N=<qubits> to cap it lower.
N ?= 33
run: bw_probe
	./bw_probe $(N) | tee ../results/bw_probe_n$(N).csv

clean:
	rm -rf bw_probe bw_probe.dSYM
