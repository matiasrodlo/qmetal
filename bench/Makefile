CXX        := clang++
CXXFLAGS   := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation

TOOLS := bw_probe reuse_probe zerocheck

.PHONY: all clean

all: $(TOOLS)

%: %.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

clean:
	rm -rf $(TOOLS) *.dSYM
