CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude

SRC  := src/reference.cpp src/circuits.cpp
TEST := tests/test_reference.cpp

.PHONY: all test bench clean

all: test

test: build/test_reference
	./build/test_reference

build/test_reference: $(SRC) $(TEST) | build
	$(CXX) $(CXXFLAGS) $(SRC) $(TEST) -o $@

build:
	@mkdir -p build

bench:
	$(MAKE) -C bench

clean:
	rm -rf build
	$(MAKE) -C bench clean
