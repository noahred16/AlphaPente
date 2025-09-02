.PHONY: all clean build test benchmark install deps

BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests

all: build

deps:
	conan install . --output-folder=$(BUILD_DIR) --build=missing

build: deps
	cmake --preset conan-default
	cmake --build $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

test: build
	cd $(BUILD_DIR) && ./tests

benchmark: build
	cd $(BUILD_DIR) && ./benchmarks

install: build
	cmake --install $(BUILD_DIR)

debug:
	cmake -DCMAKE_BUILD_TYPE=Debug --preset conan-default
	cmake --build $(BUILD_DIR)

release:
	cmake -DCMAKE_BUILD_TYPE=Release --preset conan-default
	cmake --build $(BUILD_DIR)