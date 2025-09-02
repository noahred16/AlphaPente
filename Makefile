.PHONY: build test clean benchmark

build:
	conan install . --output-folder=build --build=missing
	cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
	cd build && cmake --build .

test: build
ifdef FILTER
	cd build && ./tests/unit_tests --gtest_filter="$(FILTER)"
else
	cd build && ctest --output-on-failure
endif

benchmark: build
	cd build && ./benchmark

clean:
	rm -rf build/

run: build
	cd build && ./alphapente