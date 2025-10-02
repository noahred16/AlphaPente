.PHONY: build test test-case test-integration clean benchmark

build:
	@echo "Installing dependencies..."
	@conan install . --output-folder=build --build=missing > /dev/null 2>&1
	@echo "Configuring build..."
	@cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
	@echo "Building..."
	@cd build && cmake --build . > /dev/null 2>&1
	@echo "Build complete."

test: build
	@echo "Running all tests..."
	@cd build && ctest --output-on-failure

test-case: build
ifeq ($(CASE),)
	@echo "Usage: make test-case CASE=TestSuite.TestName"
	@echo "Examples:"
	@echo "  make test-case CASE=MCTSNodeTest.*"
	@echo "  make test-case CASE=*Performance*"
	@echo "  make test-case CASE=MCTSNodeTest.UCB1CalculationWithExploration"
else
	@echo "Running test case: $(CASE)"
	@cd build && ./tests/unit_tests --gtest_filter="$(CASE)"
endif

test-integration: build
ifeq ($(CASE),)
	@echo "Usage: make test-integration CASE=TestSuite.TestName"
	@echo "Examples:"
	@echo "  make test-integration CASE=IsAgentDumbTest.*"
	@echo "  make test-integration CASE=CompleteGamesTest.*"
	@echo "  make test-integration CASE=IsAgentDumbTest.AgentShouldPlayNearExistingStones"
else
	@echo "Running integration test case: $(CASE)"
	@cd build && ./tests/integration_tests --gtest_filter="$(CASE)"
endif

benchmark: build
	cd build && ./benchmark

clean:
	rm -rf build/

run: build
	cd build && ./alphapente

debug-mcts: build
	cd build && ./debug_mcts_tree