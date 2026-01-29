# Unit Tests

Uses [doctest](https://github.com/doctest/doctest) - a lightweight header-only testing framework.

## Running Tests

```bash
cd build
make unit_tests && ./unit_tests
```

## Command Line Flags

| Flag | Description |
|------|-------------|
| `-s`, `--success` | Show successful assertions (verbose output) |
| `-tc`, `--test-case=<filter>` | Run tests matching name pattern |
| `-sf`, `--source-file=<filter>` | Run tests from files matching pattern |
| `-ltc`, `--list-test-cases` | List all test case names |
| `-c`, `--count` | Print number of matching tests |
| `-d`, `--duration` | Show time duration of each test |

## Examples

```bash
# Run all tests
./unit_tests

# Verbose output (show each assertion)
./unit_tests -s

# Run only BitBoard tests
./unit_tests -tc="BitBoard*"

# Run only PenteGame tests
./unit_tests -tc="PenteGame*"

# Run tests from a specific file
./unit_tests -sf="*BitBoardTests*"

# List all available tests
./unit_tests -ltc

# Show test durations
./unit_tests -d
```

## Adding New Tests

1. Add tests to an existing file, or create a new `tests/YourTests.cpp`
2. If creating a new file, add it to `CMakeLists.txt`:
   ```cmake
   add_executable(unit_tests
       tests/BitBoardTests.cpp
       tests/PenteGameTests.cpp
       tests/YourTests.cpp  # Add here
   )
   ```
3. Only `BitBoardTests.cpp` should have `#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`

## Writing Tests

```cpp
#include "doctest.h"
#include "YourClass.hpp"

TEST_CASE("description of what you're testing") {
    // Setup
    YourClass obj;

    // Assertions
    CHECK(obj.value() == expected);      // Continues on failure
    REQUIRE(obj.isValid());              // Stops test on failure
    CHECK_FALSE(obj.isEmpty());          // Expects false
    CHECK_EQ(obj.count(), 5);            // Better error messages
    CHECK_THROWS(obj.badMethod());       // Expects exception
}
```
