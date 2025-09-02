# Unit tests
add_executable(unit_tests
    unit/test_bitboard.cpp
)

target_link_libraries(unit_tests
    GTest::gtest
    GTest::gtest_main
)

# Add test to CTest
add_test(NAME BitBoardTests COMMAND unit_tests)