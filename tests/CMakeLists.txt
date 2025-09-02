# Unit tests
add_executable(unit_tests
    unit/test_bitboard.cpp
    unit/test_win_detection.cpp
    unit/test_capture_detection.cpp
    unit/test_distance_rings.cpp
    unit/test_move_generator.cpp
)

target_link_libraries(unit_tests
    alphapente_core
    GTest::gtest
    GTest::gtest_main
)

# Add test to CTest
add_test(NAME BitBoardTests COMMAND unit_tests)