# Unit tests
add_executable(unit_tests
    unit/test_bitboard.cpp
    unit/test_win_detection.cpp
    unit/test_capture_detection.cpp
    unit/test_distance_rings.cpp
    unit/test_move_generator.cpp
    unit/test_mcts_node.cpp
    unit/test_rollout.cpp
    unit/test_mcts_engine.cpp
)

target_link_libraries(unit_tests
    alphapente_core
    GTest::gtest
    GTest::gtest_main
)

# Use GTest discovery to automatically find all test cases
include(GoogleTest)
gtest_discover_tests(unit_tests)