cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(FIZZBUZZ LANGUAGES CXX)

# embed library picomp in project test_pi
add_subdirectory(lib/fizzbuzz)

# catch is a header only library for testing
# embed library catch in project test_pi
add_subdirectory(lib/catch)

# create executable with all tests
add_executable(fizzbuzz_test catch_main.cpp catch_tests_fizzbuzz.cpp)

# link libraries picomp and catch to executable
target_link_libraries(fizzbuzz_test PRIVATE fizzbuzz PRIVATE catch)
