# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build

# Include any dependencies generated for this target.
include CMakeFiles/fizzbuzz_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/fizzbuzz_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/fizzbuzz_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fizzbuzz_test.dir/flags.make

CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o: CMakeFiles/fizzbuzz_test.dir/flags.make
CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o: ../catch_main.cpp
CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o: CMakeFiles/fizzbuzz_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o -MF CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o.d -o CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o -c /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_main.cpp

CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_main.cpp > CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.i

CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_main.cpp -o CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.s

CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o: CMakeFiles/fizzbuzz_test.dir/flags.make
CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o: ../catch_tests_fizzbuzz.cpp
CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o: CMakeFiles/fizzbuzz_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o -MF CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o.d -o CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o -c /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_tests_fizzbuzz.cpp

CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_tests_fizzbuzz.cpp > CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.i

CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/catch_tests_fizzbuzz.cpp -o CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.s

# Object files for target fizzbuzz_test
fizzbuzz_test_OBJECTS = \
"CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o" \
"CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o"

# External object files for target fizzbuzz_test
fizzbuzz_test_EXTERNAL_OBJECTS =

fizzbuzz_test: CMakeFiles/fizzbuzz_test.dir/catch_main.cpp.o
fizzbuzz_test: CMakeFiles/fizzbuzz_test.dir/catch_tests_fizzbuzz.cpp.o
fizzbuzz_test: CMakeFiles/fizzbuzz_test.dir/build.make
fizzbuzz_test: lib/fizzbuzz/libfizzbuzz.a
fizzbuzz_test: CMakeFiles/fizzbuzz_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable fizzbuzz_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fizzbuzz_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fizzbuzz_test.dir/build: fizzbuzz_test
.PHONY : CMakeFiles/fizzbuzz_test.dir/build

CMakeFiles/fizzbuzz_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fizzbuzz_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fizzbuzz_test.dir/clean

CMakeFiles/fizzbuzz_test.dir/depend:
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2 /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2 /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles/fizzbuzz_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fizzbuzz_test.dir/depend

