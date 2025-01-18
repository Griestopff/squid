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
include lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/progress.make

# Include the compile flags for this target's objects.
include lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/flags.make

lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o: lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/flags.make
lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o: ../lib/fizzbuzz/src/fizzbuzz.cpp
lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o: lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o"
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o -MF CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o.d -o CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o -c /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/lib/fizzbuzz/src/fizzbuzz.cpp

lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.i"
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/lib/fizzbuzz/src/fizzbuzz.cpp > CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.i

lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.s"
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/lib/fizzbuzz/src/fizzbuzz.cpp -o CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.s

# Object files for target fizzbuzz
fizzbuzz_OBJECTS = \
"CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o"

# External object files for target fizzbuzz
fizzbuzz_EXTERNAL_OBJECTS =

lib/fizzbuzz/libfizzbuzz.a: lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/src/fizzbuzz.cpp.o
lib/fizzbuzz/libfizzbuzz.a: lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/build.make
lib/fizzbuzz/libfizzbuzz.a: lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libfizzbuzz.a"
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && $(CMAKE_COMMAND) -P CMakeFiles/fizzbuzz.dir/cmake_clean_target.cmake
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fizzbuzz.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/build: lib/fizzbuzz/libfizzbuzz.a
.PHONY : lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/build

lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/clean:
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz && $(CMAKE_COMMAND) -P CMakeFiles/fizzbuzz.dir/cmake_clean.cmake
.PHONY : lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/clean

lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/depend:
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2 /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/lib/fizzbuzz /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/exam2/build/lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/fizzbuzz/CMakeFiles/fizzbuzz.dir/depend

