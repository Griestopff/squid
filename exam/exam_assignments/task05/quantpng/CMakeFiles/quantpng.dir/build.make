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
CMAKE_SOURCE_DIR = /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng

# Include any dependencies generated for this target.
include CMakeFiles/quantpng.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/quantpng.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/quantpng.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/quantpng.dir/flags.make

CMakeFiles/quantpng.dir/main.cpp.o: CMakeFiles/quantpng.dir/flags.make
CMakeFiles/quantpng.dir/main.cpp.o: main.cpp
CMakeFiles/quantpng.dir/main.cpp.o: CMakeFiles/quantpng.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/quantpng.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/quantpng.dir/main.cpp.o -MF CMakeFiles/quantpng.dir/main.cpp.o.d -o CMakeFiles/quantpng.dir/main.cpp.o -c /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/main.cpp

CMakeFiles/quantpng.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quantpng.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/main.cpp > CMakeFiles/quantpng.dir/main.cpp.i

CMakeFiles/quantpng.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quantpng.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/main.cpp -o CMakeFiles/quantpng.dir/main.cpp.s

# Object files for target quantpng
quantpng_OBJECTS = \
"CMakeFiles/quantpng.dir/main.cpp.o"

# External object files for target quantpng
quantpng_EXTERNAL_OBJECTS =

quantpng: CMakeFiles/quantpng.dir/main.cpp.o
quantpng: CMakeFiles/quantpng.dir/build.make
quantpng: libs/lodepng/liblodepng.a
quantpng: libs/quantize/libquantize.a
quantpng: CMakeFiles/quantpng.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable quantpng"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quantpng.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/quantpng.dir/build: quantpng
.PHONY : CMakeFiles/quantpng.dir/build

CMakeFiles/quantpng.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/quantpng.dir/cmake_clean.cmake
.PHONY : CMakeFiles/quantpng.dir/clean

CMakeFiles/quantpng.dir/depend:
	cd /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng /home/christoph/Dokumente/Studium/Semester_5/algo_eng/05/task05/quantpng/CMakeFiles/quantpng.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/quantpng.dir/depend

