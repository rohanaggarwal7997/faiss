# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/rohaagga/Desktop/Faiss/faiss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/rohaagga/Desktop/Faiss/faiss/build

# Include any dependencies generated for this target.
include demos/CMakeFiles/demo_ivfpq_indexing.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demos/CMakeFiles/demo_ivfpq_indexing.dir/compiler_depend.make

# Include the progress variables for this target.
include demos/CMakeFiles/demo_ivfpq_indexing.dir/progress.make

# Include the compile flags for this target's objects.
include demos/CMakeFiles/demo_ivfpq_indexing.dir/flags.make

demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o: demos/CMakeFiles/demo_ivfpq_indexing.dir/flags.make
demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o: /Users/rohaagga/Desktop/Faiss/faiss/demos/demo_ivfpq_indexing.cpp
demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o: demos/CMakeFiles/demo_ivfpq_indexing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/rohaagga/Desktop/Faiss/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o"
	cd /Users/rohaagga/Desktop/Faiss/faiss/build/demos && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o -MF CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o.d -o CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o -c /Users/rohaagga/Desktop/Faiss/faiss/demos/demo_ivfpq_indexing.cpp

demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.i"
	cd /Users/rohaagga/Desktop/Faiss/faiss/build/demos && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rohaagga/Desktop/Faiss/faiss/demos/demo_ivfpq_indexing.cpp > CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.i

demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.s"
	cd /Users/rohaagga/Desktop/Faiss/faiss/build/demos && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rohaagga/Desktop/Faiss/faiss/demos/demo_ivfpq_indexing.cpp -o CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.s

# Object files for target demo_ivfpq_indexing
demo_ivfpq_indexing_OBJECTS = \
"CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o"

# External object files for target demo_ivfpq_indexing
demo_ivfpq_indexing_EXTERNAL_OBJECTS =

demos/demo_ivfpq_indexing: demos/CMakeFiles/demo_ivfpq_indexing.dir/demo_ivfpq_indexing.cpp.o
demos/demo_ivfpq_indexing: demos/CMakeFiles/demo_ivfpq_indexing.dir/build.make
demos/demo_ivfpq_indexing: faiss/libfaiss.a
demos/demo_ivfpq_indexing: /opt/local/lib/libomp.dylib
demos/demo_ivfpq_indexing: demos/CMakeFiles/demo_ivfpq_indexing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/rohaagga/Desktop/Faiss/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_ivfpq_indexing"
	cd /Users/rohaagga/Desktop/Faiss/faiss/build/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_ivfpq_indexing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/CMakeFiles/demo_ivfpq_indexing.dir/build: demos/demo_ivfpq_indexing
.PHONY : demos/CMakeFiles/demo_ivfpq_indexing.dir/build

demos/CMakeFiles/demo_ivfpq_indexing.dir/clean:
	cd /Users/rohaagga/Desktop/Faiss/faiss/build/demos && $(CMAKE_COMMAND) -P CMakeFiles/demo_ivfpq_indexing.dir/cmake_clean.cmake
.PHONY : demos/CMakeFiles/demo_ivfpq_indexing.dir/clean

demos/CMakeFiles/demo_ivfpq_indexing.dir/depend:
	cd /Users/rohaagga/Desktop/Faiss/faiss/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/rohaagga/Desktop/Faiss/faiss /Users/rohaagga/Desktop/Faiss/faiss/demos /Users/rohaagga/Desktop/Faiss/faiss/build /Users/rohaagga/Desktop/Faiss/faiss/build/demos /Users/rohaagga/Desktop/Faiss/faiss/build/demos/CMakeFiles/demo_ivfpq_indexing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/CMakeFiles/demo_ivfpq_indexing.dir/depend

