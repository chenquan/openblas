# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2019.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2019.2\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Codes\yunqi\openblas

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Codes\yunqi\openblas\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/openblas.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/openblas.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/openblas.dir/flags.make

CMakeFiles/openblas.dir/main.cpp.obj: CMakeFiles/openblas.dir/flags.make
CMakeFiles/openblas.dir/main.cpp.obj: CMakeFiles/openblas.dir/includes_CXX.rsp
CMakeFiles/openblas.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\yunqi\openblas\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/openblas.dir/main.cpp.obj"
	"E:\Program Files\mingw64\bin\g++.exe"  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\openblas.dir\main.cpp.obj -c D:\Codes\yunqi\openblas\main.cpp

CMakeFiles/openblas.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openblas.dir/main.cpp.i"
	"E:\Program Files\mingw64\bin\g++.exe" $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\yunqi\openblas\main.cpp > CMakeFiles\openblas.dir\main.cpp.i

CMakeFiles/openblas.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openblas.dir/main.cpp.s"
	"E:\Program Files\mingw64\bin\g++.exe" $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Codes\yunqi\openblas\main.cpp -o CMakeFiles\openblas.dir\main.cpp.s

# Object files for target openblas
openblas_OBJECTS = \
"CMakeFiles/openblas.dir/main.cpp.obj"

# External object files for target openblas
openblas_EXTERNAL_OBJECTS =

openblas.exe: CMakeFiles/openblas.dir/main.cpp.obj
openblas.exe: CMakeFiles/openblas.dir/build.make
openblas.exe: CMakeFiles/openblas.dir/linklibs.rsp
openblas.exe: CMakeFiles/openblas.dir/objects1.rsp
openblas.exe: CMakeFiles/openblas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Codes\yunqi\openblas\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable openblas.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\openblas.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/openblas.dir/build: openblas.exe

.PHONY : CMakeFiles/openblas.dir/build

CMakeFiles/openblas.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\openblas.dir\cmake_clean.cmake
.PHONY : CMakeFiles/openblas.dir/clean

CMakeFiles/openblas.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\Codes\yunqi\openblas D:\Codes\yunqi\openblas D:\Codes\yunqi\openblas\cmake-build-debug D:\Codes\yunqi\openblas\cmake-build-debug D:\Codes\yunqi\openblas\cmake-build-debug\CMakeFiles\openblas.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/openblas.dir/depend

