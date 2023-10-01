cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(GeneticAlgorithm)

# Set the output directory for the executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Add the executable target
add_executable(ga src/main.cpp src/ToolsHelper.cpp src/Tests.cpp src/MLAlgorithms.cpp)

# You can set C++ standards if needed
set(CMAKE_CXX_STANDARD 20)  # Change to the desired C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2") # Use -O2 optimization

# Optionally, if you're using the displayData function, include the header file.
# target_include_directories(my_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# You may need to find packages or link libraries if necessary.
# For example, to find and use the Boost library:
# find_package(Boost REQUIRED)
# target_link_libraries(my_program PRIVATE Boost::Boost)

