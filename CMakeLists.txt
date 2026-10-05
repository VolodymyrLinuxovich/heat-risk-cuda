# Builds the C++ CPU implementation (src/heat_risk/native) as a shared library and runs its native
# unit tests. Python does not need this: heat_risk.cpu_cpp compiles the same sources on first use.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build

cmake_minimum_required(VERSION 3.16)
project(heat_hazards_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(NATIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/heat_risk/native)

# No fused multiply-add and no fast math, so float32 results match NumPy bit for bit.
set(HEAT_FP_FLAGS -ffp-contract=off -fno-fast-math)

add_library(heat_hazards_cpp SHARED ${NATIVE_DIR}/heat_hazards.cpp)
target_include_directories(heat_hazards_cpp PUBLIC ${NATIVE_DIR})
target_compile_options(heat_hazards_cpp PRIVATE ${HEAT_FP_FLAGS} -Wall -Wextra)
target_link_libraries(heat_hazards_cpp PRIVATE Threads::Threads)

enable_testing()
add_executable(test_heat_hazards native/tests/test_heat_hazards.cpp)
target_include_directories(test_heat_hazards PRIVATE ${NATIVE_DIR})
target_compile_options(test_heat_hazards PRIVATE ${HEAT_FP_FLAGS} -Wall -Wextra)
target_link_libraries(test_heat_hazards PRIVATE Threads::Threads)
add_test(NAME native_heat_hazards COMMAND test_heat_hazards)
