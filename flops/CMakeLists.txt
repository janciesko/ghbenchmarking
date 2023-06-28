#Contact: jciesko@sandia.gov

cmake_minimum_required (VERSION 3.12)

project(kokkos-bench
  LANGUAGES CXX
  VERSION 1.0.0)

FILE(GLOB TEST_SRCS *.cpp)
FILE(GLOB TEST_C_SRCS *.cc)

LIST(APPEND TEST_SRCS ${TEST_C_SRCS})

message(STATUS ${TEST_C_SRCS})

find_package(Kokkos REQUIRED)

foreach(test ${TEST_SRCS})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} PRIVATE Kokkos::kokkos)
endforeach()
