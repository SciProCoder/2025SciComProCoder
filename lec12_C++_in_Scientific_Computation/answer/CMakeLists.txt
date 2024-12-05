add_executable(01_unit_test 01-unit_test.cpp)
add_executable(02_power_method 02-power_method.cpp)

set(TARGETS
    01_unit_test
    02_power_method
)

foreach(TARGET ${TARGETS})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${TARGET} PRIVATE ${MKL_INCLUDE_DIRS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(${TARGET} PRIVATE ${OpenMP_CXX_LIBRARIES} mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl)
endforeach()