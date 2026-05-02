set (executables "maxbench")

foreach (exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE maximus cxxopts::cxxopts)
    target_include_directories(${exec} PUBLIC "${maximus_SOURCE_DIR}/third_party/cxxopts/include/")
    install(TARGETS ${exec} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()

# Big Vector Bench
if (NOT MAXIMUS_WITH_FAISS STREQUAL "OFF")
    add_executable("bigVectorBench" "big-vector-bench/big_vector_bench.cpp" "big-vector-bench/queries.cpp" "big-vector-bench/utils.cpp")
    target_link_libraries("bigVectorBench" PRIVATE maximus cxxopts::cxxopts)
    target_include_directories("bigVectorBench" PUBLIC "${maximus_SOURCE_DIR}/third_party/cxxopts/include/")
    install(TARGETS "bigVectorBench" RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR} CACHE INTERNAL "The source directory")
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
