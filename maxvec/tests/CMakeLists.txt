
add_custom_target(tests COMMENT "Builds all the tests.")

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(GTestModule)
endif()

# Unit tests
function(add_maximus_test test_name)
    set(tgt_ test.${test_name})
    add_executable(${tgt_} ${test_name}.cpp)
    target_link_libraries(${tgt_} PUBLIC GTest::gtest GTest::gtest_main ${ARGN})
    target_include_directories(${tgt_} PUBLIC . )
    add_test(NAME ${tgt_}
             WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
             COMMAND ./${tgt_})
    add_dependencies(tests ${tgt_})
    install(TARGETS ${tgt_} DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()

set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR} CACHE INTERNAL "The source directory")
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_maximus_test(acero maximus)
add_maximus_test(sql maximus)
add_maximus_test(operators maximus)
add_maximus_test(tpch maximus)

if (NOT MAXIMUS_WITH_FAISS STREQUAL "OFF")
    add_maximus_test(faiss maximus)
    add_maximus_test(vsds maximus)
    add_maximus_test(list_type_support maximus)
    if (MAXIMUS_WITH_FAISS STREQUAL "GPUCUVS")
        add_maximus_test(faiss_gpu maximus)
        add_maximus_test(vsds_gpu maximus)
        target_include_directories(test.vsds_gpu PUBLIC ${CMAKE_SOURCE_DIR}/benchmarks)
    endif()
endif()

if (MAXIMUS_WITH_CUDA)
    add_maximus_test(cuda maximus)
    add_maximus_test(tpch_gpu maximus)
endif()

