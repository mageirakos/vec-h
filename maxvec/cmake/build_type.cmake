# Uses the Release build type by default.
set(default_build_type ${CMAKE_BUILD_TYPE})
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(default_build_type "Release")
endif()
set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
