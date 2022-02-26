# This file is not part of the original spdlog codebase.
# Instead, it was created specifically for the LazyEngine project.
# The original CMakeLists.txt was renamed to CMakeLists.txt.orig

set(spdlog_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")

add_library(spdlog INTERFACE)
target_include_directories(spdlog INTERFACE ${spdlog_INCLUDE_DIRS})