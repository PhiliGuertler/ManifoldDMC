# This File is not part of the original glm code base.
# Instead it was created specificly for the LazyEngine project.
# The original CMakeLists.txt was renamed to CMakeLists.txt.orig

set(glm_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/")

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${glm_INCLUDE_DIRS})