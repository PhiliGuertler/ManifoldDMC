# define this library's the name as "GLAD"
set(Glad_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set(Glad_SRC "src/glad.c")

add_library(Glad STATIC ${Glad_SRC})
target_include_directories(Glad PUBLIC ${Glad_INCLUDE_DIRS})

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${Glad_SRC})