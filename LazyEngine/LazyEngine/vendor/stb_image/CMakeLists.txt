# This file is not part of the original stb-image code base
# Instead it was created specifically for the usage in LazyEngine.

set(stb_image_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/")

set(stb_image_SRC
	"stb_image.cpp"
	"stb_image.h"
)

add_library(stb_image ${stb_image_SRC})
target_include_directories(stb_image PUBLIC ${stb_image_INCLUDE_DIRS})

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${stb_image_SRC})