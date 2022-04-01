# This file is not part of the original GLFW source code.
# Instead, it was entirely created for the usage in LazyEngine.
# Tested only on Windows.

# set include dirs
set(GLFW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")

# set source files
set(GLFW_SRC
"src/glfw_config.h"
"src/context.c"
"src/init.c"
"src/input.c"
"src/monitor.c"
"src/vulkan.c"
"src/window.c")

if(MSVC)
	set(GLFW_SRC
	${GLFW_SRC}
	"src/win32_init.c"
	"src/win32_joystick.c"
	"src/win32_monitor.c"
	"src/win32_time.c"
	"src/win32_thread.c"
	"src/win32_window.c"
	"src/wgl_context.c"
	"src/egl_context.c"
	"src/osmesa_context.c")
endif()

# Source file extensions do not have to be explicit
cmake_policy(SET CMP0115 OLD)

add_library(GLFW STATIC ${GLFW_SRC})
target_include_directories(GLFW PUBLIC ${GLFW_INCLUDE_DIRS})
target_compile_definitions(GLFW PUBLIC "_GLFW_WIN32")

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GLFW_SRC})