# This file is not part of the original ImGui code base.
# Instead, it was created specifically for the LazyEngine project.

# set the include directories of this lib
set(ImGui_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/"
)

set(ImGui_SRC
	"imconfig.h"
	"imgui.h"
	"imgui.cpp"
	"imgui_draw.cpp"
	"imgui_internal.h"
	"imgui_widgets.cpp"
	"imstb_rectpack.h"
	"imstb_textedit.h"
	"imstb_truetype.h"
	"imgui_demo.cpp"
	"imgui-filebrowser/imfilebrowser.h"
)

add_library(ImGui STATIC ${ImGui_SRC})
target_include_directories(ImGui PUBLIC ${ImGui_INCLUDE_DIRS})

# group the source files in the same way in Visual Studio as on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ImGui_SRC})