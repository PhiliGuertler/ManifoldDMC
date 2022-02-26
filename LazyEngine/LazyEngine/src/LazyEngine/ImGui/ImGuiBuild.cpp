#include "LazyEngine/gepch.h"

// ######################################################################### //
// ### ImGuiBuild.cpp ###################################################### //
// ### This file will build the ImGui Implementations for glfw and       ### //
// ### OpenGL                                                            ### //
// ######################################################################### //

#define IMGUI_IMPL_OPENGL_LOADER_GLAD
#include "examples/imgui_impl_glfw.cpp"
#include "examples/imgui_impl_opengl3.cpp"