// ######################################################################### //
// ### LinuxGamepad.cpp #################################################### //
// ### implements LinuxGamepad.h                                         ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "LinuxGamepad.h"

#ifdef LAZYENGINE_PLATFORM_WINDOWS

// LinuxGamepad is typedef'd to be a GLFWGamepad
#include "../GLFW/GLFWGamepad.cpp"

#endif