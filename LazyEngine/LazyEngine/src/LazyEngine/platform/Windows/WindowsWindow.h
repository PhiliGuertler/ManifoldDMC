#pragma once

// ######################################################################### //
// ### WindowsWindow.h ##################################################### //
// ### A Window class using GLFW3 on Windows                             ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_WINDOWS

#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWWindow.h"

namespace LazyEngine {
	typedef GLFWWindow WindowsWindow;
}

#endif