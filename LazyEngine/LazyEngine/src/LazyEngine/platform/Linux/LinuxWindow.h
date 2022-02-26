#pragma once

// ######################################################################### //
// ### LinuxWindow.h ####################################################### //
// ### A Window class using GLFW3 on Linux                               ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_LINUX

#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWWindow.h"

namespace LazyEngine {
	typedef GLFWWindow LinuxWindow;
}

#endif