#pragma once

// ######################################################################### //
// ### LinuxGamepad.h ###################################################### //
// ### implements the Gamepad class for Linux using glfw3.              ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_LINUX

#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWGamepad.h"

namespace LazyEngine {
	typedef GLFWGamepad LinuxGamepad;
}
#endif