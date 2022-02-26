#pragma once

// ######################################################################### //
// ### LinuxInput.h ######################################################## //
// ### implements the Input class for Linux using glfw.                  ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_LINUX

#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWInput.h"

namespace LazyEngine {
	typedef GLFWInput LinuxInput;
}

#endif