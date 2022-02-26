#pragma once

// ######################################################################### //
// ### WindowsInput.h ###################################################### //
// ### implements the Input class for Windows using glfw.                ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_WINDOWS

#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWInput.h"

namespace LazyEngine {
	typedef GLFWInput WindowsInput;
}

#endif