// ######################################################################### //
// ### OpenGLContext.cpp ################################################### //
// ### Implements OpenGLContext.h                                        ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "OpenGLContext.h"

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	OpenGLContext::OpenGLContext(GLFWwindow *windowHandle)
		: m_windowHandle(windowHandle)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		LAZYENGINE_CORE_ASSERT(windowHandle, "Window handle is null!");
	}

	void OpenGLContext::init() {
		LAZYENGINE_PROFILE_FUNCTION();

		glfwMakeContextCurrent(m_windowHandle);
		int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		LAZYENGINE_CORE_ASSERT(status, "Failed to initialize \"glad\"");

		LAZYENGINE_CORE_INFO("OpenGL Info:");
		LAZYENGINE_CORE_INFO("  Vendor:   {0}", glGetString(GL_VENDOR));
		LAZYENGINE_CORE_INFO("  Renderer: {0}", glGetString(GL_RENDERER));
		LAZYENGINE_CORE_INFO("  Version:  {0}", glGetString(GL_VERSION));

		#ifdef LAZYENGINE_ENABLE_ASSERTS
			int versionMajor;
			int versionMinor;
			glGetIntegerv(GL_MAJOR_VERSION, &versionMajor);
			glGetIntegerv(GL_MINOR_VERSION, &versionMinor);
			LAZYENGINE_CORE_ASSERT(versionMajor > 4 || (versionMajor == 4 && versionMinor >= 5), "LazyEngine requires at least OpenGL version 4.5!");
		#endif
	}

	void OpenGLContext::swapBuffers() {
		LAZYENGINE_PROFILE_FUNCTION();

		glfwSwapBuffers(m_windowHandle);
	}
}