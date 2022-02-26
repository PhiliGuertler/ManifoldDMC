#pragma once

// ######################################################################### //
// ### OpenGLContext.h ##################################################### //
// ### An implementation of GraphicsContext using OpenGL                 ### //
// ######################################################################### //

#include "LazyEngine/Renderer/GraphicsContext.h"

struct GLFWwindow;

namespace LazyEngine {

	/**
	 *	A representation of the OpenGL graphics context
	 */
	class OpenGLContext : public GraphicsContext {
	public:
		/**
		 *	constructor
		 */
		OpenGLContext(GLFWwindow *windowHandle);

		/**
		 *	initializes graphics context.
		 */
		virtual void init() override;
		/**
		 *	swaps the buffer used by the graphical output.
		 */
		virtual void swapBuffers() override;
	private:
		// handle to the window
		GLFWwindow *m_windowHandle;
	};

}