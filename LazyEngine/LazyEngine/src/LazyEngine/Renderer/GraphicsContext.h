#pragma once

// ######################################################################### //
// ### GraphicsContext.h ################################################### //
// ### An interface class for all Graphics Contexts, like OpenGL,        ### //
// ### Vulcan, DirectX, Metal, etc.                                      ### //
// ######################################################################### //

namespace LazyEngine {

	/**
	 *	A representation of the graphics context
	 */
	class GraphicsContext {
	public:
		/**
		 *	initializes graphics context.
		 */
		virtual void init() = 0;
		/**
		 *	swaps the buffer used by the graphical output.
		 */
		virtual void swapBuffers() = 0;
	};

}