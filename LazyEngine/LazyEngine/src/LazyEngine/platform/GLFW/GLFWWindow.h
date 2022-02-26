#pragma once

// ######################################################################### //
// ### GLFWWindow.h ######################################################## //
// ### A Window class using GLFW3                                        ### //
// ######################################################################### //
//#define LAZYENGINE_PLATFORM_GLFW3

#ifdef LAZYENGINE_PLATFORM_GLFW3

#include "LazyEngine/Core/Window.h"
#include "LazyEngine/Renderer/GraphicsContext.h"

#include <GLFW/glfw3.h>

#include "glm/glm.hpp"

namespace LazyEngine {

	/**
	 *	An implementation of a window on Windows using glfw with opengl.
	 */
	class GLFWWindow : public Window {
	public:
		/**
		 *	constructor
		 *	creates a Window for the Windows platform
		 *	@param props: an instance of WindowProperties containing width, height and name of the window
		 *		that will be created
		 */
		GLFWWindow(const WindowProperties& props);
		/**
		 *	destructor
		 */
		virtual ~GLFWWindow();

		/**
		 *	update the contents of the window, e.g. by a buffer swap.
		 */
		void onUpdate() override;

		/**
		 *	returns width of this window in pixels
		 */
		inline virtual unsigned int getWidth() const override { return m_data.size.x; }
		/**
		 *	returns height of this window in pixels
		 */
		inline virtual unsigned int getHeight() const override { return m_data.size.y; }
		/**
		 *	returns width and height of this window in pixels
		 */
		inline virtual glm::ivec2 getSize() const override { return m_data.size; }

		virtual void setSize(const glm::ivec2& size) override;

		// --- Window attributes ----------------------------------------------
		/**
		 *	the function to be called on a event can be set here during runtime.
		 */
		inline void setEventCallback(const EventCallbackFn& callback) override { m_data.eventCallback = callback; }
		/**
		 *	enable or disable vsync
		 *	@param enabled: true if vsync should be enabled, false otherwise
		 */
		void setVSync(bool enabled) override;
		/**
		 *	returns true if vsync is turned on
		 */
		bool isVSync() const override { return m_data.vSyncEnabled; }
		/**
		 *	returns the amount of Dots-Per-Inch of the monitor this window is being displayed on.
		 */
		glm::ivec2 getDPI() const override;

		/**
		 *	Grabs or Ungrabs the cursor, which means it will be invisible and be able to move indefinitely.
		 *	@param hideIt: if true, the cursor will be grabbed from now on, otherwise it will be released
		 */
		virtual void grabMouseCursor(bool hideIt) override;

		/**
		 *	Sets the window to be fullscreen, borderless or windowed.
		 *	@param mode: one of Windowed, Fullscreen or Borderless
		 */
		virtual void setFullscreenMode(FullscreenMode mode) override;

		/**
		 *	Returns the current fullscreen mode
		 */
		virtual FullscreenMode getFullscreenMode() const override;

		/**
		 *	Sets the window title
		 *	@param title: the new title of the window (in UTF-8 encoding)
		 */
		virtual void setTitle(const std::string& title) override;

		/**
		 *	returns the current position of the top left corner in pixel coordinates
		 */
		virtual inline  const glm::ivec2 getPosition() const override { return m_data.position; }
		/**
		 *	sets the position of the top left corner in pixel coordinates
		 *	@param position: the new position of the top left corner
		 */
		virtual void setPosition(const glm::ivec2& position) override;

		/**
		 *	minimizes the window
		 */
		virtual void minimize() override;
		/**
		 *	returns whether the window is minimized or not
		 */
		virtual bool isMinimized() const override;

		/**
		 *	returns the platform specific window, in this case a glfwWindow*.
		 */
		inline virtual void* getNativeWindow() const override { return m_window; }

	private:
		/**
		 *	initializes this window with props
		 */
		virtual void init(const WindowProperties& props);
		/**
		 *	shuts down the window
		 */
		virtual void shutdown();

	private:
		// the underlying GLFW window
		GLFWwindow* m_window;
		// the graphicscontext responsible of buffer swaps.
		GraphicsContext* m_context;

		// data that might be used by glfw
		struct WindowData {
			std::string title = "LazyEngine Window";
			bool vSyncEnabled = true;

			FullscreenMode fullscreenMode = FullscreenMode::Windowed;

			glm::ivec2 size = { 0,0 };
			glm::ivec2 lastWindowedSize = { 0,0 };
			glm::ivec2 position = { 0,0 };

			EventCallbackFn eventCallback;
		};
		WindowData m_data;
	};
}
#endif