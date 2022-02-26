#pragma once

// ######################################################################### //
// ### Window.h ############################################################ //
// ### Defines a Window Superclass that is an abstraction of Platform-   ### //
// ### specific Windows. The extensions are located in platform/         ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Events/Event.h"

#include "glm/glm.hpp"

namespace LazyEngine {

	/**
	 *	Data Struct containing data nescessary for a window.
	 */
	struct WindowProperties {
		std::string title;
		glm::ivec2 size;
		glm::ivec2 position;

		WindowProperties(const std::string& title = "LazyEngine Window", unsigned int width = 1280, unsigned int height = 720, int xPosition = (1920-1280)/2, int yPosition = (1080-720)/2)
			: title(title)
			, size({ width, height })
			, position({ xPosition, yPosition })
		{
			// empty
		}
	};

	enum class FullscreenMode {
		Windowed,	// corresponds to a regular window
		Fullscreen,	// a fullscreen window that is always on top of the screen
		Borderless	// a kind-of fullscreen window that can be overlayed by other windows though
	};

	/**
	 *	Interface representing a desktop system based Window
	 *	to be implemented for each platform
	 */
	class LAZYENGINE_API Window {
	public:

		/**
		 *	default destructor
		 */
		virtual ~Window() = default;

		/**
		 *	update the contents of the window, e.g. by a buffer swap.
		 */
		virtual void onUpdate() = 0;

		/**
		 *	returns width of this window in pixels
		 */
		virtual unsigned int getWidth() const = 0;
		/**
		 *	returns height of this window in pixels
		 */
		virtual unsigned int getHeight() const = 0;
		/**
		 *	returns width and height of this window in pixels
		 */
		virtual glm::ivec2 getSize() const = 0;
		/**
		 *	sets the size of the window in pixels
		 *	@param size: the new size of the window
		 */
		virtual void setSize(const glm::ivec2& size) = 0;

		// --- Window attributes ----------------------------------------------
		/**
		 *	the function to be called on a event can be set here during runtime.
		 *	@param callback: the function to be called on an event. It must have
		 *		the signature "void func(Event& event)"
		 */
		virtual void setEventCallback(const EventCallbackFn& callback) = 0;
		/**
		 *	enable or disable vsync
		 *	@param enabled: true if vsync should be enabled, false otherwise
		 */
		virtual void setVSync(bool enabled) = 0;
		/**
		 *	returns true if vsync is turned on
		 */
		virtual bool isVSync() const = 0;
		/**
		 *	returns the amount of Dots-Per-Inch of the monitor this window is being displayed on.
		 */
		virtual glm::ivec2 getDPI() const = 0;

		/**
		 *	Grabs or Ungrabs the cursor, which means it will be invisible and be able to move indefinitely.
		 *	@param hideIt: if true, the cursor will be grabbed from now on, otherwise it will be released
		 */
		virtual void grabMouseCursor(bool hideIt) = 0;

		/**
		 *	Sets the window to be fullscreen, borderless or windowed.
		 *	@param mode: one of Windowed, Fullscreen or Borderless
		 */
		virtual void setFullscreenMode(FullscreenMode mode) = 0;

		/**
		 *	Returns the current fullscreen mode
		 */
		virtual FullscreenMode getFullscreenMode() const = 0;

		/**
		 *	Sets the window title
		 *	@param title: the new title of the window (in UTF-8 encoding)
		 */
		virtual void setTitle(const std::string& title) = 0;
		/**
		 *	returns the current position of the top left corner in pixel coordinates
		 */
		virtual const glm::ivec2 getPosition() const = 0;
		/**
		 *	sets the position of the top left corner in pixel coordinates
		 *	@param position: the new position of the top left corner
		 */
		virtual void setPosition(const glm::ivec2& position) = 0;

		/**
		 *	minimizes the window
		 */
		virtual void minimize() = 0;
		/**
		 *	returns whether the window is minimized or not
		 */
		virtual bool isMinimized() const = 0;

		/**
		 *	returns the platform specific window, e.g. a GLFWwindow*.
		 */
		virtual void* getNativeWindow() const = 0;

		/**
		 *	To support multiple platforms there is no public constructor.
		 *	Instead the Platform-dependent extensions of this class will have to implement this function
		 *	@param props: an instance of WindowProperties containing width, height and name of the window
		 *		that will be created
		 */
		static Window* create(const WindowProperties& props = WindowProperties());
	};

}