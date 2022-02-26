// ######################################################################### //
// ### GLFWWindow.cpp ###################################################### //
// ### Implements GLFWWindow.h                                           ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "GLFWWindow.h"

#ifdef LAZYENGINE_PLATFORM_GLFW3

#include "LazyEngine/Core/Input/Input.h"
#include "LazyEngine/Events/Event.h"
#include "LazyEngine/Events/ApplicationEvent.h"
#include "LazyEngine/Events/GamepadEvent.h"
#include "LazyEngine/Events/KeyEvent.h"
#include "LazyEngine/Events/MouseEvent.h"
#include "LazyEngine/Profiling/Profiler.h"

#include "LazyEngine/platform/OpenGL/OpenGLContext.h"

namespace LazyEngine {

	// don't initialize glfw more than once
	static bool s_GLFWIsInitialized = false;

	/**
	 *	log errors of glfw with the core logger
	 */
	static void GLFWErrorCallback(int error, const char *description) {
		LAZYENGINE_CORE_ERROR("GLFW Error ({0}): {1}", error, description);
	}

	/**
	 *	constructor
	 */
	GLFWWindow::GLFWWindow(const WindowProperties& props)
		: m_window(nullptr)
		, m_context(nullptr)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		init(props);
	}

	/**
	 *	destructor
	 */
	GLFWWindow::~GLFWWindow() {
		LAZYENGINE_PROFILE_FUNCTION();

		shutdown();
	}

	void GLFWWindow::init(const WindowProperties& props) {
		LAZYENGINE_PROFILE_FUNCTION();

		// save the input parameters in a member
		m_data.title = props.title;
		m_data.size = props.size;
		m_data.lastWindowedSize = props.size;
		m_data.position = props.position;


		// print the input parameters
		LAZYENGINE_CORE_INFO("Creating window {0}, size: ({1}, {2}), position: ({3}, {4})", props.title, props.size.x, props.size.y, props.position.x, props.position.y);

		// initialize glfw if it has not been initialized yet
		if (!s_GLFWIsInitialized) {
			int success = glfwInit();
			// check if glfw has been initialized successfully.
			// this check will not be part of the release build
			LAZYENGINE_CORE_ASSERT(success, "Could not initialize GLFW!");
			// set the error callback of glfw to print to the core logger.
			glfwSetErrorCallback(GLFWErrorCallback);

			s_GLFWIsInitialized = true;
		}

		// create the actual window
		glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
		glfwWindowHint(GLFW_SAMPLES, 4);
		m_window = glfwCreateWindow((int)m_data.size.x, (int)m_data.size.y, m_data.title.c_str(), nullptr, nullptr);

		// create an opengl context
		m_context = new OpenGLContext(m_window);
		m_context->init();

		// Allow the use of m_data locally for m_window.
		// It will be possible in callbacks to get this data from the argument "window" via glfwGetWindowUserPointer.
		glfwSetWindowUserPointer(m_window, &m_data);
		glfwSetWindowPos(m_window, (int)m_data.position.x, (int)m_data.position.y);
		// Enable vsync by default
		setVSync(true);

		// Setup GLFW callbacks
		// set the callback of resize events to trigger a WindowResizeEvent
		glfwSetWindowSizeCallback(m_window, [](GLFWwindow *window, int width, int height) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			// update width and height.
			data.size.x = width;
			data.size.y = height;
			if (data.fullscreenMode == FullscreenMode::Windowed) {
				data.lastWindowedSize.x = width;
				data.lastWindowedSize.y = height;
			}

			LAZYENGINE_CORE_INFO("Window resized to: ({0}, {1})", width, height);

			// dispatch a WindowResizeEvent
			WindowResizeEvent event(width, height);
			data.eventCallback(event);
			});

		// set the callback of window close to trigger a WindowCloseEvent
		glfwSetWindowCloseCallback(m_window, [](GLFWwindow *window) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			// dispatch a WindowCloseEvent
			WindowCloseEvent event;
			data.eventCallback(event);
			});

		// set the calback of window position to update the position of the window when it is moved
		glfwSetWindowPosCallback(m_window, [](GLFWwindow* window, int xpos, int ypos) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
			// update the position
			bool isMinimized = glfwGetWindowAttrib(window, GLFW_ICONIFIED);

			if (data.fullscreenMode == FullscreenMode::Windowed && !isMinimized) {
				data.position.x = xpos;
				data.position.y = ypos;
			}

			});

		// ################################################################# //
		// ### Keyboard Input ############################################## //

		// set the callback of a key input to trigger a KeyEvent
		glfwSetKeyCallback(m_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action) {
			case GLFW_PRESS:
			{
				// a key has been pressed. Dispatch a KeyPressedEvent.
				KeyPressedEvent event(static_cast<KeyCode>(key), 0);
				data.eventCallback(event);
				break;
			}
			case GLFW_RELEASE:
			{
				// A key has been released. Dispatch a KeyReleasedEvent.
				KeyReleasedEvent event(static_cast<KeyCode>(key));
				data.eventCallback(event);
				break;
			}
			case GLFW_REPEAT:
			{
				// A key repeat event has triggered. Dispatch another KeyPressedEvent with the repeat flag set.
				KeyPressedEvent event(static_cast<KeyCode>(key), 1);
				data.eventCallback(event);
				break;
			}
			default:
				LAZYENGINE_CORE_ERROR("This key action is not supported: {0}", action);
			}
			});

		// set the callback of a char input to trigger a KeyTypedEvent
		glfwSetCharCallback(m_window, [](GLFWwindow *window, unsigned int keycode) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			// Dispatch a KeyTypedEvent.
			KeyTypedEvent event(static_cast<KeyCode>(keycode));
			data.eventCallback(event);
			});


		// ################################################################# //
		// ### Mouse Input ################################################# //

		// set the callback of a mouse button input to trigger a MouseEvent
		glfwSetMouseButtonCallback(m_window, [](GLFWwindow *window, int button, int action, int mods) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action) {
			case GLFW_PRESS:
			{
				// A mouse button has been pressed. Dispatch a MouseButtonPressedEvent.
				MouseButtonPressedEvent event(static_cast<MouseButtonCode>(button));
				data.eventCallback(event);
				break;
			}
			case GLFW_RELEASE:
			{
				// A mouse button has been released. Dispatch a MouseButtonReleasedEvent.
				MouseButtonReleasedEvent event(static_cast<MouseButtonCode>(button));
				data.eventCallback(event);
				break;
			}
			default:
				LAZYENGINE_CORE_ERROR("This mouse button action is not supported: {0}", action);
			}
			});

		// set the callback of a mouse scroll to trigger a MouseScrolledEvent
		glfwSetScrollCallback(m_window, [](GLFWwindow *window, double xOffset, double yOffset) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			// A mouse scroll has been detected. Dispatch a MouseScrolledEvent.
			MouseScrolledEvent event((float)xOffset, (float)yOffset);
			data.eventCallback(event);
			});

		// set the callback of a mouse movement to trigger a MouseMovedEvent
		glfwSetCursorPosCallback(m_window, [](GLFWwindow *window, double xPos, double yPos) {
			// get user data (which is GLFWWindow::m_data)
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			// The mouse has moved. Dispatch a MouseMovedEvent.
			MouseMovedEvent event((float)xPos, (float)yPos);
			data.eventCallback(event);
			});


		// ################################################################# //
		// ### Gamepad Input ############################################### //

		// set up event handlers for connection/disconnection events
		// and register all currently connected gamepads
		Gamepad::init();


	}

	/**
	 *	clean up the glfw window
	 */
	void GLFWWindow::shutdown() {
		LAZYENGINE_PROFILE_FUNCTION();

		glfwDestroyWindow(m_window);
	}

	/**
	 *	update the contents of the window, e.g. by a buffer swap.
	 */
	void GLFWWindow::onUpdate() {
		LAZYENGINE_PROFILE_FUNCTION();

		// handle all events that are in the event queue.
		glfwPollEvents();
		Input::pollEvents();

		// update to next frame
		m_context->swapBuffers();
	}

	void GLFWWindow::setSize(const glm::ivec2& size) {
		glfwSetWindowSize(m_window, (int)size.x, (int)size.y);
	}

	void GLFWWindow::setVSync(bool enabled) {
		LAZYENGINE_PROFILE_FUNCTION();

		if (enabled) {
			// turn on vsync in glfw
			glfwSwapInterval(1);
		}
		else {
			// disable vsync in glfw
			glfwSwapInterval(0);
		}
		m_data.vSyncEnabled = enabled;
	}

	glm::ivec2 GLFWWindow::getDPI() const {
		auto monitor = glfwGetPrimaryMonitor();

		// get monitor size in millimeters
		glm::ivec2 physicalSize;
		glfwGetMonitorPhysicalSize(monitor, &physicalSize.x, &physicalSize.y);

		// get monitor resolution in pixels
		glm::ivec2 pixelSize;
		const GLFWvidmode *mode = glfwGetVideoMode(monitor);
		pixelSize.x = mode->width;
		pixelSize.y = mode->height;

		// one inch is equal to 25.4 mm
		constexpr float MM_TO_INCH = 25.4f;

		// compute the dpi by dividing the resolution (in pixels) by the physical size (in inches)
		glm::ivec2 dpi = glm::vec2(pixelSize) / (glm::vec2(physicalSize) / MM_TO_INCH);
		return dpi;
	}

	void GLFWWindow::grabMouseCursor(bool hideIt) {
		if (hideIt) {
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		else {
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void GLFWWindow::setFullscreenMode(FullscreenMode fullscreen) {
		if (getFullscreenMode() == fullscreen) {
			// this would not change the state of the window
			return;
		}

		m_data.fullscreenMode = fullscreen;

		// get current monitor
		// TODO: let the user decide which monitor to pick
		// TODO: actually, a list of monitors might be pretty useful.
		auto monitor = glfwGetPrimaryMonitor();


		if (fullscreen == FullscreenMode::Windowed) {
			// turn this into a windowed window
			glfwSetWindowMonitor(m_window, nullptr, m_data.position.x, m_data.position.y, m_data.lastWindowedSize.x, m_data.lastWindowedSize.y, 0);
			setVSync(m_data.vSyncEnabled);

		} else if(fullscreen == FullscreenMode::Fullscreen || fullscreen == FullscreenMode::Borderless) {
			// turn this into a fullscreen/borderless window
			// this code is mostly taken from https://www.glfw.org/docs/3.3/window_guide.html#window_full_screen
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);

			glm::ivec2 newSize = { mode->width, mode->height };
			setSize(newSize);
			glfwSetWindowMonitor(m_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
			setVSync(m_data.vSyncEnabled);
		}
	}

	FullscreenMode GLFWWindow::getFullscreenMode() const {
		return m_data.fullscreenMode;
	}

	void GLFWWindow::setTitle(const std::string& title) {
		m_data.title = title;
		glfwSetWindowTitle(m_window, title.c_str());
	}

	void GLFWWindow::setPosition(const glm::ivec2& position) {
		glfwSetWindowPos(m_window, (int)position.x, (int)position.y);
	}

	void GLFWWindow::minimize() {
		// TODO
		glfwIconifyWindow(m_window);
	}

	bool GLFWWindow::isMinimized() const {
		bool isMinimized = glfwGetWindowAttrib(m_window, GLFW_ICONIFIED);
		return isMinimized;
	}

}
#endif