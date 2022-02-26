#include "LazyEngine/gepch.h"

#include "Window.h"

// --- Windows -- //
#include "LazyEngine/platform/Windows/WindowsWindow.h"
// --- Linux -- //
#include "LazyEngine/platform/Linux/LinuxWindow.h"

namespace LazyEngine {

	Window *Window::create(const WindowProperties& props) {
		#ifdef LAZYENGINE_PLATFORM_WINDOWS
			return new WindowsWindow(props);
		#elif defined(LAZYENGINE_PLATFORM_LINUX)
			return new LinuxWindow(props);
		#elif defined(LAZYENGINE_PLATFORM_MACOS)
			#error MacOS is not supported yet
		#elif defined(LAZYENGINE_PLATFORM_IOS)
			#error IOS is not supported yet
		#elif defined(LAZYENGINE_PLATFORM_ANDROID)
			#error Android is not supported yet
		#endif
	}

}