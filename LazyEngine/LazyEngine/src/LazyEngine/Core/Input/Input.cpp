#include "LazyEngine/gepch.h"

#include "Input.h"

#ifdef LAZYENGINE_PLATFORM_WINDOWS
#include "LazyEngine/platform/Windows/WindowsInput.h"
#elif defined(LAZYENGINE_PLATFORM_LINUX)
#include "LazyEngine/platform/Linux/LinuxInput.h"
#else
#error No Input Implementation for this platform found!
#endif

namespace LazyEngine {

#ifdef LAZYENGINE_PLATFORM_WINDOWS
	Scope<Input> Input::s_instance = createScope<WindowsInput>();
#elif defined(LAZYENGINE_PLATFORM_LINUX)
	Scope<Input> Input::s_instance = createScope<LinuxInput>();
#elif defined(LAZYENGINE_PLATFORM_MACOS)
	#error MacOS is not supported yet
#elif defined(LAZYENGINE_PLATFORM_IOS)
	#error IOS is not supported yet
#elif defined(LAZYENGINE_PLATFORM_ANDROID)
	#error Android is not supported yet
#endif


}