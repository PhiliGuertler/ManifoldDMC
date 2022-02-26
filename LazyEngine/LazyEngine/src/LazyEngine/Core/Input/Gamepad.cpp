#include "LazyEngine/gepch.h"

#include "Gamepad.h"

// --- Windows --- //
#include "LazyEngine/platform/Windows/WindowsGamepad.h"
// --- Linux --- //
#include "LazyEngine/platform/Linux/LinuxGamepad.h"

namespace LazyEngine {

	Ref<Gamepad> Gamepad::create(GamepadID id) {
	#ifdef LAZYENGINE_PLATFORM_WINDOWS
		return WindowsGamepad::create(id);
	#elif defined(LAZYENGINE_PLATFORM_LINUX)
		return createRef<LinuxGamepad>(id);
	#elif defined(LAZYENGINE_PLATFORM_MACOS)
		#error MacOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_IOS)
		#error IOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_ANDROID)
		#error Android is not supported yet
	#endif
	}

	Ref<Gamepad> Gamepad::create(GamepadID internalID, GamepadID lazyEngineID) {
	#ifdef LAZYENGINE_PLATFORM_WINDOWS
		return WindowsGamepad::create(internalID, lazyEngineID);
	#elif defined(LAZYENGINE_PLATFORM_LINUX)
		return createRef<LinuxGamepad>(internalID, lazyEngineID);
	#elif defined(LAZYENGINE_PLATFORM_MACOS)
		#error MacOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_IOS)
		#error IOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_ANDROID)
		#error Android is not supported yet
	#endif
	}

	void Gamepad::init() {

	#ifdef LAZYENGINE_PLATFORM_WINDOWS
		WindowsGamepad::init();
	#elif defined(LAZYENGINE_PLATFORM_LINUX)
		LinuxGamepad::init();
	#elif defined(LAZYENGINE_PLATFORM_MACOS)
		#error MacOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_IOS)
		#error IOS is not supported yet
	#elif defined(LAZYENGINE_PLATFORM_ANDROID)
		#error Android is not supported yet
	#endif

	}

}