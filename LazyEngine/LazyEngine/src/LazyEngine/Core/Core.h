#pragma once

// ######################################################################### //
// ### Core.h ############################################################## //
// ### Defines LAZYENGINE_API as dllimport or dllexport depending on     ### //
// ### the LAZYENGINE_BUILD_DLL preprocessor define. Also defines some   ### //
// ### other Macros like ASSERTs and BIT                                 ### //
// ######################################################################### //

#include <memory>

// automatic platform detection
#ifdef _WIN32
	// Windows detection
	// Windows x64/x86
	#ifdef _WIN64
		// Windows x64
		#define LAZYENGINE_PLATFORM_WINDOWS
		#define LAZYENGINE_FUNCSIG __FUNCSIG__
	#else
		// Windows x86
		#error "Windows x86 Builds are not supported by LazyEngine!"
	#endif
#elif defined(__APPLE__) || defined(__MACH__)
	// Apple detection
	#include <TargetConditionals.h>
	// TARGET_OS_MAX exists on all platforms from apple.
	// So all of them must be checked in this order
	#if TARGET_IPHONE_SIMULATOR == 1
		#error "IOS simulator is not supported by LazyEngine!"
	#elif TARGET_OS_IPHONE == 1
		#define LAZYENGINE_PLATFOMR_IOS
		#error "IOS is not supported by LazyEngine!"
	#elif TARGET_OS_MAC == 1
		#define LAZYENGINE_PLATROFM_MACOS
		#error "MacOS is not supported by LazyEngine!"
	#else
		#error "Unknown Apple platform detected!"
	#endif
#elif defined(__ANDROID__)
	// Android detection
	#define LAZYENGINE_PLATFORM_ANDROID
	#error "Android is not supported by LazyEngine!"
#elif defined(__linux__)
	// Linux detection
	#define LAZYENGINE_PLATFORM_LINUX
	#define LAZYENGINE_FUNCSIG __PRETTY_FUNCTION__
	#error "Linux is not supported by LazyEngine!"
#else
	// Unknown platform
	#error "Unknown Platform detected!"
#endif


#ifdef LAZYENGINE_PLATFORM_WINDOWS
	// Windows DLL support
	#if LAZYENGINE_DYNAMIC_LINK
		#ifdef LAZYENGINE_BUILD_DLL
			// export: building the library
			#define LAZYENGINE_API __declspec(dllexport)
		#else
			// import: building the client
			#define LAZYENGINE_API __declspec(dllimport)
		#endif
	#else
		#define LAZYENGINE_API
	#endif
#elif defined(LAZYENGINE_PLATFORM_LINUX)
	// Windows DLL support
	#if LAZYENGINE_DYNAMIC_LINK
		#ifdef LAZYENGINE_BUILD_DLL
			// export: building the library
			#define LAZYENGINE_API __attribute__((visibility("default")))
		#else
			// import: building the client
			#define LAZYENGINE_API 
		#endif
	#else
		#define LAZYENGINE_API
	#endif
#else
	#error Lazyengine only supports Windows so far!
#endif


#if LAZYENGINE_ENABLE_ASSERTS
	// defines asserts to actually assert.
	#define VA_ARGS(...) , ##__VA_ARGS__
	#define LAZYENGINE_ASSERT(x, ...) {if(!(x)) {LAZYENGINE_ERROR("Assertion Failed: "##__VA_ARGS__); __debugbreak();}}
	#define LAZYENGINE_CORE_ASSERT(x, ...) {if(!(x)) { LAZYENGINE_CORE_ERROR("Assertion Failed: "##__VA_ARGS__); __debugbreak(); }}
#else
	// removes asserts.
	#define LAZYENGINE_ASSERT(x, ...)
	#define LAZYENGINE_CORE_ASSERT(x, ...)
#endif

#define BIT(x) (1 << x)

// to be used with the event dispatcher system
// see Application.cpp for an example
#define LAZYENGINE_BIND_EVENT_FUNC(fn) std::bind(&fn, this, std::placeholders::_1)

namespace LazyEngine {

	// TODO: replace these types with their own classes
	// to prevent race conditions in the renderer, etc

	// ######################################################################## //
	// ### Scope ############################################################## //
	/**
	 *	Scope: currently an alias for std::unique_ptr, but later it might be
	 *	refactored to be a custom unique_ptr class.
	 */
	template<typename T>
	using Scope = std::unique_ptr<T>;

	/**
	 *	creates a Scope by taking its construction arguments
	 */
	template<typename T, typename... Args>
	inline constexpr Scope<T> createScope(Args&&... args) {
		return std::make_unique<T>(std::forward<Args>(args)...);
	}

	/**
	 *	creates a Scope out of a raw pointer
	 */
	template<typename T>
	inline Scope<T> createScope(T *item) {
		return Scope<T>(item);
	}
	// ######################################################################## //


	// ######################################################################## //
	// ### Ref ################################################################ //
	/**
	 *	Ref: currently an alias for std::shared_ptr, but later it might be
	 *	refactored to be a custom shared_ptr class.
	 */
	template<typename T>
	using Ref = std::shared_ptr<T>;

	/**
	 *	creates a Ref by taking its construction arguments
	 */
	template<typename T, typename... Args>
	inline constexpr Ref<T> createRef(Args&&... args) {
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	/**
	 *	creates a Ref out of a raw pointer
	 */
	template<typename T>
	inline Ref<T> createRef(T *item) {
		return Ref<T>(item);
	}
	// ######################################################################## //


	// ######################################################################## //
	// ### RendererID ######################################################### //

	// FIXME: this should obviously depend on the currently used renderer api!
	typedef uint32_t RendererID;

	// ######################################################################## //

}