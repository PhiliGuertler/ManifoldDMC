#pragma once

// ######################################################################### //
// ### Log.h ############################################################### //
// ### Defines a Logger class using spdlog. Also defines some Logging    ### //
// ### Macros for easy use of the Loggers.                               ### //
// ######################################################################### //

#include "Core.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

namespace LazyEngine {

	/**
	 *	Static class to manage the loggers.
	 *	There is one Logger for the Engine-Core and another one for the Client.
	 *	Both loggers are thread-safe.
	 */
	class LAZYENGINE_API Log
	{
	public:
		/**
		 *	initializes the static loggers of this class
		 */
		static void init();

		inline static spdlog::logger& getCoreLogger() { return *s_coreLogger; }
		inline static spdlog::logger& getClientLogger() { return *s_clientLogger; }

	private:
		// spdlog returns shared_ptrs for its loggers.
		static std::shared_ptr<spdlog::logger> s_coreLogger;
		static std::shared_ptr<spdlog::logger> s_clientLogger;
	};

}

// ######################################################################### //
// ### core log macros (shortcuts) ######################################### //

/**
 *	Prints an error message in red in the log with the prefix "LazyEngineCore".
 */
#define LAZYENGINE_CORE_ERROR(...) ::LazyEngine::Log::getCoreLogger().error(__VA_ARGS__)
/**
 *	Prints a warning message in yellow in the log with the prefix "LazyEngineCore".
 */
#define LAZYENGINE_CORE_WARN(...)  ::LazyEngine::Log::getCoreLogger().warn(__VA_ARGS__)
 /**
  *	Prints an info message in green in the log with the prefix "LazyEngineCore".
  */
#define LAZYENGINE_CORE_INFO(...)  ::LazyEngine::Log::getCoreLogger().info(__VA_ARGS__)
/**
 *	Prints a trace message in white in the log with the prefix "LazyEngineCore".
 */
#define LAZYENGINE_CORE_TRACE(...) ::LazyEngine::Log::getCoreLogger().trace(__VA_ARGS__)
/**
 *	Prints a fatal error message in red in the log with the prefix "LazyEngineCore".
 */
#define LAZYENGINE_CORE_FATAL(...) ::LazyEngine::Log::getCoreLogger().fatal(__VA_ARGS__)

// ######################################################################### //
// ### client log macros (shortcuts) ####################################### //

/**
 *	Prints an error message in red in the log with the prefix "Application".
 */
#define LAZYENGINE_ERROR(...) ::LazyEngine::Log::getClientLogger().error(__VA_ARGS__)
/**
 *	Prints a warning message in yellow in the log with the prefix "Application".
 */
#define LAZYENGINE_WARN(...)  ::LazyEngine::Log::getClientLogger().warn(__VA_ARGS__)
 /**
  *	Prints an info message in green in the log with the prefix "Application".
  */
#define LAZYENGINE_INFO(...)  ::LazyEngine::Log::getClientLogger().info(__VA_ARGS__)
/**
 *	Prints a trace message in white in the log with the prefix "Application".
 */
#define LAZYENGINE_TRACE(...) ::LazyEngine::Log::getClientLogger().trace(__VA_ARGS__)
/**
 *	Prints a fatal error message in red in the log with the prefix "Application".
 */
#define LAZYENGINE_FATAL(...) ::LazyEngine::Log::getClientLogger().fatal(__VA_ARGS__)