// ######################################################################### //
// ### Log.cpp ############################################################# //
// ### Implements Log.h                                                  ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	std::shared_ptr<spdlog::logger> Log::s_coreLogger;
	std::shared_ptr<spdlog::logger> Log::s_clientLogger;

	void Log::init() {
		LAZYENGINE_PROFILE_FUNCTION();

		spdlog::set_pattern("%^[%T] %n: %v%$");
		s_coreLogger = spdlog::stdout_color_mt("LazyEngineCore");
		s_coreLogger->set_level(spdlog::level::trace);

		s_clientLogger = spdlog::stdout_color_mt("Application");
		s_clientLogger->set_level(spdlog::level::trace);
	}
}