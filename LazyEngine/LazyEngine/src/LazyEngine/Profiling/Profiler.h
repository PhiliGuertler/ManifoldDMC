#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "LazyEngine/Core/Time.h"
#include "Timer.h"

namespace LazyEngine {

	/**
	 *	Data that represents one profiled function call
	 */
	struct ProfileResult {
		std::string name;
		TimePoint start;
		TimePoint end;
		uint32_t threadID;
	};

	/**
	 *	Data representing a whole session
	 */
	struct ProfilerSession {
		std::string name;
	};

	/**
	 *	Singleton class creating profiles of the performance of a given program portion.
	 *	It creates a json-output of a session that can be visualized with chromes "chrome://tracing" page.
	 */
	class Profiler {
	public:
		/**
		 *	static getter for the singleton instance of this class
		 */
		static Profiler& getInstance() {
			static Profiler profiler;
			return profiler;
		}

	public:
		/**
		 *	starts a new session that will be written to filePath.
		 *	@param sessionName: The name of the session.
		 *	@param filePath: The path of the resulting file that can be loaded into chrome://tracing
		 *		The default path is "results.json"
		 */
		void beginSession(const std::string& sessionName, const std::string& filePath = "results.json");

		/**
		 *	ends a session and clears up its resources.
		 */
		void endSession();

		/**
		 *	Prepares a session that will only profile the next n frames.
		 *	This prepared session can be started by calling "beginNSession()"
		 *	@param n: The amount of frames to be profiled
		 *	@param sessionName: the name of the session. The dafault name is "quickProfile"
		 *	@param filePath: the path of the resulting file that can be loaded into chrome://tracing.
		 *		The default filePath if "quickProfile.json"
		 */
		void profileNextNFrames(int n, const std::string& sessionName = "quickProfile", const std::string& filePath = "quickProfile.json");

		/**
		 *	starts a session that will profile the next n frames that have previously been set
		 *	by "profileNextNFrames()".
		 */
		void beginNSession();

		/**
		 *	writes a profile result to the file specified in beginSession().
		 *	@param result: The profile result to be written.
		 */
		void writeProfile(const ProfileResult& result);

		/**
		 *	writes the header of a chrome://tracing-compatible json file to the file specified in beginSession().
		 */
		void writeHeader();

		/**
		 *	writes the footer of a chrome://tracing-compatible json file to the file specified in beginSession().
		 */
		void writeFooter();

		inline long long getNProfileEndFrame() { return m_endFrame; }

		inline bool sessionIsRunning() { return m_currentSession != nullptr; }

	private:
		/**
		 *	private constructor to ensure Singleton nature
		 */
		Profiler();

		std::ofstream m_outputStream;
		int m_profileCount;
		std::unique_ptr<ProfilerSession> m_currentSession;

		// members for profileNextNFrames
		long long m_endFrame;
		std::string m_profileSessionName;
		std::string m_profileSessionFilePath;

	};

}

// to be defined by a client application
#ifdef LAZYENGINE_ENABLE_PROFILING
	#define LAZYENGINE_PROFILE_BEGIN_SESSION(sessionName, filePath) ::LazyEngine::Profiler::getInstance().beginSession(sessionName, filePath)
	#define LAZYENGINE_PROFILE_END_SESSION() ::LazyEngine::Profiler::getInstance().endSession()

	#define LAZYENGINE_PROFILE_SCOPE(name) ::LazyEngine::ScopedProfilerTimer timer##__LINE__(name)
	#define LAZYENGINE_PROFILE_FUNCTION() LAZYENGINE_PROFILE_SCOPE(LAZYENGINE_FUNCSIG)
#else
	// these macros will be expanded to nothing if profiling is disabled
	#define LAZYENGINE_PROFILE_BEGIN_SESSION(sessionName, filePath)
	#define LAZYENGINE_PROFILE_END_SESSION()
	
	#define LAZYENGINE_PROFILE_SCOPE(name)
	#define LAZYENGINE_PROFILE_FUNCTION()
#endif

// renderer profiling must be enabled seperately because it would slow the whole program down by a lot for regular profiling
#ifdef LAZYENGINE_ENABLE_RENDERER_PROFILING
	#define LAZYENGINE_PROFILE_RENDERER_FUNCTION() LAZYENGINE_PROFILE_SCOPE("Renderer__" LAZYENGINE_FUNCSIG)
#else
	// these macros will be expanded to nothing if profiling is disabled
	#define LAZYENGINE_PROFILE_RENDERER_FUNCTION()
#endif