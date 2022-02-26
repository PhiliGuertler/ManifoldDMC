#include "LazyEngine/gepch.h"

#include "Timer.h"

#include "Profiler.h"

namespace LazyEngine {

	// ######################################################################## //
	// ### Timer ############################################################## //
	// ######################################################################## //

	Timer::Timer()
		: m_startTimePoint()
		, m_timeMeasured()
		, m_isStopped(true)
		, m_pauseStartTime()
		, m_timePausedSeconds(0.f)
		, m_isPaused(false)
	{
		// empty
	}

	Timer::~Timer() {
		// empty
	}

	void Timer::start() {
		// set the start time point to now
		m_startTimePoint = TimePoint();
		// set the timer to not be in the stopped state
		m_isStopped = false;
		// set the timer to not be in the paused state
		m_isPaused = false;
		// reset the paused time
		m_timePausedSeconds = 0.f;
	}

	TimeStep Timer::stop() {
		// save the current time
		m_timeMeasured = peek();
		// set the timer to stopped
		m_isStopped = true;
		// return the measured time
		return m_timeMeasured;
	}

	TimeStep Timer::peek() {
		if (m_isStopped) {
			// return the time set by stop, as this timer is not running
			return m_timeMeasured;
		}
		else {
			// return the time since starting the timer
			updatePauseTime();
			TimeStep sinceStart = TimeStep(m_startTimePoint, TimePoint());
			sinceStart.addTimeSpan(-m_timePausedSeconds);
			return sinceStart;
		}
	}

	void Timer::updatePauseTime() {
		if (m_isPaused) {
			m_timePausedSeconds += TimeStep(m_pauseStartTime, TimePoint());
			pause();
		}
	}

	void Timer::pause() {
		m_pauseStartTime = TimePoint();
		m_isPaused = true;
	}

	void Timer::unpause() {
		m_isPaused = false;
	}

	bool Timer::isRunning() {
		return !m_isPaused && !m_isStopped;
	}


	// ######################################################################## //
	// ### ScopedTimer ######################################################## //
	// ######################################################################## //

	ScopedTimer::ScopedTimer(const std::string& name)
		: m_startTimePoint()
		, m_name(name)
	{
		// empty
	}

	ScopedTimer::~ScopedTimer() {
		LAZYENGINE_CORE_WARN("ScopedTimer \"{0}\": {1}ms", m_name, TimeStep(m_startTimePoint, TimePoint()).getMilliseconds());
	}


	// ######################################################################## //
	// ### ScopedProfilerTimer ################################################ //
	// ######################################################################## //

	ScopedProfilerTimer::ScopedProfilerTimer(const std::string& name)
		: m_start()
		, m_name(name)
	{
		// remove "__cdecl" from the name if it is part of name
		size_t pos = m_name.find("__cdecl ");
		if (pos != std::string::npos) {
			m_name.erase(pos, 8);
		}
	}

	ScopedProfilerTimer::~ScopedProfilerTimer() {
		Profiler& profiler = Profiler::getInstance();

		// do nothing if there is no active session
		if (!profiler.sessionIsRunning()) return;

		TimePoint end;

		// hash the thread id to get a uint
		uint32_t threadID = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

		// create the profile result
		ProfileResult result = { m_name, m_start, end, threadID };

		Profiler::getInstance().writeProfile(result);
	}
}