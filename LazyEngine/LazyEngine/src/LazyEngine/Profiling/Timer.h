#pragma once

#include <string>

#include "LazyEngine/Core/Time.h"

namespace LazyEngine {

	/**
	 *	A timer that can be started and stopped
	 */
	class Timer {
	public:
		/**
		 *	constructor
		 */
		Timer();
		/**
		 *	destructor
		 */
		~Timer();

		/**
		 *	starts the timer
		 */
		void start();
		/**
		 *	Returns the time passed since the last call of start()
		 *	This also stops the timer, which can be restarted via start().
		 */
		TimeStep stop();
		/**
		 *	Returns the time passed since the last call of start()
		 *	The timer will not be stopped by this call
		 */
		TimeStep peek();
		/**
		 *	pauses the timer
		 */
		void pause();
		/**
		 *	unpauses the timer
		 */
		void unpause();
		/**
		 *	returns true if the timer is running
		 */
		bool isRunning();
	protected:
		void updatePauseTime();

	protected:
		// stores the time point of the last call of start()
		TimePoint m_startTimePoint;
		TimeStep m_timeMeasured;
		bool m_isStopped;

		TimePoint m_pauseStartTime;
		float m_timePausedSeconds;
		bool m_isPaused;
	};

	/**
	 *	Starts a timer on creation and prints the time to console on destruction
	 */
	class ScopedTimer {
	public:
		/**
		 *	constructor
		 *	@param name: name of the timer to be printed in the console
		 */
		ScopedTimer(const std::string& name);
		/**
		 *	destructor
		 *	prints the time difference between creation and destruction into the console
		 */
		~ScopedTimer();

	protected:
		// time of creation
		TimePoint m_startTimePoint;
		// name of the timer
		std::string m_name;
	};

	/**
	 *	A timer class to be used with the profiler defined in "Profiler.h".
	 *	It is a variant of a scoped timer, which means it starts a profile
	 *	measurement on creation and sends its results to the Profiler on
	 *	destruction.
	 */
	class ScopedProfilerTimer {
	public:
		/**
		 *	constructor
		 *	@param name: name of the profiled section, e.g. the function name
		 */
		ScopedProfilerTimer(const std::string& name);

		/**
		 *	destructor
		 *	sends the profiling result to the Profiler defined in "Profiler.h"
		 */
		~ScopedProfilerTimer();

	private:
		// time of creation
		TimePoint m_start;
		// name of the timer.
		std::string m_name;
	};
}