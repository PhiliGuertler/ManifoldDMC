// ######################################################################### //
// ### Time.cpp ############################################################ //
// ### Implements Time.h                                                 ### //
// ######################################################################### //


#include "LazyEngine/gepch.h"

#include "Time.h"

#include <iomanip>	// std::put_time

namespace LazyEngine {

	// ######################################################################## //
	// ### TimePoint ########################################################## //
	// ######################################################################## //

	TimePoint::TimePoint()
		: m_timePoint(std::chrono::system_clock::now())
	{
		// empty
	}

	std::string TimePoint::toString() const {
		auto localTime = std::chrono::system_clock::to_time_t(m_timePoint);
		std::stringstream result;
		result << std::put_time(std::localtime(&localTime), "%T");
		return result.str();
	}

	long long TimePoint::toMicroseconds() const {
		return std::chrono::time_point_cast<std::chrono::microseconds>(m_timePoint).time_since_epoch().count();
	}



	// ######################################################################## //
	// ### TimeStep ########################################################### //
	// ######################################################################## //

	TimeStep::TimeStep(const TimePoint& start, const TimePoint& end)
		: m_duration(end.m_timePoint - start.m_timePoint)
	{
		// empty
	}

	TimeStep::TimeStep(float deltaTime)
		: m_duration(deltaTime)
	{
		// empty
	}

	TimeStep::operator float() const {
		return getSeconds();
	}

	void TimeStep::addTimeSpan(const TimeStep& deltaTime) {
		m_duration += deltaTime.m_duration;
	}
	
	void TimeStep::setTimeSpan(float timeSpanSeconds) {
		m_duration = std::chrono::duration<float>(timeSpanSeconds);
	}


	float TimeStep::getSeconds() const {
		return m_duration.count();
	}

	float TimeStep::getMilliseconds() const {
		auto milliDuration = std::chrono::duration<float, std::milli>(m_duration);
		return milliDuration.count();
	}

	float TimeStep::getMicroseconds() const {
		auto microDuration = std::chrono::duration<float, std::micro>(m_duration);
		return microDuration.count();
	}
	
}