#pragma once

// ######################################################################### //
// ### Time.h ############################################################## //
// ### Defines classes providing easy means of time measurement. The     ### //
// ### default time-scale is a second.                                   ### //
// ######################################################################### //

namespace LazyEngine {

	/**
	 *	Represents the timepoint during its creation
	 */
	class TimePoint {
	public:
		/**
		 *	default constructor.
		 *	stores the time of creation.
		 */
		TimePoint();
		~TimePoint() = default;

		/**
		 *	prints the time in "weekday month day hh:mm:ss year"
		 */
		std::string toString() const;

		long long toMicroseconds() const;

		/**
		 *	override the stream operator to print toString().
		 */
		friend std::ostream& operator<<(std::ostream& stream, const TimePoint& timePoint) {
			return stream << timePoint.toString();
		}

	private:
		// the internal representation of the timepoint
		std::chrono::time_point<std::chrono::system_clock> m_timePoint;

		// allow TimeStep to use m_timePoint directly.
		friend class TimeStep;
	};

	/**
	 *	Represents a duration
	 */
	class TimeStep {
	public:
		/**
		 *	takes two TimePoints and stores their difference in time as the duration
		 *	@param start: the begin of this timestep
		 *	@param end: the end of this timestep
		 */
		TimeStep(const TimePoint& start, const TimePoint& end);
		/**
		 *	creates a duration from the input
		 *	@param deltaTime: the amount of time to be represented by this in seconds
		 */
		TimeStep(float deltaTime = 0.f);
		~TimeStep() = default;

		/**
		 *	implicit or explicit cast to float defaults to seconds
		 */
		operator float() const;

		/**
		 *	adds a timestep to this time step
		 *	@param deltaTime
		 */
		void addTimeSpan(const TimeStep& deltaTime);

		inline TimeStep operator+(const TimeStep& deltaTime) {
			TimeStep copy(*this);
			copy.addTimeSpan(deltaTime);
			return copy;
		}

		inline TimeStep& operator+=(const TimeStep& deltaTime) {
			addTimeSpan(deltaTime);
			return *this;
		}

		/**
		 *	sets the time span to a given value
		 *	@param timeSpanSeconds: the new value of this time step in seconds
		 */
		void setTimeSpan(float timeSpanSeconds);

		/**
		 *	returns the amount of seconds that this TimeStep is representing
		 */
		float getSeconds() const;
		/**
		 *	returns the amount of milliseconds that this TimeStep is representing
		 */
		float getMilliseconds() const;
		/**
		 *	returns the amount of mircoseconds that this TimeStep is representing
		 */
		float getMicroseconds() const;
		/**
		 *	override the stream operator to display this timestep in seconds
		 */
		friend std::ostream& operator<<(std::ostream& stream, const TimeStep& timeStep) {
			return stream << timeStep.getSeconds() << "s";
		}

	private:
		// this duration is in seconds by default
		std::chrono::duration<float> m_duration;
	};
}