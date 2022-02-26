#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Core/Input/GamepadCodes.h"
#include "LazyEngine/Events/GamepadEvent.h"

namespace LazyEngine {


	class LAZYENGINE_API Gamepad {
	public:

		static Ref<Gamepad> create(GamepadID id);

		static Ref<Gamepad> create(GamepadID internalID, GamepadID lazyEngineID);

		/**
		 *	Looks for connected gamepads and sets up event handlers 
		 *	for gamepad connection and disconnection events.
		 *	This will be called once on startup.
		 */
		static void init();

	public:
		/**
		 *	checks if this gamepad is still connected.
		 */
		virtual bool isConnected() const = 0;

		/**
		 *	returns the id for this gamepad
		 */
		inline GamepadID getID() const {
			return m_lazyEngineID;
		}

		/**
		 *	returns the name of this gamepad
		 */
		virtual const std::string getName() const = 0;

		/**
		 *	checks the state of a button at this moment
		 *	@param button: The button to query
		 */
		virtual bool isButtonPressed(GamepadButtonCode button) const = 0;

		/**
		 *	returns the value of an axis
		 *	@param axis: The axis to query
		 */
		virtual float getAxisValue(GamepadAxisCode axis) const = 0;

		/**
		 *	sets the vibration intensity of this gamepad's rumble motors in the main body
		 *	@param intensityLeft: the intensity of the vibration on the left. If this is NAN, no changes will be made
		 *	@param intensityRight: the intensity of the vibration on the right. If this is NAN, no changes will be made
		 */
		virtual void setVibration(float intensityLeft, float intensityRight) = 0;

		/**
		 *	returns the vibration intensity of this gamepad's rumble motors in the main body
		 *	The result is [leftMotorIntensity, rightMotorIntensity]
		 */
		virtual const glm::vec2 getVibration() const = 0;

		/**
		 *	sets the vibration intensity of the rumble motors in the triggers.
		 *	This will not be supported by any platform.
		 *	@param intensityLeft: intensity of the left trigger. if this is NAN, no changes will be made
		 *	@param intensityRight: intensity of the right trigger. if this is NAN, no changes will be made
		 */
		virtual void setTriggerVibration(float intensityLeft, float intensityRight) = 0;

		/**
		 *	returns the vibration intensity of this gamepad's rumble motors in the triggers
		 *	The result is [leftTriggerIntensity, rightTriggerIntensity]
		 */
		virtual const glm::vec2 getTriggerVibration() const = 0;

		/**
		 *	return the values of a stick in x- and y-direction
		 *	@param stick: The stick to query
		 */
		virtual const glm::vec2 getStickValue(GamepadStickCode stick) const = 0;

		/**
		 *	polls all events of this gamepad. 
		 *	For each event the callback function set in "setEventCallbackFunc" will be called
		 */
		virtual void pollEvents() = 0;
		
		/**
		 *	sets the function that should be called on an event.
		 */
		virtual void setEventCallbackFunc(const EventCallbackFn& callback) = 0;

		/**
		 *	Sets the deadzone of a specific stick
		 */
		virtual void setDeadzone(GamepadAxisCode axis, float radius) = 0;

		/**
		 *	Returns the deadzone of a specific stick
		 */
		virtual float getDeadzone(GamepadAxisCode axis) const = 0;

		/**
		 *	The equals operator
		 */
		virtual bool operator==(const Gamepad& other) const = 0;

		inline void setLazyEngineID(GamepadID id) { m_lazyEngineID = id; }
	protected:
		Gamepad(GamepadID lazyEngineID) : m_lazyEngineID(lazyEngineID) {}

		inline GamepadID getLazyEngineID() { return m_lazyEngineID; }

	private:
		GamepadID m_lazyEngineID;
	};

}