#pragma once

#include "LazyEngine/Core/Input/Gamepad.h"
#ifdef LAZYENGINE_PLATFORM_GLFW3

namespace LazyEngine {

	class LAZYENGINE_API GLFWGamepad : public Gamepad {
	public:
		/**
		 *	Looks for connected gamepads and sets up event handlers
		 *	for gamepad connection and disconnection events.
		 *	This will be called once on startup.
		 */
		static void init();

		static Ref<Gamepad> create(GamepadID id);

		static Ref<Gamepad> create(GamepadID internalID, GamepadID lazyEngineID);

	public:
		GLFWGamepad(GamepadID gamepadID);

		GLFWGamepad(GamepadID internalID, GamepadID lazyEngineID);

		/**
		 *	checks if this gamepad is still connected.
		 */
		virtual bool isConnected() const override;

		/**
		 *	returns the name of this gamepad
		 */
		virtual const std::string getName() const override;

		/**
		 *	checks the state of a button at this moment
		 *	@param button: The button to query
		 */
		virtual bool isButtonPressed(GamepadButtonCode button) const override;

		/**
		 *	returns the value of an axis
		 *	@param axis: The axis to query
		 */
		virtual float getAxisValue(GamepadAxisCode axis) const override;

		/**
		 *	sets the vibration intensity of this gamepad's rumble motors in the main body
		 *	@param intensityLeft: the intensity of the vibration on the left. If this is NAN, no changes will be made
		 *	@param intensityRight: the intensity of the vibration on the right. If this is NAN, no changes will be made
		 */
		virtual void setVibration(float intensityLeft, float intensityRight) override;

		/**
		 *	returns the vibration intensity of this gamepad's rumble motors in the main body
		 *	The result is [leftMotorIntensity, rightMotorIntensity]
		 */
		virtual const glm::vec2 getVibration() const override;


		/**
		 *	sets the vibration intensity of the rumble motors in the triggers.
		 *	This will not be supported by any platform.
		 *	@param intensityLeft: intensity of the left trigger. if this is NAN, no changes will be made
		 *	@param intensityRight: intensity of the right trigger. if this is NAN, no changes will be made
		 */
		virtual void setTriggerVibration(float intensityLeft, float intensityRight) override;

		/**
		 *	returns the vibration intensity of this gamepad's rumble motors in the triggers
		 *	The result is [leftTriggerIntensity, rightTriggerIntensity]
		 */
		virtual const glm::vec2 getTriggerVibration() const override;


		/**
		 *	return the values of a stick in x- and y-direction
		 *	@param stick: The stick to query
		 */
		virtual const glm::vec2 getStickValue(GamepadStickCode stick) const override;

		/**
		 *	polls all events of this gamepad
		 *	For each event the callback function set in "setEventCallbackFunc" will be called
		 */
		virtual void pollEvents() override;

		/**
		 *	sets the function that should be called on an event.
		 */
		virtual void setEventCallbackFunc(const EventCallbackFn& callback) override;

		/**
		 *	Sets the deadzone of a specific stick
		 */
		virtual void setDeadzone(GamepadAxisCode axis, float radius) override;

		/**
		 *	Returns the deadzone of a specific stick
		 */
		virtual float getDeadzone(GamepadAxisCode axis) const override;

		/**
		 *	The equals operator
		 */
		virtual bool operator==(const Gamepad& other) const override;

	private:
		// the glfw id of this gamepad
		int m_glfwID;
		// the name of this gamepad
		std::string m_name;

		// data structures to detect events
		std::array<bool, 17> m_buttonStates;
		std::array<float, 6> m_axisStates;

		EventCallbackFn m_callback;

		std::array<float, 6> m_axisDeadzones;
	};

}
#endif