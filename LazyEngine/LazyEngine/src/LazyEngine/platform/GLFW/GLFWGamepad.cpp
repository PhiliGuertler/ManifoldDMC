#include "LazyEngine/gepch.h"

#include "GLFWGamepad.h"

#ifdef LAZYENGINE_PLATFORM_GLFW3

#include "LazyEngine/Core/Constants.h"
#include "LazyEngine/Core/Util.h"
#include "LazyEngine/Core/Input/Input.h"

#include <GLFW/glfw3.h>

namespace LazyEngine {

	void GLFWGamepad::init() {

		// detect all currently connected joysticks
		for (int i = 0; i < 16; ++i) {
			if (glfwJoystickPresent(GLFW_JOYSTICK_1 + i)) {
				Ref<Gamepad> gamepad = GLFWGamepad::create((GamepadID)i);
				GamepadID lazyEngineID = Input::connectGamepad(gamepad);
				gamepad->setLazyEngineID(lazyEngineID);
			}
		}

		// set the callback of gamepad connections
		glfwSetJoystickCallback([](int gamepadID, int event) {
			if (event == GLFW_CONNECTED) {
				Ref<Gamepad> gamepad = create(gamepadID);
				// gamepad has connected
				GamepadID lazyEngineID = Input::connectGamepad(gamepad);
				gamepad->setLazyEngineID(lazyEngineID);
			}
			else if (event == GLFW_DISCONNECTED) {
				// gamepad has been disconnected
				Input::disconnectGamepad(gamepadID);
			}
		});
	}

	Ref<Gamepad> GLFWGamepad::create(GamepadID id) {
		return createRef<GLFWGamepad>(id);
	}

	Ref<Gamepad> GLFWGamepad::create(GamepadID internalID, GamepadID lazyEngineID) {
		return createRef<GLFWGamepad>(internalID, lazyEngineID);
	}


	GLFWGamepad::GLFWGamepad(GamepadID gamepadID)
		: GLFWGamepad(gamepadID, gamepadID)
	{
		// empty
	}

	GLFWGamepad::GLFWGamepad(GamepadID internalID, GamepadID lazyEngineID)
		: Gamepad(lazyEngineID)
		, m_glfwID(internalID)
		, m_name("<Unknown Name>")
		, m_buttonStates()
		, m_axisStates()
		, m_callback()
		, m_axisDeadzones()
	{
		if (internalID == -1) return;

		std::fill(m_buttonStates.begin(), m_buttonStates.end(), false);
		std::fill(m_axisStates.begin(), m_axisStates.end(), 0.f);
		std::fill(m_axisDeadzones.begin(), m_axisDeadzones.end(), 0.1f);

		// get the name of this gamepad
		if (glfwJoystickIsGamepad(internalID)) {
			const char* name = glfwGetGamepadName(internalID);
			m_name = std::string(name);
		}
		else {
			LAZYENGINE_CORE_WARN("Gamepad {0} doesn't have an appropriate gamepad mapping!", lazyEngineID);
			const char* name = glfwGetJoystickName(lazyEngineID);
			m_name = std::string(name);
		}
	}


	bool GLFWGamepad::isConnected() const {
		int present = glfwJoystickPresent(m_glfwID);
		return present == GLFW_TRUE;
	}

	const std::string GLFWGamepad::getName() const {
		return m_name;
	}

	bool GLFWGamepad::isButtonPressed(GamepadButtonCode button) const {
		if (!isConnected()) return false;

		// LT and RT are not really buttons, but they will be handled as such.
		if (button == GamepadButtonCode::XBox_LT) {
			// this is a "button" that is not directly supported by glfw, as it is actually an axis
			return getAxisValue(GamepadAxisCode::Left_Trigger) > Constants::EPSILON;
		}
		else if (button == GamepadButtonCode::XBox_RT) {
			// this is a "button" that is not directly supported by glfw, as it is actually an axis
			return getAxisValue(GamepadAxisCode::Right_Trigger) > Constants::EPSILON;
		}

		if (glfwJoystickIsGamepad(m_glfwID)) {
			// This is actually a gamepad, glfw offers a mapping
			GLFWgamepadstate state;
			if (glfwGetGamepadState(m_glfwID, &state)) {
				return state.buttons[static_cast<size_t>(button)] == GLFW_PRESS;
			}
			else {
				LAZYENGINE_CORE_ERROR("Error retrieving gamepad {0}'s state", getID());
				return false;
			}
		}
		else {
			// this is a joystick without a mapping in glfw
			// query the button states of this gamepad
			int buttonCount = 0;
			const unsigned char *buttons = glfwGetJoystickButtons(m_glfwID, &buttonCount);

			return buttons[static_cast<size_t>(button)] == GLFW_PRESS;
		}
	}

	float GLFWGamepad::getAxisValue(GamepadAxisCode axis) const {
		if (!isConnected()) return NAN;

		float deadZone = m_axisDeadzones[static_cast<size_t>(axis)];
		float fullInterval = 1.f - deadZone;

		if (glfwJoystickIsGamepad(m_glfwID)) {
			// This is actually a gamepad, glfw offers a mapping
			GLFWgamepadstate state;
			if (glfwGetGamepadState(m_glfwID, &state)) {
				float result = state.axes[static_cast<size_t>(axis)];
				if (axis == GamepadAxisCode::Left_Trigger || axis == GamepadAxisCode::Right_Trigger) {
					// move the trigger range from [-1,1] to [0,1]
					result = result * 0.5f + 0.5f;
				}

				result = fabs(result) < deadZone ? 0.f : Util::sgn(result) * (fabs(result) - deadZone) / fullInterval;
				return result;
			}
			else {
				LAZYENGINE_CORE_ERROR("Error retrieving gamepad {0}'s state", getID());
				return false;
			}
		}
		else {
			// query the state of this gamepad's axes
			int axisCount = 0;
			const float *axes = glfwGetJoystickAxes(m_glfwID, &axisCount);

			// check if this gamepad has the desired axis
			if (static_cast<size_t>(axis) >= axisCount) {
				//LAZYENGINE_CORE_ERROR("Gamepad {0} only has {1} axes, axis {2} is invalid!", m_gamepadID, axisCount, axis);
				return NAN;
			}

			// return the value
			float result = axes[static_cast<size_t>(axis)];
			if (axis == GamepadAxisCode::Left_Trigger || axis == GamepadAxisCode::Right_Trigger) {
				// move the trigger range from [-1,1] to [0,1]
				result = result * 0.5f + 0.5f;
			}
			result = fabs(result) < deadZone ? 0.f : Util::sgn(result) * (fabs(result) - deadZone) / fullInterval;
			return result;
		}
	}

	void GLFWGamepad::setVibration(float intensityLeft, float intensityRight) {
		//LAZYENGINE_CORE_WARN("GLFWGamepad does not support Vibrations!");
	}

	const glm::vec2 GLFWGamepad::getVibration() const {
		//LAZYENGINE_CORE_WARN("GLFWGamepad does not support Vibrations!");
		return { 0.f, 0.f };
	}

	void GLFWGamepad::setTriggerVibration(float intensityLeft, float intensityRight) {
		//LAZYENGINE_CORE_WARN("GLFWGamepad does not support TriggerVibrations!");
	}

	const glm::vec2 GLFWGamepad::getTriggerVibration() const {
		//LAZYENGINE_CORE_WARN("GLFWGamepad does not support TriggerVibrations!");
		return { 0.f, 0.f };
	}

	const glm::vec2 GLFWGamepad::getStickValue(GamepadStickCode stick) const {
		glm::vec2 result = glm::vec2(NAN, NAN);
		// get values of the stick's two axes
		GamepadAxisCode xAxis, yAxis;
		if (stick == GamepadStickCode::Left_Stick) {
			xAxis = GamepadAxisCode::Left_Stick_X;
			yAxis = GamepadAxisCode::Left_Stick_Y;
		}
		else {
			xAxis = GamepadAxisCode::Right_Stick_X;
			yAxis = GamepadAxisCode::Right_Stick_Y;
		}
		result.x = getAxisValue(xAxis);
		// if the gamepad is not connected anymore, result.x will stay NAN. early out!
		if (result.x != result.x) return result;
		result.y = -getAxisValue(yAxis);

		return result;
	}

	void GLFWGamepad::pollEvents() {
		if (!m_callback) {
			LAZYENGINE_CORE_WARN("No event callback function set for gamepad {0}", getID());
		}

		for (int i = 0; i < m_buttonStates.size(); ++i) {
			bool isPressed = isButtonPressed((GamepadButtonCode)i);
			if (m_buttonStates[i] != isPressed) {
				if (isPressed) {
					GamepadButtonPressedEvent event(getID(), (GamepadButtonCode)i);
					m_callback(event);
				}
				else {
					GamepadButtonReleasedEvent event(getID(), (GamepadButtonCode)i);
					m_callback(event);
				}
				m_buttonStates[i] = isPressed;
			}
		}

		for (int i = 0; i < m_axisStates.size(); ++i) {
			float value = getAxisValue((GamepadAxisCode)i);

			if ((fabs(value) > Constants::EPSILON) && (fabs(m_axisStates[i]) < Constants::EPSILON)) {
				// fire an axis event
				GamepadAxisEvent event(getID(), (GamepadAxisCode)i, value);
				m_callback(event);
			}
			m_axisStates[i] = value;
		}
	}

	void GLFWGamepad::setEventCallbackFunc(const EventCallbackFn& callback) {
		m_callback = callback;
	}

	void GLFWGamepad::setDeadzone(GamepadAxisCode axis, float radius) {
		m_axisDeadzones[static_cast<size_t>(axis)] = radius;
	}

	float GLFWGamepad::getDeadzone(GamepadAxisCode axis) const {
		return m_axisDeadzones[static_cast<size_t>(axis)];
	}

	bool GLFWGamepad::operator==(const Gamepad& other) const {
		try {
			const GLFWGamepad& cast = dynamic_cast<const GLFWGamepad&>(other);
			return m_glfwID == cast.m_glfwID;
		}
		catch (const std::bad_cast&) {
			return false;
		}
	}

}
#endif