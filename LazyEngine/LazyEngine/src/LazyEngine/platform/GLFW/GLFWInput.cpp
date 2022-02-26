// ######################################################################### //
// ### GLFWInput.cpp ####################################################### //
// ### implements GLFWInput.h                                            ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "GLFWInput.h"
#include "LazyEngine/Core/Input/Gamepad.h"

#ifdef LAZYENGINE_PLATFORM_GLFW3

#include <GLFW/glfw3.h>

#include "LazyEngine/Core/Application.h"
#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	bool GLFWInput::isKeyPressedImpl(KeyCode keycode) const {
		LAZYENGINE_PROFILE_FUNCTION();

		auto window = static_cast<GLFWwindow*>(Application::getInstance().getWindow().getNativeWindow());
		auto keyState = glfwGetKey(window, static_cast<int>(keycode));
		return keyState == GLFW_PRESS || keyState == GLFW_REPEAT;
	}

	bool GLFWInput::isMouseButtonPressedImpl(MouseButtonCode button) const {
		LAZYENGINE_PROFILE_FUNCTION();

		auto window = static_cast<GLFWwindow*>(Application::getInstance().getWindow().getNativeWindow());
		auto buttonState = glfwGetMouseButton(window, static_cast<int>(button));
		return buttonState == GLFW_PRESS;
	}

	const glm::vec2 GLFWInput::getMousePositionImpl() const {
		LAZYENGINE_PROFILE_FUNCTION();

		auto& window = Application::getInstance().getWindow();
		auto nativeWindow = static_cast<GLFWwindow*>(window.getNativeWindow());
		double xpos, ypos;
		glfwGetCursorPos(nativeWindow, &xpos, &ypos);
		return { (float)xpos, (float)ypos };
	}

	float GLFWInput::getMouseXImpl() const {
		LAZYENGINE_PROFILE_FUNCTION();

		return getMousePositionImpl().x;
	}

	float GLFWInput::getMouseYImpl() const {
		LAZYENGINE_PROFILE_FUNCTION();

		return getMousePositionImpl().y;
	}

	GamepadID GLFWInput::connectGamepadImpl(Ref<Gamepad> gamepad) {
		std::scoped_lock lock(m_mutex);

		// find the first available GamepadID in the list of connected gamepads
		GamepadID emptySpace;
		for (emptySpace = 0; emptySpace < m_connectedGamepads.size(); ++emptySpace) {
			if (m_connectedGamepads.find(emptySpace) == m_connectedGamepads.end()) {
				// this place is free, the gamepad can be inserted here
				break;
			}
		}

		// set the gamepad's event callback function to this class's event callback
		gamepad->setEventCallbackFunc(m_eventCallbackGamepad);

		// add the gamepad to the list of connected gamepads
		m_connectedGamepads[emptySpace] = gamepad;

		LAZYENGINE_CORE_INFO("Gamepad connected: id={0}, name={1}", emptySpace, gamepad->getName());

		// trigger a gamepad connection event
		GamepadConnectionEvent event(emptySpace, true);
		if (m_eventCallbackGamepad) {
			m_eventCallbackGamepad(event);
		}

		// return the index of the gamepad, that can be used to retrieve it later
		return emptySpace;
	}

	bool GLFWInput::disconnectGamepadImpl(Ref<Gamepad> gamepad) {
		GamepadID location = getGamepadIDOfImpl(*gamepad);
		if (location == -1) {
			// this gamepad is not registered
			return false;
		}
		return disconnectGamepadImpl(location);
	}

	bool GLFWInput::disconnectGamepadImpl(GamepadID gamepadID) {
		std::scoped_lock lock(m_mutex);

		auto gamepadIter = m_connectedGamepads.find(gamepadID);
		if (gamepadIter == m_connectedGamepads.end()) {
			// this gamepadID is not mapped to a valid gamepad
			return false;
		}

		// get the gamepad to print its infos before erasing it from the list
		auto gamepad = gamepadIter->second;

		// remove the gamepad from the list of connected gamepads
		m_connectedGamepads.erase(gamepadID);

		LAZYENGINE_CORE_INFO("Gamepad disconnected: id={0}, name={1}", gamepad->getID(), gamepad->getName());
		
		// trigger a gamepad disconnection event
		GamepadConnectionEvent event(gamepadID, false);
		if (m_eventCallbackGamepad) {
			m_eventCallbackGamepad(event);
		}

		return true;
	}

	Ref<Gamepad> GLFWInput::getGamepadImpl(GamepadID id) const {
		LAZYENGINE_PROFILE_FUNCTION();

		if (m_connectedGamepads.find(id) == m_connectedGamepads.end()) {
			//LAZYENGINE_CORE_ERROR("Gamepad with id {0} is not connected!", id);
			return nullptr;
		}

		return m_connectedGamepads.at(id);
	}

	Ref<Gamepad> GLFWInput::getFirstGamepadImpl() const {
		LAZYENGINE_PROFILE_FUNCTION();

		if (getConnectedGamepadCountImpl() == 0) return nullptr;
		return m_connectedGamepads.begin()->second;
	}

	GamepadID GLFWInput::getGamepadIDOfImpl(const Gamepad& gamepad) const {
		for (const auto& pad : m_connectedGamepads) {
			if (*(pad.second) == gamepad) {
				return pad.first;
			}
		}
		return -1;
	}

	int GLFWInput::getConnectedGamepadCountImpl() const {
		LAZYENGINE_PROFILE_FUNCTION();

		return (int)m_connectedGamepads.size();
	}

	void GLFWInput::setGamepadEventCallbackImpl(const EventCallbackFn& callback) {
		LAZYENGINE_PROFILE_FUNCTION();

		m_eventCallbackGamepad = callback;
		for (auto gamepad : m_connectedGamepads) {
			gamepad.second->setEventCallbackFunc(m_eventCallbackGamepad);
		}
	}

	void GLFWInput::pollEventsImpl() {
		std::scoped_lock lock(m_mutex);
		LAZYENGINE_PROFILE_FUNCTION();

		for (auto& gamepad : m_connectedGamepads) {
			gamepad.second->pollEvents();
		}
	}
}
#endif