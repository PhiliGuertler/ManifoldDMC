#pragma once

// ######################################################################### //
// ### WindowsInput.h ###################################################### //
// ### implements the Input class for Windows using glfw.                ### //
// ######################################################################### //

// defines the platform
#include "LazyEngine/Core/Core.h"

#ifdef LAZYENGINE_PLATFORM_GLFW3
#include "LazyEngine/Core/Input/Input.h"

#include <map>

namespace LazyEngine {

	/**
	 *	The Windows implementation for the Input class using glfw
	 */
	class GLFWInput : public Input {
	protected:

		// ################################################################# //
		// ### Keyboard Input ############################################## //
		// ################################################################# //
		/**
		 *	Checks the state of a key at this moment for the Windows platform.
		 *	@param keyCode: the key to be checked.
		 */
		virtual bool isKeyPressedImpl(KeyCode keycode) const override;


		// ################################################################# //
		// ### Mouse Input ################################################# //
		// ################################################################# //
		/**
		 *	Checks the state of a mouse button at this moment for the Windows platform.
		 *	@param button: the button to be checked.
		 */
		virtual bool isMouseButtonPressedImpl(MouseButtonCode button) const override;
		/**
		 *	returns the current x-coordinate of the mouse for the Windows platform.
		 */
		virtual float getMouseXImpl() const override;
		/**
		 *	returns the current y-coordinate of the mouse for the Windows platform.
		 */
		virtual float getMouseYImpl() const override;
		/**
		 *	returns the mouse position as a pair of [x,y] for the Windows platform.
		 */
		virtual const glm::vec2 getMousePositionImpl() const override;


		// ################################################################# //
		// ### Gamepad Input ############################################### //
		// ################################################################# //

		virtual GamepadID connectGamepadImpl(Ref<Gamepad> gamepad) override;

		virtual bool disconnectGamepadImpl(Ref<Gamepad> gamepad) override;

		virtual bool disconnectGamepadImpl(GamepadID id) override;

		virtual Ref<Gamepad> getGamepadImpl(GamepadID id) const override;

		virtual Ref<Gamepad> getFirstGamepadImpl() const override;
		
		virtual GamepadID getGamepadIDOfImpl(const Gamepad& gamepad) const override;

		virtual int getConnectedGamepadCountImpl() const override;

		virtual void setGamepadEventCallbackImpl(const EventCallbackFn& callback) override;

		virtual void pollEventsImpl() override;

	private:
		std::map<GamepadID, Ref<Gamepad>> m_connectedGamepads;

		EventCallbackFn m_eventCallbackGamepad;

		std::mutex m_mutex;
	};

}
#endif