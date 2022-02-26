#pragma once

// ######################################################################### //
// ### WindowsGamepad.h #################################################### //
// ### implements the Gamepad class for Windows using glfw.              ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#ifdef LAZYENGINE_PLATFORM_WINDOWS

#define LAZYENGINE_PLATFORM_GLFW3
// TODO: make this line below work, so that WindowsXBoxGamepad and GLFWGamepad can be used at the same time!
//#define LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS

#if defined(LAZYENGINE_PLATFORM_GLFW3) && !defined(LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS)
#include "../GLFW/GLFWGamepad.h"

namespace LazyEngine {
	typedef GLFWGamepad WindowsGamepad;
}

#elif defined(LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS)
#define LAZYENGINE_PLATFORM_GLFW3
#include "../GLFW/GLFWGamepad.h"

namespace LazyEngine {
	class WindowsRawGamepad;
}

#else

namespace LazyEngine {
	class WindowsRawGamepad;
}

#endif

#if defined(LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS) || !defined(LAZYENGINE_PLATFORM_GLFW3)

#include "LazyEngine/Core/Input/Gamepad.h"

#pragma comment(lib, "windowsapp")
#include <concrt.h>
#include <winrt/Windows.Gaming.Input.h>

#define WinWin winrt::Windows::Gaming::Input

namespace LazyEngine {



	class WindowsGamepad : public Gamepad {
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

		/**
		 *	checks if this gamepad is still connected.
		 */
		virtual bool isConnected() const override;

		/**
		 *	returns the name of this gamepad
		 */
		virtual const std::string getName() const override;

		/**
		 *	return the values of a stick in x- and y-direction
		 *	@param stick: The stick to query
		 */
		virtual const glm::vec2 getStickValue(GamepadStickCode stick) const override;

		/**
		 *	polls all events of this gamepad.
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

	protected:
		WindowsGamepad(GamepadID id);

	protected:
		GamepadID m_winrtID;
		//GamepadID m_lazyEngineID;

		std::array<bool, static_cast<size_t>(GamepadButtonCode::ButtonCount)> m_oldButtonStates;
		std::array<float, 6> m_oldAxisValues;

		std::array<float, 6> m_axisDeadzones;
		EventCallbackFn m_callback;

		std::string m_name;

	private:
		static WinWin::Gamepad::GamepadAdded_revoker s_addedRevoker;
		static WinWin::Gamepad::GamepadRemoved_revoker s_removedRevoker;

		static WinWin::RawGameController::RawGameControllerAdded_revoker s_rawAddedRevoker;
		static WinWin::RawGameController::RawGameControllerRemoved_revoker s_rawRemovedRevoker;

		static std::mutex s_mutex;

		static bool s_isRegisteringRaw;

		static void onGamepadAdded(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::Gamepad& gamepad);
		static void onGamepadRemoved(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::Gamepad& gamepad);

		static void onRawGameControllerAdded(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::RawGameController& controller);
		static void onRawGameControllerRemoved(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::RawGameController& controller);
	};

	class WindowsXBoxGamepad : public WindowsGamepad {
	public:
		WindowsXBoxGamepad(GamepadID id);
		WindowsXBoxGamepad(GamepadID internalID, GamepadID lazyEngineID);
		WindowsXBoxGamepad(GamepadID id, const WinWin::Gamepad& gamepad);

		/**
		 *	checks the state of a button at this moment
		 *	@param button: The button to query
		 */
		virtual bool isButtonPressed(GamepadButtonCode button) const override;

		/**
		 *	returns the value of an axis. The value will be in [-1.f, 1.f]
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
		 *	The equals operator
		 */
		virtual bool operator==(const Gamepad& other) const override;

		bool operator==(const WindowsRawGamepad& other) const;

	private:
//#ifndef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
		friend class WindowsRawGamepad;
//#endif

		WinWin::Gamepad m_gamepad;

	};

//#ifndef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
	class WindowsRawGamepad : public WindowsGamepad {
	public:
		WindowsRawGamepad(GamepadID id);
		WindowsRawGamepad(GamepadID internalID, GamepadID lazyEngineID);
		WindowsRawGamepad(GamepadID id, const WinWin::RawGameController& gamepad);

		/**
		 *	checks the state of a button at this moment
		 *	@param button: The button to query
		 */
		virtual bool isButtonPressed(GamepadButtonCode button) const override;

		/**
		 *	returns the value of an axis. The value will be in [-1.f, 1.f]
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
		 *	The equals operator
		 */
		virtual bool operator==(const Gamepad& other) const override;

		virtual bool operator==(const WindowsXBoxGamepad& other) const;

	private:
		friend class WindowsXBoxGamepad;

		struct RawGamepadState {
			std::vector<bool> buttonStates;
			std::vector<double> axisStates;
			std::vector<WinWin::GameControllerSwitchPosition> switchStates;
		};

		RawGamepadState getRawGamepadState() const;

		WinWin::RawGameController m_gamepad;

	};
//#endif // LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS

}
#endif // LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS || !LAZYENGINE_PLATFORM_GLFW3
#endif // LAZYENGINE_PLATFORM_WINDOWS