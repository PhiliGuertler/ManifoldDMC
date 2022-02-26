// ######################################################################### //
// ### WindowsGamepad.cpp ################################################## //
// ### implements WindowsGamepad.h                                       ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "WindowsGamepad.h"

#ifdef LAZYENGINE_PLATFORM_WINDOWS

#if !defined(LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS) && defined(LAZYENGINE_PLATFORM_GLFW3)
// WindowsGamepad is typedef'd to be a GLFWGamepad
#include "../GLFW/GLFWGamepad.cpp"

#else

#include "LazyEngine/Core/Constants.h"
#include "LazyEngine/Core/Util.h"
#include "LazyEngine/Core/Input/Input.h"


namespace LazyEngine {

	// ##################################################################### //
	// ### WindowsGamepad static functions and variables ################### //
	// ##################################################################### //

	WinWin::Gamepad::GamepadAdded_revoker WindowsGamepad::s_addedRevoker;
	WinWin::Gamepad::GamepadRemoved_revoker WindowsGamepad::s_removedRevoker;

	WinWin::RawGameController::RawGameControllerAdded_revoker WindowsGamepad::s_rawAddedRevoker;
	WinWin::RawGameController::RawGameControllerRemoved_revoker WindowsGamepad::s_rawRemovedRevoker;

	std::mutex WindowsGamepad::s_mutex;

	bool WindowsGamepad::s_isRegisteringRaw = true;


	static winrt::Windows::Gaming::Input::GamepadButtons gamepadButtonCodeToWindowsGamepadButtons(GamepadButtonCode button) {
		using namespace winrt::Windows::Gaming::Input;
		switch (button) {
		case GamepadButtonCode::XBox_Start: return GamepadButtons::Menu;
		case GamepadButtonCode::XBox_Back: return GamepadButtons::View;
		case GamepadButtonCode::XBox_A: return GamepadButtons::A;
		case GamepadButtonCode::XBox_B: return GamepadButtons::B;
		case GamepadButtonCode::XBox_X: return GamepadButtons::X;
		case GamepadButtonCode::XBox_Y: return GamepadButtons::Y;
		case GamepadButtonCode::DPad_Up: return GamepadButtons::DPadUp;
		case GamepadButtonCode::DPad_Down: return GamepadButtons::DPadDown;
		case GamepadButtonCode::DPad_Left: return GamepadButtons::DPadLeft;
		case GamepadButtonCode::DPad_Right: return GamepadButtons::DPadRight;
		case GamepadButtonCode::XBox_LB: return GamepadButtons::LeftShoulder;
		case GamepadButtonCode::XBox_RB: return GamepadButtons::RightShoulder;
		case GamepadButtonCode::XBox_Thumb_Left: return GamepadButtons::LeftThumbstick;
		case GamepadButtonCode::XBox_Thumb_Right: return GamepadButtons::RightThumbstick;
		default:
			return GamepadButtons::None;
		}
	}

	void WindowsGamepad::onGamepadAdded(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::Gamepad& gamepad) {
		std::lock_guard<std::mutex> lock(s_mutex);
		// get the id of the gamepad
		uint32_t index;
		if (WinWin::Gamepad::Gamepads().IndexOf(gamepad, index)) {
			s_isRegisteringRaw = false;
			Ref<Gamepad> wingamepad = createRef<WindowsXBoxGamepad>(index, gamepad);
			GamepadID lazyEngineID = Input::connectGamepad(wingamepad);
			wingamepad->setLazyEngineID(lazyEngineID);
			
			LAZYENGINE_CORE_INFO("XBox Game Controller Added!");
		}
	}

	void WindowsGamepad::onGamepadRemoved(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::Gamepad& gamepad) {
		std::lock_guard<std::mutex> lock(s_mutex);
		// get the id of the gamepad
		Ref<Gamepad> tmp = createRef<WindowsXBoxGamepad>(-1, gamepad);
		if (Input::disconnectGamepad(tmp)) {
			LAZYENGINE_CORE_INFO("XBox Game Controller Removed!");
		}
	}

	void WindowsGamepad::onRawGameControllerAdded(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::RawGameController& controller) {
		std::lock_guard<std::mutex> lock(s_mutex);

		// check if this gamepad has already been added before
		WindowsRawGamepad tmp(-1, controller);
		for (int i = 0; i < Input::getConnectedGamepadCount(); ++i) {
			if (Input::getGamepadIDOf(tmp) != -1) {
				LAZYENGINE_CORE_WARN("This Raw Gamepad already exists as a regular gamepad!");
				return;
			}
		}

		// get the id of the gamepad
		uint32_t index;
		if (WinWin::RawGameController::RawGameControllers().IndexOf(controller, index)) {
			s_isRegisteringRaw = true;
#ifdef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
			Ref<Gamepad> gamepad = GLFWGamepad::create(index);
			GamepadID lazyEngineID = Input::connectGamepad(gamepad);
			gamepad->setLazyEngineID(lazyEngineID);
#else
			Ref<Gamepad> gamepad = createRef<WindowsRawGamepad>(index, controller);
			GamepadID lazyEngineID = Input::connectGamepad(gamepad);
			std::static_pointer_cast<WindowsGamepad>(gamepad)->setLazyEngineID(lazyEngineID);
			
#endif
			LAZYENGINE_CORE_INFO("Raw Game Controller Added!");
		}
	}

	void WindowsGamepad::onRawGameControllerRemoved(const winrt::Windows::Foundation::IInspectable& sender, const WinWin::RawGameController& controller) {
		std::lock_guard<std::mutex> lock(s_mutex);

#ifdef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
		s_isRegisteringRaw = true;

#else
		// get the id of the gamepad
		Ref<Gamepad> tmp = createRef<WindowsRawGamepad>(-1, controller);
		if (Input::disconnectGamepad(tmp)) {
			LAZYENGINE_CORE_INFO("Raw Game Controller Removed!");
		}
#endif
	}


	void WindowsGamepad::init() {
		// set up the event handlers for (dis)connection events of gamepads
		s_addedRevoker = WinWin::Gamepad::GamepadAdded(winrt::auto_revoke, std::bind(&WindowsGamepad::onGamepadAdded, std::placeholders::_1, std::placeholders::_2));
		s_removedRevoker = WinWin::Gamepad::GamepadRemoved(winrt::auto_revoke, std::bind(&WindowsGamepad::onGamepadRemoved, std::placeholders::_1, std::placeholders::_2));

		// set up the event handlers for (dis)connection events of generic xinput gamepads
		s_rawAddedRevoker = WinWin::RawGameController::RawGameControllerAdded(winrt::auto_revoke, std::bind(&WindowsGamepad::onRawGameControllerAdded, std::placeholders::_1, std::placeholders::_2));
		s_rawRemovedRevoker = WinWin::RawGameController::RawGameControllerRemoved(winrt::auto_revoke, std::bind(&WindowsGamepad::onRawGameControllerRemoved, std::placeholders::_1, std::placeholders::_2));

		// Look for connected gamepads and add them
		auto gamepads = WinWin::Gamepad::Gamepads();
		for (const auto& gamepad : gamepads) {
			onGamepadAdded(nullptr, gamepad);
		}

#ifdef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
		GLFWGamepad::init();
#else
		// Look for connected generic xinput gamepads and add them
		auto rawGamepads = WinWin::RawGameController::RawGameControllers();
		for (const auto& rawGamepad : rawGamepads) {
			onRawGameControllerAdded(nullptr, rawGamepad);
		}
#endif
	}

	Ref<Gamepad> WindowsGamepad::create(GamepadID id) {
		if (s_isRegisteringRaw) {
#ifdef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
			//return createRef<GLFWGamepad>(id);
#else
			return createRef<WindowsRawGamepad>(id);
#endif
		}
		else {
			return createRef<WindowsXBoxGamepad>(id);
		}
	}

	Ref<Gamepad> WindowsGamepad::create(GamepadID internalID, GamepadID lazyEngineID) {
		if (s_isRegisteringRaw) {
#ifdef LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS
			//return createRef<GLFWGamepad>(internalID, lazyEngineID);
#else
			return createRef<WindowsRawGamepad>(internalID, lazyEngineID);
#endif
		}
		else {
			return createRef<WindowsXBoxGamepad>(internalID, lazyEngineID);
		}
	}


	// ##################################################################### //
	// ### WindowsGamepad ################################################## //
	// ##################################################################### //

	WindowsGamepad::WindowsGamepad(GamepadID id)
		: Gamepad(id)
		, m_winrtID(id)
		//, m_lazyEngineID(id)
		, m_oldButtonStates()
		, m_oldAxisValues()
		, m_axisDeadzones()
		, m_callback()
		, m_name("<Unknown Name>")
	{
		if (id == -1) return;

		std::fill(m_oldAxisValues.begin(), m_oldAxisValues.end(), 0.f);
		std::fill(m_axisDeadzones.begin(), m_axisDeadzones.end(), 0.1f);
		std::fill(m_oldButtonStates.begin(), m_oldButtonStates.end(), false);
	}

	bool WindowsGamepad::isConnected() const {
		// TODO: implement!
		return true;
	}

	const std::string WindowsGamepad::getName() const {
		return m_name;
	}

	const glm::vec2 WindowsGamepad::getStickValue(GamepadStickCode stick) const {
		switch (stick) {
		case GamepadStickCode::Left_Stick:
			return { getAxisValue(GamepadAxisCode::Left_Stick_X), getAxisValue(GamepadAxisCode::Left_Stick_Y) };
		case GamepadStickCode::Right_Stick:
			return { getAxisValue(GamepadAxisCode::Right_Stick_X), getAxisValue(GamepadAxisCode::Right_Stick_Y) };
		default:
			return { 0.f, 0.f };
		}
	}

	void WindowsGamepad::pollEvents() {
		// update buttons
		for (int i = 0; i < m_oldButtonStates.size(); ++i) {
			// The Guide button is not supported
			if (static_cast<GamepadButtonCode>(i) == GamepadButtonCode::XBox_Guide) continue;

			// convert the buttoncode to a windows specific button code
			bool buttonIsPressed = isButtonPressed((GamepadButtonCode)i);
			if (buttonIsPressed != m_oldButtonStates[i]) {
				// the current state differs from the old state, trigger an event
				if (buttonIsPressed) {
					// the button is now pressed
					GamepadButtonPressedEvent event(getLazyEngineID(), static_cast<GamepadButtonCode>(i));
					m_callback(event);
					m_oldButtonStates[i] = true;
				}
				else {
					// the button has been released
					GamepadButtonReleasedEvent event(getLazyEngineID(), static_cast<GamepadButtonCode>(i));
					m_callback(event);
					m_oldButtonStates[i] = false;
				}
			}
		}

		// check for axis events
		for (int i = 0; i < m_oldAxisValues.size(); ++i) {
			float currentValue = getAxisValue((GamepadAxisCode)i);
			if (fabs(currentValue) > Constants::EPSILON) {
				// this axis is currently being moved
				bool directionChanged = (Util::sgn(currentValue) != Util::sgn(m_oldAxisValues[i]));
				bool movementStarted = fabs(m_oldAxisValues[i]) < Constants::EPSILON;
				if (movementStarted || directionChanged) {
					GamepadAxisEvent event(getLazyEngineID(), (GamepadAxisCode)i, currentValue);
					m_callback(event);
				}
			}
			m_oldAxisValues[i] = currentValue;
		}
	}

	void WindowsGamepad::setEventCallbackFunc(const EventCallbackFn& callback) {
		m_callback = callback;
	}

	void WindowsGamepad::setDeadzone(GamepadAxisCode axis, float radius) {
		m_axisDeadzones[static_cast<size_t>(axis)] = radius;
	}

	float WindowsGamepad::getDeadzone(GamepadAxisCode axis) const {
		return m_axisDeadzones[static_cast<size_t>(axis)];
	}


	// ##################################################################### //
	// ### WindowsXBoxGamepad ############################################## //
	// ##################################################################### //

	WindowsXBoxGamepad::WindowsXBoxGamepad(GamepadID id)
		: WindowsXBoxGamepad(id, WinWin::Gamepad::Gamepads().GetAt(id))
	{
		// empty
	}

	WindowsXBoxGamepad::WindowsXBoxGamepad(GamepadID internalID, GamepadID lazyEngineID)
		: WindowsXBoxGamepad(internalID, WinWin::Gamepad::Gamepads().GetAt(internalID))
	{
		setLazyEngineID(lazyEngineID);
	}

	WindowsXBoxGamepad::WindowsXBoxGamepad(GamepadID id, const WinWin::Gamepad& gamepad)
		: WindowsGamepad(id)
		, m_gamepad(gamepad)
	{
		if (id == -1) return;

		WinWin::RawGameController rawController = WinWin::RawGameController::FromGameController(m_gamepad);
		if (!rawController) {
			LAZYENGINE_CORE_ERROR("Error during initialization of Controller!");
		}
		else {
			m_name = winrt::to_string(rawController.DisplayName());
		}
	}

	bool WindowsXBoxGamepad::isButtonPressed(GamepadButtonCode button) const {
		using namespace winrt::Windows::Gaming::Input;

		if (button == GamepadButtonCode::XBox_Guide) {
			// special case for Guide: This button is not supported
			return false;
		}
		else if (button == GamepadButtonCode::XBox_LT) {
			// special case for left trigger
			return getAxisValue(GamepadAxisCode::Left_Trigger) > Constants::EPSILON;
		}
		else if (button == GamepadButtonCode::XBox_RT) {
			// special case for right trigger
			return getAxisValue(GamepadAxisCode::Right_Trigger) > Constants::EPSILON;
		}

		// get the current state of the gamepad
		GamepadButtons state = m_gamepad.GetCurrentReading().Buttons;
		// convert the argument to a windows specific button code
		GamepadButtons windowsCode = gamepadButtonCodeToWindowsGamepadButtons(button);

		// use bitwise and to isolate the value of the gamepad's state
		GamepadButtons result = state & windowsCode;
		return result == windowsCode;
	}

	float WindowsXBoxGamepad::getAxisValue(GamepadAxisCode axis) const {
		float result = 0.f;
		// get the deadzone for this axis
		float deadZone = m_axisDeadzones[static_cast<size_t>(axis)];
		float fullInterval = 1.f - deadZone;

		// retrieve the current value of this axis
		switch (axis) {
		case GamepadAxisCode::Left_Stick_X:
			result = static_cast<float>(m_gamepad.GetCurrentReading().LeftThumbstickX);
			break;
		case GamepadAxisCode::Left_Stick_Y:
			result = static_cast<float>(m_gamepad.GetCurrentReading().LeftThumbstickY);
			break;
		case GamepadAxisCode::Right_Stick_X:
			result = static_cast<float>(m_gamepad.GetCurrentReading().RightThumbstickX);
			break;
		case GamepadAxisCode::Right_Stick_Y:
			result = static_cast<float>(m_gamepad.GetCurrentReading().RightThumbstickY);
			break;
		case GamepadAxisCode::Left_Trigger:
			result = static_cast<float>(m_gamepad.GetCurrentReading().LeftTrigger);
			break;
		case GamepadAxisCode::Right_Trigger:
			result = static_cast<float>(m_gamepad.GetCurrentReading().RightTrigger);
			break;
		default:
			return 0.f;
		}

		// map the resulting value minus the deadzone to [-1, 1] or [0, 1] if it is a trigger
		result = fabs(result) < deadZone ? 0.f : Util::sgn(result) * (fabs(result) - deadZone) / fullInterval;
		return result;
	}

	void WindowsXBoxGamepad::setVibration(float intensityLeft, float intensityRight) {
		using namespace winrt::Windows::Gaming::Input;

		GamepadVibration vibration = m_gamepad.Vibration();
		// don't change the value of the left motor, if intensityLeft is NAN
		vibration.LeftMotor = intensityLeft != intensityLeft ? vibration.LeftMotor : intensityLeft;
		// don't change the value of the right motor, if intensityRight is NAN
		vibration.RightMotor = intensityRight != intensityRight ? vibration.LeftMotor : intensityRight;

		m_gamepad.Vibration(vibration);
	}

	const glm::vec2 WindowsXBoxGamepad::getVibration() const {
		auto vibration = m_gamepad.Vibration();
		return { vibration.LeftMotor, vibration.RightMotor };
	}

	void WindowsXBoxGamepad::setTriggerVibration(float intensityLeft, float intensityRight) {
		using namespace winrt::Windows::Gaming::Input;

		GamepadVibration vibration = m_gamepad.Vibration();
		// don't change the value of the left motor, if intensityLeft is NAN
		vibration.LeftTrigger = intensityLeft != intensityLeft ? vibration.LeftTrigger : intensityLeft;
		// don't change the value of the right motor, if intensityRight is NAN
		vibration.RightTrigger = intensityRight != intensityRight ? vibration.LeftTrigger : intensityRight;

		m_gamepad.Vibration(vibration);
	}

	const glm::vec2 WindowsXBoxGamepad::getTriggerVibration() const {
		auto vibration = m_gamepad.Vibration();
		return { vibration.LeftTrigger, vibration.RightTrigger };
	}

	bool WindowsXBoxGamepad::operator==(const Gamepad& other) const {
		try {
			const WindowsXBoxGamepad& cast = dynamic_cast<const WindowsXBoxGamepad&>(other);
			return m_gamepad == cast.m_gamepad;
		}
		catch (const std::bad_cast&) {
			try {
				const WindowsRawGamepad& cast = dynamic_cast<const WindowsRawGamepad&>(other);
				bool result = operator==(cast);
				return result;
			}
			catch (const std::bad_cast&) {
				return false;
			}
		}
	}

	bool WindowsXBoxGamepad::operator==(const WindowsRawGamepad& other) const {
		const auto rawGamepad = WinWin::RawGameController::FromGameController(m_gamepad);
		return rawGamepad == other.m_gamepad;
	}

	
	// ##################################################################### //
	// ### WindowsRawGamepad ############################################### //
	// ##################################################################### //

	static const std::string buttonLabelToString(WinWin::GameControllerButtonLabel label) {
		typedef WinWin::GameControllerButtonLabel Label;
		switch (label) {
		case Label::Back: return "Back";
		case Label::Circle: return "Circle";
		case Label::Cross: return "Cross";
		case Label::DialLeft: return "DialLeft";
		case Label::DialRight: return "DialRight";
		case Label::Down: return "Down";
		case Label::DownLeftArrow: return "DownLeftArrow";
		case Label::Left: return "Left";
		case Label::Left1: return "Left1";
		case Label::Left2: return "Left2";
		case Label::Left3: return "Left3";
		case Label::LeftBumper: return "LeftBumper";
		case Label::LeftStickButton: return "LeftStickButton";
		case Label::LeftTrigger: return "LeftTrigger";
		case Label::LetterA: return "LetterA";
		case Label::LetterB: return "LetterB";
		case Label::LetterC: return "LetterC";
		case Label::LetterL: return "LetterL";
		case Label::LetterR: return "LetterR";
		case Label::LetterX: return "LetterX";
		case Label::LetterY: return "LetterY";
		case Label::LetterZ: return "LetterZ";
		case Label::Menu: return "Menu";
		case Label::Minus: return "Minus";
		case Label::Mode: return "Mode";
		case Label::None: return "None";
		case Label::Options: return "Options";
		case Label::Paddle1: return "Paddle1";
		case Label::Paddle2: return "Paddle2";
		case Label::Paddle3: return "Paddle3";
		case Label::Paddle4: return "Paddle4";
		case Label::Plus: return "Plus";
		case Label::Right: return "Right";
		case Label::Right1: return "Right1";
		case Label::Right2: return "Right2";
		case Label::Right3: return "Right3";
		case Label::RightBumper: return "RightBumper";
		case Label::RightStickButton: return "RightStickButton";
		case Label::RightTrigger: return "RightTrigger";
		case Label::Select: return "Select";
		case Label::Share: return "Share";
		case Label::Square: return "Square";
		case Label::Start: return "Start";
		case Label::Suspension: return "Suspension";
		case Label::Triangle: return "Triangle";
		case Label::Up: return "Up";
		case Label::View: return "View";
		case Label::XboxA: return "XboxA";
		case Label::XboxB: return "XboxB";
		case Label::XboxBack: return "XboxBack";
		case Label::XboxDown: return "XboxDown";
		case Label::XboxLeft: return "XboxLeft";
		case Label::XboxLeftBumper: return "XboxLeftBumper";
		case Label::XboxLeftStickButton: return "XboxLeftStickButton";
		case Label::XboxLeftTrigger: return "XboxLeftTrigger";
		case Label::XboxMenu: return "XboxMenu";
		case Label::XboxPaddle1: return "XboxPaddle1";
		case Label::XboxPaddle2: return "XboxPaddle2";
		case Label::XboxPaddle3: return "XboxPaddle3";
		case Label::XboxPaddle4: return "XboxPaddle4";
		case Label::XboxRight: return "XboxRight";
		case Label::XboxRightBumper: return "XboxRightBumper";
		case Label::XboxRightStickButton: return "XboxRightStickButton";
		case Label::XboxRightTrigger: return "XboxRightTrigger";
		case Label::XboxStart: return "XboxStart";
		case Label::XboxUp: return "XboxUp";
		case Label::XboxView: return "XboxView";
		case Label::XboxX: return "XboxX";
		case Label::XboxY: return "XboxY";
		default:
			return "<Unknown>";
		}
	}

	static GamepadButtonCode rawButtonLabelToGamepadButtonCode(WinWin::GameControllerButtonLabel label) {
		switch (label) {
		case WinWin::GameControllerButtonLabel::XboxA: 
		case WinWin::GameControllerButtonLabel::Cross:
		case WinWin::GameControllerButtonLabel::LetterA:
			return GamepadButtonCode::XBox_A;
		case WinWin::GameControllerButtonLabel::XboxB:
		case WinWin::GameControllerButtonLabel::Circle:
		case WinWin::GameControllerButtonLabel::LetterB:
			return GamepadButtonCode::XBox_B;
		case WinWin::GameControllerButtonLabel::XboxX:
		case WinWin::GameControllerButtonLabel::Square:
		case WinWin::GameControllerButtonLabel::LetterY:
			return GamepadButtonCode::XBox_X;
		case WinWin::GameControllerButtonLabel::XboxY:
		case WinWin::GameControllerButtonLabel::Triangle:
		case WinWin::GameControllerButtonLabel::LetterX:
			return GamepadButtonCode::XBox_Y;
		case WinWin::GameControllerButtonLabel::XboxLeftBumper:
		case WinWin::GameControllerButtonLabel::LeftBumper:
		case WinWin::GameControllerButtonLabel::Left1:
		case WinWin::GameControllerButtonLabel::LetterL:
			return GamepadButtonCode::XBox_LB;
		case WinWin::GameControllerButtonLabel::XboxRightBumper:
		case WinWin::GameControllerButtonLabel::RightBumper:
		case WinWin::GameControllerButtonLabel::Right1:
		case WinWin::GameControllerButtonLabel::LetterR:
			return GamepadButtonCode::XBox_RB;
		case WinWin::GameControllerButtonLabel::XboxLeft:
		case WinWin::GameControllerButtonLabel::Left:
			return GamepadButtonCode::DPad_Left;
		case WinWin::GameControllerButtonLabel::XboxRight:
		case WinWin::GameControllerButtonLabel::Right:
			return GamepadButtonCode::DPad_Right;
		case WinWin::GameControllerButtonLabel::XboxUp:
		case WinWin::GameControllerButtonLabel::Up:
			return GamepadButtonCode::DPad_Up;
		case WinWin::GameControllerButtonLabel::XboxDown:
		case WinWin::GameControllerButtonLabel::Down:
			return GamepadButtonCode::DPad_Down;
		case WinWin::GameControllerButtonLabel::XboxLeftStickButton:
		case WinWin::GameControllerButtonLabel::LeftStickButton:
		case WinWin::GameControllerButtonLabel::Left3:
			return GamepadButtonCode::XBox_Thumb_Left;
		case WinWin::GameControllerButtonLabel::XboxRightStickButton:
		case WinWin::GameControllerButtonLabel::RightStickButton:
		case WinWin::GameControllerButtonLabel::Right3:
			return GamepadButtonCode::XBox_Thumb_Right;
		case WinWin::GameControllerButtonLabel::XboxView:
		case WinWin::GameControllerButtonLabel::XboxBack:
		case WinWin::GameControllerButtonLabel::View:
		case WinWin::GameControllerButtonLabel::Back:
		case WinWin::GameControllerButtonLabel::Select:
		case WinWin::GameControllerButtonLabel::Share:
		case WinWin::GameControllerButtonLabel::Minus:
			return GamepadButtonCode::XBox_Back;
		case WinWin::GameControllerButtonLabel::XboxStart:
		case WinWin::GameControllerButtonLabel::Start:
		case WinWin::GameControllerButtonLabel::Options:
		case WinWin::GameControllerButtonLabel::Plus:
			return GamepadButtonCode::XBox_Start;
		case WinWin::GameControllerButtonLabel::XboxLeftTrigger:
		case WinWin::GameControllerButtonLabel::LeftTrigger:
		case WinWin::GameControllerButtonLabel::Left2:
			return GamepadButtonCode::XBox_LT;
		case WinWin::GameControllerButtonLabel::XboxRightTrigger:
		case WinWin::GameControllerButtonLabel::RightTrigger:
		case WinWin::GameControllerButtonLabel::Right2:
			return GamepadButtonCode::XBox_RT;
		default:
			//LAZYENGINE_CORE_INFO("Button: {0}", buttonLabelToString(label));
			return GamepadButtonCode::ButtonCount;
		}
	}

	WindowsRawGamepad::RawGamepadState WindowsRawGamepad::getRawGamepadState() const {
		auto buttonCount = m_gamepad.ButtonCount();
		auto axisCount = m_gamepad.AxisCount();
		auto switchCount = m_gamepad.SwitchCount();

		Scope<bool[]> buttonStatesHeap = createScope<bool[]>(buttonCount);
		winrt::array_view<bool> buttonStates(buttonStatesHeap.get(), buttonStatesHeap.get() + buttonCount);

		RawGamepadState result;
		result.axisStates = std::vector<double>(axisCount);
		result.switchStates = std::vector<WinWin::GameControllerSwitchPosition>(switchCount);

		m_gamepad.GetCurrentReading(buttonStates, result.switchStates, result.axisStates);

		result.buttonStates = std::vector<bool>(static_cast<size_t>(GamepadButtonCode::ButtonCount));
		// sort the buttonstates by their button label
		for (int i = 0; i < buttonCount; ++i) {
			auto id = rawButtonLabelToGamepadButtonCode(m_gamepad.GetButtonLabel(i));
			if (id == GamepadButtonCode::ButtonCount) {
				continue;
			}
			result.buttonStates[static_cast<size_t>(id)] = buttonStates[i];
		}
		return result;
	}

	WindowsRawGamepad::WindowsRawGamepad(GamepadID id)
		: WindowsRawGamepad(id, WinWin::RawGameController::RawGameControllers().GetAt(id))
	{
		// empty
	}

	WindowsRawGamepad::WindowsRawGamepad(GamepadID internalID, GamepadID lazyEngineID)
		: WindowsRawGamepad(internalID, WinWin::RawGameController::RawGameControllers().GetAt(internalID))
	{
		setLazyEngineID(lazyEngineID);
	}

	WindowsRawGamepad::WindowsRawGamepad(GamepadID id, const WinWin::RawGameController& gamepad)
		: WindowsGamepad(id)
		, m_gamepad(gamepad)
	{
		if (id == -1) return;
		auto hwID = gamepad.HardwareProductId();
		auto vendor = gamepad.HardwareVendorId();
		std::stringstream ss;
		ss << winrt::to_string(gamepad.DisplayName()) << " [" << hwID << "] (" << vendor << ")";
		m_name = ss.str();
	}

	bool WindowsRawGamepad::isButtonPressed(GamepadButtonCode button) const {
		using namespace winrt::Windows::Gaming::Input;

		if (button == GamepadButtonCode::XBox_Guide) {
			// special case for Guide: This button is not supported
			return false;
		}
		else if (button == GamepadButtonCode::XBox_LT) {
			// special case for left trigger
			return getAxisValue(GamepadAxisCode::Left_Trigger) > Constants::EPSILON;
		}
		else if (button == GamepadButtonCode::XBox_RT) {
			// special case for right trigger
			return getAxisValue(GamepadAxisCode::Right_Trigger) > Constants::EPSILON;
		}

		// get the current state of the gamepad
		auto state = getRawGamepadState();

		return state.buttonStates[static_cast<size_t>(button)];
	}

	float WindowsRawGamepad::getAxisValue(GamepadAxisCode axis) const {
		float result = 0.f;
		// get the deadzone for this axis
		float deadZone = m_axisDeadzones[static_cast<size_t>(axis)];
		float fullInterval = 1.f - deadZone;

		auto state = getRawGamepadState();
		if (state.axisStates.size() > static_cast<size_t>(axis)) {
			result = static_cast<float>(state.axisStates[static_cast<size_t>(axis)]);
		}
		else {
			if (axis == GamepadAxisCode::Left_Trigger) {
				//result = isButtonPressed(GamepadButtonCode::XBox_LT);
			}
			else if (axis == GamepadAxisCode::Right_Trigger) {
				//result = isButtonPressed(GamepadButtonCode::XBox_RT);
			}
		}

		// map the resulting value minus the deadzone to [-1, 1] or [0, 1] if it is a trigger
		result = fabs(result) < deadZone ? 0.f : Util::sgn(result) * (fabs(result) - deadZone) / fullInterval;
		return result;
	}

	void WindowsRawGamepad::setVibration(float intensityLeft, float intensityRight) {
		//LAZYENGINE_CORE_WARN("WindowsRawGamepad does not support Vibrations!");
		/*	These lines are not working because some library is not linked...
		auto motors = m_gamepad.ForceFeedbackMotors();
		for (auto& motor : motors) {
			if (motor.IsEnabled()) {
				WinWin::ForceFeedback::IForceFeedbackEffect effect;
				//auto currentState = effect.State();
				effect.Gain(intensityLeft);
				motor.LoadEffectAsync(effect);
				effect.Start();
			}
		}
		*/
	}

	const glm::vec2 WindowsRawGamepad::getVibration() const {
		//LAZYENGINE_CORE_WARN("WindowsRawGamepad does not support Vibrations!");
		return { 0.f, 0.f };
	}

	void WindowsRawGamepad::setTriggerVibration(float intensityLeft, float intensityRight) {
		//LAZYENGINE_CORE_WARN("WindowsRawGamepad does not support TriggerVibrations!");
	}

	const glm::vec2 WindowsRawGamepad::getTriggerVibration() const {
		//LAZYENGINE_CORE_WARN("WindowsRawGamepad does not support TriggerVibrations!");
		return { 0.f, 0.f };
	}

	bool WindowsRawGamepad::operator==(const Gamepad& other) const {
		try {
			const WindowsRawGamepad& cast = dynamic_cast<const WindowsRawGamepad&>(other);
			return m_gamepad == cast.m_gamepad;
		}
		catch (const std::bad_cast&) {
			try {
				const WindowsXBoxGamepad& cast = dynamic_cast<const WindowsXBoxGamepad&>(other);
				bool result = operator==(cast);
				return result;
			}
			catch (const std::bad_cast&) {
				return false;
			}
		}
	}

	bool WindowsRawGamepad::operator==(const WindowsXBoxGamepad& other) const {
		const auto rawGamepadOther = WinWin::RawGameController::FromGameController(other.m_gamepad);
		return rawGamepadOther == m_gamepad;
	}
}

#if defined(LAZYENGINE_USE_GLFW_FOR_RAW_CONTROLLERS)
#include "../GLFW/GLFWGamepad.cpp"
#endif

#endif	// LAZYENGINE_PLATFORM_GLFW3


#endif	// LAZYENGINE_PLATFORM_WINDOWS