#pragma once

namespace LazyEngine {
	
	typedef int GamepadID;

	enum class GamepadButtonCode {
		// --- Buttons --- //
		// bottom button
		XBox_A = 0,
		Nintendo_B = XBox_A,
		Sony_Cross = XBox_A,
		// right button
		XBox_B = 1,
		Nintendo_A = XBox_B,
		Sony_Circle = XBox_B,
		// left button
		XBox_X = 2,
		Nintendo_Y = XBox_X,
		Sony_Square = XBox_X,
		// top button
		XBox_Y = 3,
		Nintendo_X = XBox_Y,
		Sony_Triangle = XBox_Y,

		// -- Bumpers -- //
		// left bumper
		XBox_LB = 4,
		Nintendo_L = XBox_LB,
		Sony_L1 = XBox_LB,
		// right bumper
		XBox_RB = 5,
		Nintendo_R = XBox_RB,
		Sony_R1 = XBox_RB,

		// --- Options Buttons --- //
		// select button
		XBox_Back = 6,
		Nintendo_Minus = XBox_Back,
		Nintendo_Select = XBox_Back,
		Sony_Select = XBox_Back,
		// start button
		XBox_Start = 7,
		Nintendo_Plus = XBox_Start,
		Nintendo_Start = XBox_Start,
		Sony_Options = XBox_Start,
		Sony_Start = XBox_Start,

		// home button
		XBox_Guide = 8,
		Nintendo_Home = XBox_Guide,
		Sony_PS = XBox_Guide,

		// --- Sticks --- //
		// Press of the left Stick
		XBox_Thumb_Left = 9,
		Nintendo_Thumb_Left = XBox_Thumb_Left,
		Sony_L3 = XBox_Thumb_Left,
		// Press of the right Stick
		XBox_Thumb_Right = 10,
		Nintendo_Thumb_Right = XBox_Thumb_Right,
		Sony_R3 = XBox_Thumb_Right,

		// --- DigiPad Buttons --- //
		DPad_Up = 11,
		DPad_Right = 12,
		DPad_Down = 13,
		DPad_Left = 14,


		// --- Shoulder Buttons --- //
		// -- Triggers -- //
		// left trigger
		XBox_LT = 15,
		Nintendo_ZL = XBox_LT,
		Sony_L2 = XBox_LT,
		// right trigger
		XBox_RT = 16,
		Nintendo_ZR = XBox_RT,
		Sony_R2 = XBox_RT,

		ButtonCount = XBox_RT+1
	};
	
	enum class GamepadStickCode {
		Left_Stick = 0,
		Right_Stick = 1
	};

	enum class GamepadAxisCode {
		Left_Stick_X = 0,
		Left_Stick_Y = 1,
		Right_Stick_X = 2,
		Right_Stick_Y = 3,
		Left_Trigger = 4,
		Right_Trigger = 5,

		AxisCount = Right_Trigger + 1
	};

	inline const std::string GamepadCodeToXBoxString(GamepadButtonCode code) {
		switch (code) {
		case GamepadButtonCode::XBox_A: return "A";
		case GamepadButtonCode::XBox_B: return "B";
		case GamepadButtonCode::XBox_X: return "X";
		case GamepadButtonCode::XBox_Y: return "Y";
		case GamepadButtonCode::XBox_RB: return "RB";
		case GamepadButtonCode::XBox_RT: return "RT";
		case GamepadButtonCode::XBox_LB: return "LB";
		case GamepadButtonCode::XBox_LT: return "LT";
		case GamepadButtonCode::DPad_Down: return "DPad Down";
		case GamepadButtonCode::DPad_Up: return "DPad Up";
		case GamepadButtonCode::DPad_Left: return "DPad Left";
		case GamepadButtonCode::DPad_Right: return "DPad Right";
		case GamepadButtonCode::XBox_Back: return "Back";
		case GamepadButtonCode::XBox_Start: return "Start";
		case GamepadButtonCode::XBox_Guide: return "Guide";
		case GamepadButtonCode::XBox_Thumb_Left: return "Left Thumb";
		case GamepadButtonCode::XBox_Thumb_Right: return "Right Thumb";
		default: return "<Unknown>";
		}
	}

	inline const std::string GamepadCodeToNintendoString(GamepadButtonCode code) {
		switch (code) {
		case GamepadButtonCode::Nintendo_A: return "A";
		case GamepadButtonCode::Nintendo_B: return "B";
		case GamepadButtonCode::Nintendo_X: return "X";
		case GamepadButtonCode::Nintendo_Y: return "Y";
		case GamepadButtonCode::Nintendo_R: return "R";
		case GamepadButtonCode::Nintendo_ZR: return "ZR";
		case GamepadButtonCode::Nintendo_L: return "L";
		case GamepadButtonCode::Nintendo_ZL: return "ZL";
		case GamepadButtonCode::DPad_Down: return "DPad Down";
		case GamepadButtonCode::DPad_Up: return "DPad Up";
		case GamepadButtonCode::DPad_Left: return "DPad Left";
		case GamepadButtonCode::DPad_Right: return "DPad Right";
		case GamepadButtonCode::Nintendo_Minus: return "Minus";
		case GamepadButtonCode::Nintendo_Plus: return "Plus";
		case GamepadButtonCode::Nintendo_Home: return "Home";
		case GamepadButtonCode::XBox_Thumb_Left: return "Left Thumb";
		case GamepadButtonCode::XBox_Thumb_Right: return "Right Thumb";
		default: return "<Unknown>";
		}
	}

	inline const std::string GamepadCodeToSonyString(GamepadButtonCode code) {
		switch (code) {
		case GamepadButtonCode::Sony_Cross: return "Cross";
		case GamepadButtonCode::Sony_Circle: return "Circle";
		case GamepadButtonCode::Sony_Triangle: return "Triangle";
		case GamepadButtonCode::Sony_Square: return "Square";
		case GamepadButtonCode::Sony_R1: return "R1";
		case GamepadButtonCode::Sony_R2: return "R2";
		case GamepadButtonCode::Sony_L1: return "L1";
		case GamepadButtonCode::Sony_L2: return "L2";
		case GamepadButtonCode::DPad_Down: return "DPad Down";
		case GamepadButtonCode::DPad_Up: return "DPad Up";
		case GamepadButtonCode::DPad_Left: return "DPad Left";
		case GamepadButtonCode::DPad_Right: return "DPad Right";
		case GamepadButtonCode::Sony_Select: return "Select";
		case GamepadButtonCode::Sony_Start: return "Start";
		case GamepadButtonCode::Sony_PS: return "PS";
		case GamepadButtonCode::XBox_Thumb_Left: return "Left Thumb";
		case GamepadButtonCode::XBox_Thumb_Right: return "Right Thumb";
		default: return "<Unknown>";
		}
	}

}
