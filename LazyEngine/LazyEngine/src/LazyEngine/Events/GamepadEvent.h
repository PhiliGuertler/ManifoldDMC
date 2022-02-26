#pragma once

#include "Event.h"

#include "../Core/Input/GamepadCodes.h"

#include <glm/glm.hpp>

namespace LazyEngine {

		// ##################################################################### //
	// ### GamepadEvent #################################################### //

	class LAZYENGINE_API GamepadEvent : public Event {
	public:
		/**
		 *	returns the gamepad ID of the gamepad that caused this event
		 */
		inline GamepadID getGamepadID() const { return m_gamepadID; }

		/**
		 *	Category flags for any gamepad event
		 */
		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryGamepad) | static_cast<int>(EventCategory::EventCategoryInput); }

	protected:
		/**
		 *	protected constructor that can only be called by subclasses.
		 */
		GamepadEvent(int gamepadID)
			: m_gamepadID(gamepadID) {}

		// the gamepad ID. This is important, as multiple controllers can be connected at once.
		GamepadID m_gamepadID;
	};


	// ##################################################################### //
	// ### GamepadConnectionEvent ########################################## //

	/**
	 *	An event that triggers if a gamepad connects or disconnects.
	 */
	class LAZYENGINE_API GamepadConnectionEvent : public GamepadEvent {
	public:
		GamepadConnectionEvent(GamepadID gamepadID, bool hasConnected)
			: GamepadEvent(gamepadID), m_hasConnected(hasConnected) {}

		/**
		 *	returns whether this gamepad has connected or disconnected
		 */
		inline bool hasConnected() const { return m_hasConnected; }

		inline std::string toString() const override {
			std::stringstream ss;
			ss << "GamepadConnectionEvent: gamepadID=" << getGamepadID() << ", hasConnected=" << hasConnected();
			return ss.str();
		}

		static EventType getStaticType() { return EventType::GamepadConnection; }
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "GamepadConnection"; }

	private:
		bool m_hasConnected;
	};


	// ##################################################################### //
	// ### GamepadButtonPressedEvent ####################################### //

	/**
	 *	An event that triggers if a gamepad button is pressed down
	 */
	class LAZYENGINE_API GamepadButtonPressedEvent : public GamepadEvent {
	public:

		GamepadButtonPressedEvent(GamepadID gamepadID, GamepadButtonCode button)
			: GamepadEvent(gamepadID), m_button(button) {}

		inline GamepadButtonCode getButton() const { return m_button; }

		inline std::string toString() const override {
			std::stringstream ss;
			ss << "GamepadButtonPressedEvent: gamepadID=" << getGamepadID() << ", button=" << GamepadCodeToXBoxString(getButton());
			return ss.str();
		}

		static EventType getStaticType() { return EventType::GamepadButtonPressed; }
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "GamepadButtonPressed"; }

	private:
		GamepadButtonCode m_button;
	};


	// ##################################################################### //
	// ### GamepadButtonReleasedEvent ###################################### //

	/**
	 *	An event that triggers if a gamepad button is released
	 */
	class LAZYENGINE_API GamepadButtonReleasedEvent : public GamepadEvent {
	public:

		GamepadButtonReleasedEvent(GamepadID gamepadID, GamepadButtonCode button)
			: GamepadEvent(gamepadID), m_button(button) {}

		inline GamepadButtonCode getButton() const { return m_button; }

		inline std::string toString() const override {
			std::stringstream ss;
			ss << "GamepadButtonReleasedEvent: gamepadID=" << getGamepadID() << ", button=" << GamepadCodeToXBoxString(getButton());
			return ss.str();
		}

		static EventType getStaticType() { return EventType::GamepadButtonReleased; }
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "GamepadButtonReleased"; }

	private:
		GamepadButtonCode m_button;
	};

	   
	// ##################################################################### //
	// ### GamepadAxisEvent ################################################ //

	/**
	 *	An event that triggers if a gamepad axis is moved to a different direction
	 */
	class LAZYENGINE_API GamepadAxisEvent : public GamepadEvent {
	public:
		GamepadAxisEvent(GamepadID gamepadID, GamepadAxisCode axis, float value)
			: GamepadEvent(gamepadID), m_axis(axis), m_value(value) {}

		inline GamepadAxisCode getAxis() const { return m_axis; }

		inline float getValue() const { return m_value; }

		inline std::string toString() const override {
			std::stringstream ss;
			ss << "GamepadAxisEvent: gamepadID=" << getGamepadID() << ", axis=" << static_cast<int>(getAxis()) << ", value=" << getValue();
			return ss.str();
		}

		static EventType getStaticType() { return EventType::GamepadAxis; }
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "GamepadStick"; }

	private:
		GamepadAxisCode m_axis;
		float m_value;
	};

}