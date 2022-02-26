#pragma once

// ######################################################################### //
// ### KeyEvent.h ########################################################## //
// ### Defines KeyEvents such as KeyPressedEvents and KeyReleasedEvents  ### //
// ######################################################################### //

#include "Event.h"

namespace LazyEngine {


	// ##################################################################### //
	// ### KeyEvents ####################################################### //
	// ##################################################################### //

	/**
	 *	superclass of all keyboard related events
	 */
	class LAZYENGINE_API KeyEvent : public Event {
	public:
		/**
		 *	returns the KeyCode of the key that has triggered the event.
		 */
		inline KeyCode getKeyCode() const { return m_keyCode; }
		/**
		 *	returns category flags
		 */
		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryKeyboard) | static_cast<int>(EventCategory::EventCategoryInput); }
	protected:
		/**
		 *	protected constructor to prevent instanciation of this class.
		 *	only sub classes should be instanciable
		 */
		KeyEvent(KeyCode keycode)
			: m_keyCode(keycode) {}

		// member storing the keycode of the key that triggered this event.
		KeyCode m_keyCode;
	};


	// ##################################################################### //
	// ### KeyPressedEvent ################################################# //

	/**
	 *	An event that triggers if a key has been pressed or if that key is
	 *	being pressed down for a longer period of time and triggers repeated
	 *	action.
	 */
	class LAZYENGINE_API KeyPressedEvent : public KeyEvent {
	public:
		/**
		 *	constructor
		 *	@param keycode: the key that triggered this event
		 *	@param repeatCount: the number of times that this key has triggered
		 *		a repeat event
		 */
		KeyPressedEvent(KeyCode keycode, int repeatCount)
			: KeyEvent(keycode), m_repeatCount(repeatCount) {}

		/**
		 *	returns the amount of repeats this key has triggered up until now.
		 */
		inline int getRepeatCount() const { return m_repeatCount; }

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyPressedEvent: " << static_cast<int>(getKeyCode()) << " (" << getRepeatCount() << " repeats)";
			return ss.str();
		}

		/**
		 *	returns the type of KeyPressedEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::KeyPressed; }
		/**
		 *	returns the type of KeyPressedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "KeyPressed"; }
	private:
		int m_repeatCount;
	};


	// ##################################################################### //
	// ### KeyReleasedEvent ################################################ //

	/**
	 *	An event that triggers if a key has been released.
	 */
	class LAZYENGINE_API KeyReleasedEvent : public KeyEvent {
	public:
		/**
		 *	constructor
		 *	@param keycode: the keycode of the key that triggered this event
		 */
		KeyReleasedEvent(KeyCode keycode)
			: KeyEvent(keycode) {}

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << static_cast<int>(getKeyCode());
			return ss.str();
		}

		/**
		 *	returns the type of KeyReleasedEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::KeyReleased; }
		/**
		 *	returns the type of KeyReleasedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "KeyReleased"; }
	};

	// ##################################################################### //
	// ### KeyTypedEvent ################################################### //

	/**
	 *	An event that is triggered if a char is being input.
	 *	Unlike KeyPressedEvent the keyCode represents a UTF-32 encoded char
	 *	while obeying the keyboard layout.
	 */
	class LAZYENGINE_API KeyTypedEvent : public KeyEvent {
	public:
		/**
		 *	constructor
		 *	@param keycode: the UTF-32 representation of the typed character
		 */
		KeyTypedEvent(KeyCode keycode)
			: KeyEvent(keycode) {}

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyTypedEvent: " << static_cast<int>(getKeyCode());
			return ss.str();
		}

		/**
		 *	returns the type of KeyTypedEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::KeyTyped; }
		/**
		 *	returns the type of KeyTypedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "KeyTyped"; }
	};
}