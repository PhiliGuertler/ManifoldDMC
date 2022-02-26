#pragma once

// ######################################################################### //
// ### MouseEvent.h ######################################################## //
// ### Defines Events for Mouse movement, button clicks and button       ### //
// ### releases.                                                         ### //
// ######################################################################### //

#include "Event.h"

#include <glm/glm.hpp>

namespace LazyEngine {


	// ##################################################################### //
	// ### MouseMovedEvent ################################################# //

	/**
	 *	An event that triggers when the mouse is being moved.
	 */
	class LAZYENGINE_API MouseMovedEvent : public Event {
	public:
		/**
		 *	constructor
		 *	@param x: the new x-position in relative window coordinates.
		 *	@param y: the new y-position in relative window coordinates.
		 */
		MouseMovedEvent(float x, float y)
			: m_mouseX(x), m_mouseY(y) {}

		/**
		 *	returns the x-position relative to the top left corner of the window.
		 *	A value of 0.75f means that the mouse is 75% to the right of the left
		 *	side and 25% to the left of the right side of the window.
		 */
		inline float getX() const { return m_mouseX; }
		/**
		 *	returns the y-position relative to the top left corner of the window.
		 *	A value of 0.75f means that the mouse is 75% to the bottom of the top
		 *	side and 25% to the top of the bottom side of the window.
		 */
		inline float getY() const { return m_mouseY; }

		inline glm::vec2 getPosition() const { return glm::vec2(m_mouseX, m_mouseY); }

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseMovedEvent: x=" << getX() << ", y=" << getY();
			return ss.str();
		}

		/**
		 *	returns the type of KeyReleasedEvent in a dynamic way to enable comparisons with static event types
		 */
		static EventType getStaticType() { return EventType::MouseMoved; }
		/**
		 *	returns the type of KeyReleasedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "MouseMoved"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryMouse) | static_cast<int>(EventCategory::EventCategoryInput); }
	private:
		// members containing the new position of the mouse
		float m_mouseX, m_mouseY;
	};


	// ##################################################################### //
	// ### MouseScrolledEvent ############################################## //

	/**
	 *	An event that triggers if the scrollwheel of the mouse is being used
	 */
	class LAZYENGINE_API MouseScrolledEvent : public Event {
	public:
		/**
		 *	constructor
		 *	@param xOffset: the change in x direction
		 *	@param yOffset: the change in y direction
		 */
		MouseScrolledEvent(float xOffset, float yOffset)
			: m_xOffset(xOffset), m_yOffset(yOffset) {}

		inline float getXOffset() const { return m_xOffset; }
		inline float getYOffset() const { return m_yOffset; }

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseScrolledEvent: xOffset=" << getXOffset() << ", yOffset=" << getYOffset();
			return ss.str();
		}

		/**
		 *	returns the type of MouseScrolledEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::MouseScrolled; }
		/**
		 *	returns the type of MouseScrolledEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "MouseScrolled"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryMouse) | static_cast<int>(EventCategory::EventCategoryInput); }
	private:
		// offset in the x direction
		float m_xOffset;
		// offset in the y direction: the usual direction of a mouse wheel
		float m_yOffset;
	};


	// ##################################################################### //
	// ### MouseButtons #################################################### //
	// ##################################################################### //

	/**
	 *	superclass of mouse button events.
	 */
	class LAZYENGINE_API MouseButtonEvent : public Event {
	public:
		/**
		 *	returns the button that triggered this event
		 */
		inline MouseButtonCode getMouseButton() const { return m_button; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryMouse) | static_cast<int>(EventCategory::EventCategoryInput); }
	protected:
		/**
		 *	protected constructor
		 *	only subclasses of this should be able to be instanciated
		 */
		MouseButtonEvent(MouseButtonCode button)
			: m_button(button) {}

		// the button that triggered this event
		MouseButtonCode m_button;
	};

	
	// ##################################################################### //
	// ### MouseButtonPressedEvent ######################################### //

	/**
	 *	An event that triggers if a mouse button has been pressed.
	 */
	class LAZYENGINE_API MouseButtonPressedEvent : public MouseButtonEvent {
	public:
		/**
		 *	constructor
		 *	@param button: the button that has been pressed
		 */
		MouseButtonPressedEvent(MouseButtonCode button)
			: MouseButtonEvent(button) {}

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << static_cast<int>(getMouseButton());
			return ss.str();
		}

		/**
		 *	returns the type of MouseButtonPressedEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::MouseButtonPressed; }
		/**
		 *	returns the type of MouseButtonPressedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "MouseButtonPressed"; }
	};


	// ##################################################################### //
	// ### MouseButtonReleasedEvent ######################################## //

	/**
	 *	An event that triggers if a mouse button is released
	 */
	class LAZYENGINE_API MouseButtonReleasedEvent : public MouseButtonEvent {
	public:
		/**
		 *	constructor
		 *	@param button: the button that has been released
		 */
		MouseButtonReleasedEvent(MouseButtonCode button)
			: MouseButtonEvent(button) {}

		/**
		 *	returns a string representation of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << static_cast<int>(getMouseButton());
			return ss.str();
		}

		/**
		 *	returns the type of MouseButtonReleasedEvent in a static way to enable comparisons with dynamic event types
		 */
		static EventType getStaticType() { return EventType::MouseButtonReleased; }
		/**
		 *	returns the type of MouseButtonReleasedEvent in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "MouseButtonReleased"; }
	};
}