#pragma once

// ######################################################################### //
// ### Event.h ############################################################# //
// ### Defines Event Types and Event Categories aswell as the SuperClass ### //
// ### "Event" of all extending Events such as Key- and Mouse-Events.    ### //
// ### Also The Event-Dispatcher is defined here.                        ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "LazyEngine/Core/Core.h"

#include "LazyEngine/Core/Input/KeyCodes.h"
#include "LazyEngine/Core/Input/MouseButtonCodes.h"

namespace LazyEngine {


	// currently events are blocking and get dispatched immediately. This might be subject to change.
	// For example they can be handled asynchronously

	enum class EventType {
		None = 0,
		// Window events
		WindowClose,			// triggers on "Alt+F4" or window close button press
		WindowResize,			// triggers if the window is being resized
		WindowFocus,			// triggers if the window gets in focus
		WindowLostFocus,		// triggers if the window loses focus
		WindowMoved,			// triggers if the window is being moved
		// ### currently not used anywhere #################################################################### //
		AppTick,				// triggers every tick aka frame
		AppUpdate,				// triggers every time the onUpdate method of Application is called
		AppRender,				// triggers every time a frame is being rendered
		// #################################################################################################### //
		// Keyboard Events
		KeyPressed,				// triggers if a key is pressed
		KeyReleased,			// triggers if a key is released
		KeyTyped,				// triggers if a key is pressed as well... TODO: find a better description
		// Mouse Events
		MouseButtonPressed,		// triggers if the mouse button is being pressed
		MouseButtonReleased,	// triggers if the mouse button is being released
		MouseMoved,				// triggers if the mouse is being moved
		MouseScrolled,			// triggers if the mouse wheel is being scrolled
		// Gamepad Events
		GamepadButtonPressed,	// triggers if a gamepad button is pressed
		GamepadButtonReleased,	// triggers if a gamepad button is released
		GamepadConnection,		// triggers if a gamepad connects or disconnects
		GamepadAxis				// triggers if a gamepad axis changes direction
	};

	enum class EventCategory {
		None = 0,
		EventCategoryApplication = BIT(0),
		EventCategoryInput = BIT(1),
		EventCategoryKeyboard = BIT(2),
		EventCategoryMouse = BIT(3),
		EventCategoryMouseButton = BIT(4),
		EventCategoryGamepad = BIT(5)
	};


	// ##################################################################### //
	// ### Event ########################################################### //
	// ##################################################################### //

	/**
	 *	Superclass of all Events in the event system
	 */
	class LAZYENGINE_API Event {
	public:
		/**
		 *	returns the type of Event in a dynamic way to enable comparisons with static event types
		 */
		virtual EventType getEventType() const = 0;
		/**
		 *	returns the name of this Event. Usually this should be the class name
		 */
		virtual const char *getName() const = 0;
		/**
		 *	returns categoryflags
		 */
		virtual int getCategoryFlags() const = 0;
		/**
		 *	returns a string representation of this.
		 *	The default implementation just returns the event name without
		 *	any additional data.
		 */
		virtual std::string toString() const { return getName(); }

		/**
		 *	returns true if this Event it a member of the given category.
		 *	@param category: the category to be tested.
		 */
		inline bool isInCategory(EventCategory category) {
			return getCategoryFlags() & static_cast<int>(category);
		}

		/**
		 *	override the stream operator to call toString()
		 */
		friend inline std::ostream& operator<<(std::ostream& os, const Event& e) {
			return os << e.toString();
		}

		/**
		 *	returns true if this event has been consumed and should
		 *	not be processed any further.
		 */
		inline bool wasConsumed() const { return m_wasConsumed; }

	private:
		// allow EventDispatcher to access private members of this class
		friend class EventDispatcher;

		// to be set to true if this event should be consumed
		bool m_wasConsumed = false;
	};



	// ##################################################################### //
	// ### EventDispatcher ################################################# //
	// ##################################################################### //

	/**
	 *	Dispatches an event to different functions while garanteeing that
	 *	only those functions will be called that are matching the type of the
	 *	event wrapped by this class.
	 */
	class EventDispatcher {
		// alias for function objects representing functions like "bool func(T& event)"
		template<typename T>
		using EventFn = std::function<bool(T&)>;
	public:
		/**
		 *	constructor.
		 *	@param event: the event to be dispatched
		 */
		EventDispatcher(Event& event)
			: m_event(event) {}

		/**
		 *	dispatches m_event to a function with return type bool 
		 *	if the template type and the event's dynamic type are the same.
		 *
		 *	If 'func' returns 'true' the event will be marked as consumed and will not propagate any further.
		 *
		 *	Example:
		 *		KeyEvent e;
		 *		EventDispatcher dispatcher(e);
		 *		dispatcher.dispatch<KeyEvent>(handleKeyEvent);				// this will call handleKeyEvent.
		 *		dispatcher.dispatch<MousePressedEvent>(handleMouseEvent);	// this will do nothing because the types are not the same.
		 */
		template<typename T>
		bool dispatch(EventFn<T> func) {
			if (m_event.getEventType() == T::getStaticType()) {
				m_event.m_wasConsumed = func(*(T*) &m_event);
				return true;
			}
			return false;
		}

		inline void consumeEvent(bool consumed) {
			m_event.m_wasConsumed |= consumed;
		}
	private:
		Event& m_event;
	};


	// alias for callbacks
	using EventCallbackFn = std::function<void(Event&)>;
}