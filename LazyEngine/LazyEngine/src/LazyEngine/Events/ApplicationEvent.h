#pragma once

// ######################################################################### //
// ### ApplicationEvent.h ################################################## //
// ### Defines WindowEvents such as WindowResize- and WindowCloseEvents, ### //
// ### aswell as AppEvents like AppTick-, AppUpdate- and AppRenderEvents ### //
// ######################################################################### //

#include "Event.h"

namespace LazyEngine {


	// ##################################################################### //
	// ### WindowEvents #################################################### //
	// ##################################################################### //

	// ##################################################################### //
	// ### WindowResizeEvent ############################################### //

	/**
	 *	An Event that triggers if the window is being resized.
	 */
	class LAZYENGINE_API WindowResizeEvent : public Event {
	public:
		/**
		 *	Constructor
		 *	@param width: the new width of the window.
		 *	@param height: the new height of the window.
		 */
		WindowResizeEvent(unsigned int width, unsigned int height)
			: m_width(width), m_height(height) {}

		/**
		 *	Constructor
		 *	@param size: the new size of the window in width and height.
		 */
		WindowResizeEvent(glm::ivec2 size)
			: m_width(size.x), m_height(size.y) {}

		/**
		 *	returns the new width of the resized window
		 */
		inline unsigned int getWidth() const { return m_width; }
		/**
		 *	returns the new height of the resized window
		 */
		inline unsigned int getHeight() const { return m_height; }

		/**
		 *	prints the contents of this event
		 */
		std::string toString() const override {
			std::stringstream ss;
			ss << "WindowResizeEvent: width=" << getWidth() << ", height=" << getHeight();
			return ss.str();
		}

		/**
		 *	returns the type of WindowResizeEvent in a static way to enable comparisons with the dynamic event type
		 */
		static EventType getStaticType() { return EventType::WindowResize; }
		/**
		 *	returns the type of WindowResizeEvent in a dynamic way to enable comparisons with the static event type
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "WindowResize"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryApplication); }
	private:
		unsigned int m_width;
		unsigned int m_height;
	};


	// ##################################################################### //
	// ### WindowCloseEvent ################################################ //

	/**
	 *	An event that triggers if the window is being closed
	 *	(by pressing the little X on the top right)
	 */
	class LAZYENGINE_API WindowCloseEvent : public Event {
	public:
		WindowCloseEvent() {}

		/**
		 *	returns the type of WindowCloseEvent in a static way to enable comparisons with the dynamic event type
		 */
		static EventType getStaticType() { return EventType::WindowClose; }
		/**
		 *	returns the type of WindowCloseEvent in a dynamic way to enable comparisons with the static event type
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "WindowClose"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryApplication); }
	};


	// ##################################################################### //
	// ### AppEvents ####################################################### //
	// ##################################################################### //
	// ### FIXME: these events are currently not used anywhere.          ### //
	// ##################################################################### //


	// ##################################################################### //
	// ### AppTickEvent #################################################### //

	class LAZYENGINE_API AppTickEvent : public Event {
	public:
		AppTickEvent() {}

		/**
		 *	returns the type of AppTickEvent in a static way to enable comparisons with the dynamic event type
		 */
		static EventType getStaticType() { return EventType::AppTick; }
		/**
		 *	returns the type of AppTickEvent in a dynamic way to enable comparisons with the static event type
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "AppTick"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryApplication); }
	};


	// ##################################################################### //
	// ### AppUpdateEvent ################################################## //

	class LAZYENGINE_API AppUpdateEvent : public Event {
	public:
		AppUpdateEvent() {}

		/**
		 *	returns the type of AppUpdateEvent in a static way to enable comparisons with the dynamic event type
		 */
		static EventType getStaticType() { return EventType::AppUpdate; }
		/**
		 *	returns the type of AppUpdateEvent in a dynamic way to enable comparisons with the static event type
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "AppUpdate"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryApplication); }
	};


	// ##################################################################### //
	// ### AppRenderEvent ################################################## //

	class LAZYENGINE_API AppRenderEvent : public Event {
	public:
		AppRenderEvent() {}

		/**
		 *	returns the type of AppRenderEvent in a static way to enable comparisons with the dynamic event type
		 */
		static EventType getStaticType() { return EventType::AppRender; }
		/**
		 *	returns the type of AppRenderEvent in a dynamic way to enable comparisons with the static event type
		 */
		virtual EventType getEventType() const override { return getStaticType(); }
		virtual const char *getName() const override { return "AppRender"; }

		virtual int getCategoryFlags() const override { return static_cast<int>(EventCategory::EventCategoryApplication); }
	};
}