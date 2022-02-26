#pragma once

// ######################################################################### //
// ### Layer.h ############################################################# //
// ### Defines a Layer Class that should be extended later on.           ### //
// ### Layers will be used for rendering and event handling hirachies.   ### //
// ######################################################################### //

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Events/Event.h"

#include "LazyEngine/Core/Time.h"

namespace LazyEngine {

	/**
	 *	Defines a Rendering-Layer that will be called on events or updates by the Application.
	 *	This reference implementation is doing nothing to allow for stack allocations.
	 *	The Client should build his own layers by extending this class.
	 */
	class LAZYENGINE_API Layer {
	public:
		Layer(const std::string& name = "Layer") : m_debugName(name) { /* do nothing by default */ }
		virtual ~Layer() { /* do nothing by default */ }

		/**
		 *	Will be called when this object is being pushed onto a LayerStack.
		 *	Should be extended to handle initializing code.
		 */
		virtual void onAttach() { /* do nothing by default */ }
		/**
		 *	Will be called when this object is being removed from a LayerStack.
		 *	Should be extended to handle cleanup code.
		 */
		virtual void onDetach() { /* do nothing by default */ }
		/**
		 *	Will be called once every frame.
		 *	This should update elements.
		 *	@param deltaTime: the time that has passed since the last frame
		 */
		virtual void onUpdate(const TimeStep& deltaTime) { /* do nothing by default */ }
		/**
		 *	## Experimental: ImGui Dockspace ##
		 *	Will be called once every frame.
		 *	This should contain all the render calls of this layer.
		 */
		virtual void onRender() { /* do nothing by default */ }
		/**
		 *	Will be called once every frame.
		 *	Anything that should be displayed in the main imgui layer goes here.
		 */
		virtual void onImGuiRender() { /* do nothing by default */ }
		/**
		 *	Will be called if an event occured that has not been consumed by any previous layer.
		 *	Event handling should be executed here.
		 *	@param event: the event that occured.
		 */
		virtual void onEvent(Event& event) { /* do nothing by default */ }

		// FIXME: this is not used anywhere
		inline const std::string& getName() const { return m_debugName; }

	protected:
		// FIXME: this name is not used anywhere except the getter
		std::string m_debugName;
	};
}