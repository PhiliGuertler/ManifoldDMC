#pragma once

// ######################################################################### //
// ### Application.h ####################################################### //
// ### Defines the actual Application that should be extended in a       ### //
// ### Project using this engine.                                        ### //
// ######################################################################### //

#include "Core.h"

#include "Input/KeyCodes.h"
#include "LayerStack.h"
#include "Time.h"
#include "Window.h"

#include "LazyEngine/Events/Event.h"
#include "LazyEngine/Events/ApplicationEvent.h"

#include "LazyEngine/ImGui/ImGuiLayer.h"

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	/**
	 *	This class represents an Application containing a graphics-window and a LayerStack.
	 *	It also defines the Renderloop and dispatches Events to the individual Layers.
	 */
	class LAZYENGINE_API Application
	{
	protected:
		/**
		 *	Constructor
		 */
		Application();
		
	public:
		/**
		 *	Destructor
		 */
		virtual ~Application();

		/** 
		 *	the actual renderloop of the application
		 *	that is being called once every frame
		 */
		void run();

		/**
		 *	the event handler for keyevents, mouseevents, etc.
		 *	@param e: the event to be handled
		 */
		void onEvent(Event& e);

		/**
		 *	layer management
		 *	@param layer: raw pointer to the layer to be added.
		 *		ownership of the layer will be transferred to application.
		 */
		inline void pushLayer(Layer *layer) { m_layerStack.pushLayer(layer); }

		/**
		 *	overlay management: overlays will always be rendered on top of regular layers.
		 *	@param layer: raw pointer to the layer to be added.
		 *		ownership of the layer will be transferred to application.
		 */
		inline void pushOverlay(Layer *overlay) { m_layerStack.pushOverlay(overlay); }

		/**
		 *	pops layer from the application's layer stack.
		 *		ownership of the layer will be transferred to the caller.
		 *		The application will not handle its deletion anymore.
		 */
		inline void popLayer(Layer *layer) { m_layerStack.popLayer(layer); }

		/**
		 *	pops overlay from the application's layer stack.
		 *		ownership of the layer will be transferred to the caller.
		 *		The application will not handle its deletion anymore.
		 */
		inline void popOverlay(Layer *overlay) { m_layerStack.popOverlay(overlay); }

		/**
		 *	returns the window this application is representing.
		 */
		inline Window& getWindow() { return *m_window; }

		/**
		 *	Returns the size of a specific named (ImGui)-Viewport
		 *  @param name: The name of the viewport
		 */
		inline glm::ivec2 getViewportSize(const std::string& name) {
			if (m_imguiLayer != nullptr) {
				return m_imguiLayer->getViewportSize(name);
			}
			return glm::ivec2(-1);
		}

		inline glm::ivec2 getMainViewportSize() {
			return getViewportSize(c_mainViewportName);
		}

		inline const ImGuiViewport& getMainViewport() const {
			return m_imguiLayer->getViewport(c_mainViewportName);
		}

		/**
		 *	Returns how long the last frame took to be rendered.
		 *	Can be used to measure performance.
		 */
		inline TimeStep getLastRenderTime() { return m_lastRenderTime; }

		/**
		 *	Returns the amount of frames that have been processed completely until now.
		 */
		inline long long getFrameCount() { return m_frameCount; }

		/**
		 *	Closes the Application and the Window
		 */
		inline void close() { m_isRunning = false; }

		/**
		 *	For Debugging purposes, every onUpdate will receive a deltaTime of 1/60th of a second
		 *	if set to true, otherwise the real time difference will be used.
		 */
		inline void setFixedTimesteps(bool value) { m_useFixedTimeSteps = value; }

	public:
		/**
		 *	Getter of the singleton instance of this class
		 */
		inline static Application& getInstance() { return *s_instance; }

	private:
		void propagateEvent(Event& e);

		/**
		 *	Takes care of window close events, for example to just minimize instead of close,
		 *	or to handle resource freeing
		 *	@param e: the event that triggered on window close.
		 */
		bool onWindowClose(WindowCloseEvent& e);

		/**
		 *	Takes care of window resize events by updating framebuffer sizes
		 *	@param e: the event that triggered on window resize
		 */
		bool onWindowResize(WindowResizeEvent& e);

	protected:
		// This layer will be treated seperately from all the other layers.
		ImGuiLayer *m_imguiLayer;
	private:
		// The graphical window of the application
		Scope<Window> m_window;
		// Flag that indicates if the application is running.
		bool m_isRunning;
		// The layerStack will be rendered from bottom to top.
		LayerStack m_layerStack;
		// The TimePoint of the last frame that will be used to compute the frame times
		TimePoint m_lastFrameTime;
		// The amount of time that passed during the last render loop run
		TimeStep m_lastRenderTime;
		// The number of frames that have already been processed
		long long m_frameCount;
		// For Debugging Purposes, every Frame will be handled as if 1/60th of a second has passed
		bool m_useFixedTimeSteps;
		// The name of the main viewport that handles OpenGL
		const std::string c_mainViewportName = "Viewport";

	private:
		// This class is a singleton
		static Application *s_instance;
	};

	// to be defined in a client
	Application *createApplication();

}