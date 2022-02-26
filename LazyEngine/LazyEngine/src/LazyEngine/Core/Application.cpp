// ######################################################################### //
// ### Application.cpp ##################################################### //
// ### Implements Application.h                                          ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "Application.h"

#include "../Renderer/Renderer.h"
#include "../Renderer/RendererImpl.h"
#include "../Renderer/RenderCommand.h"
#include "../Profiling/Profiler.h"
#include "Input/Input.h"

namespace LazyEngine {

	/**
	 *	Application is a Singleton Class
	 */
	Application* Application::s_instance = nullptr;

	/**
	 *	Application Constructor
	 */
	Application::Application()
		: m_window(nullptr)
		, m_imguiLayer(nullptr)
		, m_isRunning(true)
		, m_layerStack()
		, m_lastFrameTime()
		, m_lastRenderTime()
		, m_frameCount(0)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		// check if an application is already existing.
		LAZYENGINE_CORE_ASSERT(s_instance == nullptr, "Application already exists");
		s_instance = this;

		// create the window
		m_window = createScope<Window>(Window::create());
		m_window->setEventCallback(LAZYENGINE_BIND_EVENT_FUNC(Application::onEvent));
		m_window->setVSync(true);

		// set the callback for inputs
		Input::setGamepadEventCallback(LAZYENGINE_BIND_EVENT_FUNC(Application::onEvent));

		// initialize Renderer
		RenderCommand::init();

		// create an imgui-layer for in-engine debugging
		m_imguiLayer = new ImGuiLayer();
		pushOverlay(m_imguiLayer);
	}


	Application::~Application()
	{
		LAZYENGINE_PROFILE_FUNCTION();
		// empty. The LayerStack will delete its content automatically.
	}


	void Application::run() {
		// set the first frame time to be the time of startup.
		m_lastFrameTime = TimePoint();

		TimePoint currentTime;
		TimeStep deltaTime;

#ifdef LAZYENGINE_ENABLE_PROFILING
		Profiler& profiler = Profiler::getInstance();
#endif

		while (m_isRunning) {
#ifdef LAZYENGINE_ENABLE_PROFILING
			// check if profiling should be turned on
			if (profiler.getNProfileEndFrame() != -1 && !profiler.sessionIsRunning()) {
				// Turn the profiler on
				Profiler::getInstance().beginNSession();
			}
#endif
			try {
				LAZYENGINE_PROFILE_SCOPE("RunLoop");
				// measure the current time and compute the time passed since the last frame started
				currentTime = TimePoint();
				deltaTime = TimeStep(m_lastFrameTime, currentTime);
				m_lastFrameTime = currentTime;

				TimePoint beginRendering;

				// update window
				m_window->onUpdate();

				if (m_useFixedTimeSteps) {
					deltaTime.setTimeSpan(1.f / 60.f);
				}

				// only update Layerstack if the window is not minimized to save cpu time
				if (!m_window->isMinimized()) {
					for (Layer* layer : m_layerStack) {
						layer->onUpdate(deltaTime);
						// ### EXPERIMENTAL ### //
#ifdef LAZYENGINE_ENABLE_IMGUI
						if (!m_imguiLayer->isDockingEnabled()) {
#endif
							layer->onRender();
#ifdef LAZYENGINE_ENABLE_IMGUI
						}
#endif
						// ### /EXPERIMENTAL ### //
					}
				}

				// imgui can be undocked and will still be responsive if the window is minimized
				// FIXME: a memory leak is occuring somewhere around here in imgui
#ifdef LAZYENGINE_ENABLE_IMGUI
				m_imguiLayer->begin();
				for (Layer* layer : m_layerStack) {
					layer->onImGuiRender();
				}
				// ### EXPERIMENTAL ### //
				if (m_imguiLayer->isDockingEnabled()) {
					if (m_imguiLayer->beginViewport(c_mainViewportName)) {
						// the viewport size changed, trigger a window resize event
						WindowResizeEvent event(m_imguiLayer->getViewportSize(c_mainViewportName));
						propagateEvent(event);
					}
					for (Layer* layer : m_layerStack) {
						layer->onRender();
					}
					m_imguiLayer->endViewport(c_mainViewportName);
				}
				// ### /EXPERIMENTAL ### //

				m_imguiLayer->end();
#endif

				TimePoint endRendering;
				m_lastRenderTime = TimeStep(beginRendering, endRendering);


				// adjust frame count
				m_frameCount++;

#ifdef LAZYENGINE_ENABLE_PROFILING
				// check if a running profiler should be stopped
				if (profiler.sessionIsRunning() && m_frameCount >= profiler.getNProfileEndFrame()) {
					Profiler::getInstance().endSession();
				}
#endif
			}
			catch (std::exception& e) {
				LAZYENGINE_CORE_ERROR("Uncaught Exception: {0}", e.what());
			}
		}
	}

	void Application::propagateEvent(Event& e) {
		// dispatch events to the layers of the layer stack, starting at the top and moving to the bottom.
		for (auto iter = m_layerStack.end(); iter != m_layerStack.begin(); /* do nothing here */) {
			--iter;
			(*iter)->onEvent(e);
			// if the event has been consumed by the current layer, stop the propagation.
			if (e.wasConsumed()) break;
		}
	}


	void Application::onEvent(Event& e) {
		LAZYENGINE_PROFILE_FUNCTION();

		EventDispatcher dispatcher(e);

		// handle WindowClose- and WindowResizeEvents in this class.
		dispatcher.dispatch<WindowCloseEvent>(LAZYENGINE_BIND_EVENT_FUNC(Application::onWindowClose));
		dispatcher.dispatch<WindowResizeEvent>(LAZYENGINE_BIND_EVENT_FUNC(Application::onWindowResize));

		propagateEvent(e);
	}

	bool Application::onWindowClose(WindowCloseEvent& e) {
		LAZYENGINE_PROFILE_FUNCTION();

		m_isRunning = false;
		// consume this event.
		return true;
	}

	bool Application::onWindowResize(WindowResizeEvent& e) {
		LAZYENGINE_PROFILE_FUNCTION();

		if (m_window->isMinimized() || e.getWidth() == 0 || e.getHeight() == 0) {
			return false;
		}

		RendererImpl::getInstance().onWindowResize(e.getWidth(), e.getHeight());

		// don't consume this event.
		return false;
	}
}