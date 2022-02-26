#pragma once

// ######################################################################### //
// ### ImGuiLayer.h ######################################################## //
// ### An Adapter of ImGui to match this engine's Layer System.          ### //
// ######################################################################### //

#include "LazyEngine/Core/Layer.h"
#include "LazyEngine/Events/MouseEvent.h"
#include "LazyEngine/Events/KeyEvent.h"
#include "LazyEngine/Events/ApplicationEvent.h"
#include "LazyEngine/Renderer/Framebuffer.h"
#include "ImGuiViewport.h"

#include "imgui.h"

#include <map>

namespace LazyEngine {

	// ######################################################################## //
	// ### ImGuiLayer ######################################################### //
	// ######################################################################## //

	/**
	 *	An adapter for ImGui::Begin and ImGui::End to match the Layer API.
	 */
	class LAZYENGINE_API ImGuiLayer : public Layer {
	public:
		/**
		 *	constructor
		 */
		ImGuiLayer();
		/**
		 *	destructor
		 */
		virtual ~ImGuiLayer();

		/**
		 *	initializes ImGui
		 */
		virtual void onAttach() override;
		/**
		 *	shuts down ImGui
		 */
		virtual void onDetach() override;

		/**
		 *	Will be called if an event occured that has not been consumed by any previous layer.
		 *	Event handling should be executed here.
		 *	@param event: the event that occured.
		 */
		virtual void onEvent(Event& event) override;

		/**
		 *	begins an ImGui Frame.
		 */
		void begin();
		/**
		 *	ends an ImGui Frame and renders it.
		 */
		void end();

		/**
		 *	Renders the Menu at the top of the screen
		 *	By default, the Menu "File" only contains the option "Exit"
		 */
		virtual void renderMenu();

		// ### EXPERIMENTAL ### //

		/**
		 *	Starts a viewport with a given title name
		 *	@param name: The name of the Viewport
		 *	@returns true if the viewport size changed, false otherwise
		 */
		bool beginViewport(const std::string& name);
		void endViewport(const std::string& name);

		glm::ivec2 getViewportSize(const std::string& name) const;
		bool isHovered(const std::string& name) const;

		inline const ImGuiViewport& getViewport(const std::string& name) const {
			return *m_viewports.at(name);
		}

		/**
		 *	enables or disables docking mode
		 */
		inline void enableDocking(bool enable) { m_dockingEnabled = enable; }

		inline bool isDockingEnabled() const { return m_dockingEnabled; }

	protected:
		void beginDockmode();

		void endDockmode();
	protected:
		// defaults to false
		bool m_dockingEnabled;

		std::map<std::string, Ref<ImGuiViewport>> m_viewports;
		// ### /EXPERIMENTAL ### //
	};


	// ######################################################################## //
	// ### PerformanceLayer ################################################### //
	// ######################################################################## //

	/**
	 *	An example Layer displaying the frame times and avg fps.
	 *	@param N: the amount of individual frame data to be stored.
	 *	@param showHistogram: if true the data will be displayed in histograms,
	 *		otherwise the data will be displayed as lines
	 */
	template <int N = 128, bool showHistogram = true>
	class LAZYENGINE_API PerformanceLayer : public Layer {
	public:
		/**
		 *	constructor
		 *	@param textUpdateInterval: speciefies the amount of frames that the text
		 *		will not change its numbers.
		 *	@param fpsScore: determines the range of influence that is used to calculate
		 *		the average framerate. A value of 0.05f for example means, that the last
		 *		5% of frame times will be averaged to compute the average FPS.
		 */
		PerformanceLayer(int textUpdateInterval = 6, float fpsScope = 0.05f);
		/**
		 *	default destructor
		 */
		virtual ~PerformanceLayer() = default;

		/**
		 *	updates the frame data.
		 *	@param deltaTime: the time that has passed since the last frame
		 */
		virtual void onUpdate(const TimeStep& deltaTime) override;
		/**
		 *	displays the frame data in an ImGui window
		 */
		virtual void onImGuiRender() override;
	private:
		// stores the last N frame times (with vsync this is longer than the render time)
		std::array<float, N> m_frameTimes;
		// stores the last N render times
		std::array<float, N> m_renderTimes;
		// stores the last N average FPS
		std::array<float, N> m_avgFps;

		// index of the next timestep to be stored
		int m_nextIndex;
		// index of the most recent entry
		int m_currentIndex;

		// true for the first N frames, false after that.
		bool m_isNotFirstLoop;
		// the range influence of frame times for the avgFPS computation
		float m_fpsScope;

		// counter to keep track of the number of frames that have not updated the text until now.
		int m_skippedFrames;
		// only update the text every m_maxSkippedFrames frame
		int m_maxSkippedFrames;

		float m_maxFrameTime, m_maxRenderTime, m_maxAvgFps;
	};
}

#include "ImGuiLayer.inl"