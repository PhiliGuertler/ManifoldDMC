#pragma once

#include "ImGuiLayer.h"

#include "LazyEngine/Profiling/Profiler.h"
#include "LazyEngine/Core/Application.h"

#include <algorithm>

namespace LazyEngine {

	// ######################################################################## //
	// ### PerformanceLayer ################################################### //
	// ######################################################################## //

	template <int N, bool showHistogram>
	PerformanceLayer<N, showHistogram>::PerformanceLayer(int textUpdateInterval, float fpsScope)
		: Layer("PerformanceLayer")
		, m_frameTimes()
		, m_renderTimes()
		, m_avgFps()
		, m_nextIndex(0)
		, m_currentIndex(0)
		, m_isNotFirstLoop(false)
		, m_fpsScope(fpsScope)
		, m_skippedFrames(0)
		, m_maxSkippedFrames(textUpdateInterval)
		, m_maxFrameTime(0.f)
		, m_maxRenderTime(0.f)
		, m_maxAvgFps(0.f)
	{
		// empty
	}

	template <int N, bool showHistogram>
	void PerformanceLayer<N, showHistogram>::onUpdate(const TimeStep& deltaTime) {
		// save the current deltaTime
		m_frameTimes[m_nextIndex] = deltaTime.getMilliseconds();
		m_maxFrameTime = std::max(m_frameTimes[m_nextIndex], m_maxFrameTime);
		/*
		// save last render time
		m_renderTimes[m_nextIndex] = Application::getInstance().getLastRenderTime().getMilliseconds();
		m_maxRenderTime = std::max(m_renderTimes[m_nextIndex], m_maxRenderTime);
		*/

		if (!m_isNotFirstLoop && m_nextIndex == 0) {
			// ignore the performance of the very first frame
			m_currentIndex = m_nextIndex;
			m_nextIndex = (m_nextIndex + 1) % N;
			return;
		}
		else if (!m_isNotFirstLoop && m_nextIndex == 1) {
			// duplicate the performance of the second frame into the first frame
			m_frameTimes[m_nextIndex - 1] = m_frameTimes[m_nextIndex];
			//m_renderTimes[m_nextIndex - 1] = m_renderTimes[m_nextIndex];
		}

		// calculate the average of the last 'fpsScope' frame times
		float avgFrameTime = 0.f;
		for (int i = 0; i < static_cast<int>(std::ceil(N*m_fpsScope)); ++i) {
			int index = ((m_nextIndex - i) < 0) ? (N - m_nextIndex - i) : (m_nextIndex - i);
			avgFrameTime += m_frameTimes[index];
		}
		if (!m_isNotFirstLoop) {
			// the amount of samples is different in the first iteration than in the others
			avgFrameTime /= std::max(static_cast<float>(std::min(m_nextIndex, static_cast<int>(std::ceil(N*m_fpsScope)))), 1.f);
		}
		else {
			avgFrameTime /= std::ceil(N*m_fpsScope);
		}
		

		// calculate the average fps over the last 5% of frame times
		m_avgFps[m_nextIndex] = 1000.f / avgFrameTime;
		m_maxAvgFps = std::max(m_avgFps[m_nextIndex], m_maxAvgFps);

		// update index counters
		m_currentIndex = m_nextIndex;
		m_nextIndex = (m_nextIndex + 1) % N;
		m_isNotFirstLoop |= (m_nextIndex == 0);
	}

	template <int N, bool showHistogram>
	void PerformanceLayer<N, showHistogram>::onImGuiRender() {
		static bool opened = true;

		ImGui::Begin("Performance Tracker", &opened);
		size_t skippedIndex = (m_currentIndex < m_skippedFrames) ? (N - 1 - m_skippedFrames + m_currentIndex) : (m_currentIndex - m_skippedFrames);
		m_skippedFrames = (m_skippedFrames + 1) % m_maxSkippedFrames;

		// Draw a plot of the frame times
		ImGui::Text("Frame Time: %.2f ms", m_frameTimes[skippedIndex]);
		// only compile the relevant part
		if constexpr(showHistogram) {
			ImGui::PlotHistogram("Frame Times", m_frameTimes.data(), static_cast<int>(m_frameTimes.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxFrameTime, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}
		else {
			ImGui::PlotLines("Frame Times", m_frameTimes.data(), static_cast<int>(m_frameTimes.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxFrameTime, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}

		// Draw a plot of the render times
		ImGui::Text("Render Time: %.2f ms", m_renderTimes[skippedIndex]);
		// only compile the relevant part
		if constexpr(showHistogram) {
			ImGui::PlotHistogram("Render Times", m_renderTimes.data(), static_cast<int>(m_renderTimes.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxRenderTime, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}
		else {
			ImGui::PlotLines("Render Times", m_renderTimes.data(), static_cast<int>(m_renderTimes.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxRenderTime, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}

		// Draw a plot of the avg fps
		ImGui::Text("Average FPS: %.2f", m_avgFps[skippedIndex]);
		// only compile the relevant part
		if constexpr(showHistogram) {
			ImGui::PlotHistogram("Average FPS", m_avgFps.data(), static_cast<int>(m_avgFps.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxAvgFps, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}
		else {
			ImGui::PlotLines("Average FPS", m_avgFps.data(), static_cast<int>(m_avgFps.size()), static_cast<int>(m_nextIndex), NULL, 0.f, m_maxAvgFps, ImVec2(static_cast<float>(std::max(N, 250)), 80));
		}

		/*
		auto& window = Application::getInstance().getWindow();
		bool isVsync = window.isVSync();
		if (ImGui::Checkbox("Enable VSync", &isVsync)) {
			window.setVSync(isVsync);
		}
		*/

		// enable profiling for the next n frames
		ImGui::Separator();
		ImGui::Text("Options to produce a profiling file output that can be used with chrome://tracing");
		static int numFrames = 10;
		ImGui::SliderInt("Frame Count", &numFrames, 1, 300);
		Profiler& profiler = Profiler::getInstance();
		if (!profiler.sessionIsRunning()) {
			if (ImGui::Button("Begin Session")) {
				profiler.profileNextNFrames(numFrames);
			}
		}


		ImGui::End();
	}
}