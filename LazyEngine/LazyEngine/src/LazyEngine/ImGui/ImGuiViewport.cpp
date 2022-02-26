#include "LazyEngine/gepch.h"

#include "ImGuiViewport.h"

#include "imgui.h"

namespace LazyEngine {

	ImGuiViewport::ImGuiViewport(const std::string& name)
		: m_name(name)
		, m_framebuffer(nullptr)
		, m_viewportSize(1,1)
		, m_isHovered(false)
		, m_isFocused(false)
		, m_sizeChanged(false)
	{
		// create the framebuffer for this viewport
		m_framebuffer = Framebuffer::create(1, 1);
	}

	void ImGuiViewport::begin() {
		// set the style to be borderless
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		// begin the viewport in a new imgui window
		ImGui::Begin(m_name.c_str());

		ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;

		// check if this viewport is currently being hovered
		m_isHovered = ImGui::IsWindowHovered();
		m_isFocused = ImGui::IsWindowFocused();

		// get the current size of the viewport
		ImVec2 viewportSize = ImGui::GetContentRegionAvail();
		glm::ivec2 currentSize = {viewportSize.x, viewportSize.y};

		m_sizeChanged = false;
		// compare the current size with the stored size
		if (currentSize.x != m_viewportSize.x || currentSize.y != m_viewportSize.y) {
			// the viewport size changed, resize the framebuffer and trigger a resize-event
			m_framebuffer->resize(currentSize);
			m_viewportSize = currentSize;
			m_sizeChanged = true;
		}

		m_framebuffer->bind();

		// Rendering goes here (after this function)
	}

	void ImGuiViewport::end() {
		// Rendering goes before this function

		m_framebuffer->unbind();

		// Get the position of this viewport relative to its parent window
		auto viewport = ImGui::GetWindowViewport();
		ImVec2 viewportWorkPos = viewport->GetWorkPos();
		ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

		m_offset = glm::ivec2(cursorScreenPos.x, cursorScreenPos.y) - glm::ivec2(viewportWorkPos.x, viewportWorkPos.y);

		ImGui::Image((void *)(m_framebuffer->getColorTexture()->getRendererID()), ImVec2((float)m_viewportSize.x, (float)m_viewportSize.y), ImVec2{ 0,1 }, ImVec2{ 1,0 });
		
		ImGui::End();
		ImGui::PopStyleVar();
	}

	void ImGuiViewport::renderInfos() const {
		ImGui::Text("Offset: [%d, %d]", m_offset.x, m_offset.y);
	}

}