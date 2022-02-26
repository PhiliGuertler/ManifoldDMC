#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Renderer/Framebuffer.h"

#include "imgui.h"

namespace LazyEngine {

	class ImGuiViewport {
	public:
		ImGuiViewport(const std::string& name);

		void begin();
		void end();

		inline bool isHovered() const { return m_isHovered; }
		inline bool isFocused() const { return m_isFocused; }

		inline bool hasResized() const { return m_sizeChanged; }

		inline glm::ivec2 getSize() const { return m_viewportSize; }
		inline glm::ivec2 getOffset() const { return m_offset; }

		inline const std::string getName() const { return m_name; }

		void renderInfos() const;

	protected:
		std::string m_name;
		Ref<Framebuffer> m_framebuffer;
		glm::ivec2 m_viewportSize;
		bool m_isHovered;
		bool m_isFocused;

		// Stores the Viewport offset relative to the main window's top left corner
		glm::ivec2 m_offset;

		bool m_sizeChanged;
	};

}