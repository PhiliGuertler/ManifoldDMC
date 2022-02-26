// ######################################################################### //
// ### ImGuiLayer.cpp ###################################################### //
// ### The Implementation of ImGuiLayer.h                                ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "ImGuiLayer.h"

#include "imgui.h"
#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_opengl3.h"

#include "LazyEngine/Core/Application.h"

// TEMPORARY
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	// ######################################################################## //
	// ### ImGuiLayer ######################################################### //
	// ######################################################################## //

	ImGuiLayer::ImGuiLayer()
		: Layer("ImGuiLayer")
		, m_dockingEnabled(false)
		, m_viewports()
	{
		LAZYENGINE_PROFILE_FUNCTION();

	}

	ImGuiLayer::~ImGuiLayer() {
		LAZYENGINE_PROFILE_FUNCTION();

		// do nothing
	}

	void ImGuiLayer::onAttach() {
		LAZYENGINE_PROFILE_FUNCTION();

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;	// Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;	// Enable Gamepad Controls
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;		// Enable Multi-Viewport / Platform Windows
		//io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
		//io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		// ImGui::StyleColorsClassic();

		// when viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical
		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			style.WindowRounding = 0.f;
			style.Colors[ImGuiCol_WindowBg].w = 1.f;
		}

		Application& app = Application::getInstance();
		GLFWwindow *window = static_cast<GLFWwindow *>(app.getWindow().getNativeWindow());

		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 410");
	}

	void ImGuiLayer::onDetach() {
		LAZYENGINE_PROFILE_FUNCTION();

		// shutdown ImGui
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiLayer::onEvent(Event& event) {
		ImGuiIO& io = ImGui::GetIO();

		// if a viewport is being hovered over, don't consume the event
		for (auto viewport : m_viewports) {
			if (viewport.second->isHovered()) {
				return;
			}
		}

		EventDispatcher dispatcher(event);
		// go for all events like an idiot
		bool consumed = false;
		consumed |= event.isInCategory(EventCategory::EventCategoryMouse) & io.WantCaptureMouse;
		consumed |= event.isInCategory(EventCategory::EventCategoryKeyboard) & io.WantCaptureKeyboard;
		dispatcher.consumeEvent(consumed);
	}

	void ImGuiLayer::begin() {
		LAZYENGINE_PROFILE_FUNCTION();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (m_dockingEnabled) {
			beginDockmode();
		}
	}

	void ImGuiLayer::end() {
		LAZYENGINE_PROFILE_FUNCTION();

		if (m_dockingEnabled) {
			endDockmode();
		}

		ImGuiIO& io = ImGui::GetIO();
		Application& app = Application::getInstance();
		io.DisplaySize = ImVec2((float)app.getWindow().getWidth(), (float)app.getWindow().getHeight());

		// Rendering
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			GLFWwindow *backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}
	}

	glm::ivec2 ImGuiLayer::getViewportSize(const std::string& name) const {
		if (m_viewports.find(name) != m_viewports.end()) {
			return m_viewports.at(name)->getSize();
		}
		return { -1, -1 };
	}

	bool ImGuiLayer::isHovered(const std::string& name) const {
		return m_viewports.at(name)->isHovered();
	}

	bool ImGuiLayer::beginViewport(const std::string& name) {
		Ref<ImGuiViewport> viewport = nullptr;

		auto iter = m_viewports.find(name);
		if (iter == m_viewports.end()) {
			// create a new viewport
			m_viewports[name] = createRef<ImGuiViewport>(name);
			viewport = m_viewports[name];
		}
		else {
			viewport = iter->second;
		}

		viewport->begin();
		return viewport->hasResized();
		// do your rendering here (after this function obviously)
	}

	void ImGuiLayer::endViewport(const std::string& name) {
		auto iter = m_viewports.find(name);
		LAZYENGINE_CORE_ASSERT(iter != m_viewports.end(), "There exists no viewport with that name! Did you forget to call beginViewport?");
		
		iter->second->end();
	}

	void ImGuiLayer::beginDockmode() {
		static bool dockspaceIsOpen = true;

		static ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_None;

		// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		// because it would be confusing to have two docking targets within each other.
		ImGuiWindowFlags windowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

		// make the docking area as big as the whole window
		static bool opt_fullscreen_persistant = true;
		bool opt_fullscreen = opt_fullscreen_persistant;
		if (opt_fullscreen)
		{

			::ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->Pos);
			ImGui::SetNextWindowSize(viewport->Size);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			windowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		}

		// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
		if (dockspaceFlags & ImGuiDockNodeFlags_PassthruCentralNode) {
			windowFlags |= ImGuiWindowFlags_NoBackground;
		}

		// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
		// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive, 
		// all active windows docked into it will lose their parent and become undocked.
		// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise 
		// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace", &dockspaceIsOpen, windowFlags);
		ImGui::PopStyleVar();

		if (opt_fullscreen) {
			ImGui::PopStyleVar(2);
		}

		// Dockspace
		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
			ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspaceFlags);
		}

		// Menu on the top of the window
		if (ImGui::BeginMenuBar()) {
			renderMenu();
			ImGui::EndMenuBar();
		}

		// After this is where all ImGui::Begin()'s go.
		// They will of course be called in onImGuiRender() of each layer
	}

	void ImGuiLayer::renderMenu() {
		if (ImGui::BeginMenu("File"))
		{
			// Disabling fullscreen would allow the window to be moved to the front of other windows, 
			// which we can't undo at the moment without finer window depth/z control.
			//ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen_persistant);

			if (ImGui::MenuItem("Exit")) {
				Application::getInstance().close();
			}
			ImGui::EndMenu();
		}
	}

	void ImGuiLayer::endDockmode() {
		// Before this is where all ImGui::Begin()'s go.
		// They will of course be called in onImGuiRender() of each layer

		ImGui::End();
	}
}