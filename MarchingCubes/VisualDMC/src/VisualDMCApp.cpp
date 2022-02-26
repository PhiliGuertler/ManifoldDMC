// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### VisualDMCApp.cpp #################################################### //
// ### A graphical UI for the Dual Marching Cubes Algorithm.             ### //
// ######################################################################### //

#include <filesystem>

// Library imports
#include <LazyEngine/LazyEngine.h>
#include <LazyEngine/Core/EntryPoint.h>

// Project imports
#include "LayerCUDA.h"

class VisualImGuiLayer : public LazyEngine::ImGuiLayer {
public:
	VisualImGuiLayer(DMC::DMCLayer* layer)
		: m_layer(layer)
	{
		// empty
	}

	virtual void renderMenu() override {
		if (ImGui::BeginMenu("File")) {
			m_layer->renderMenu();
			ImGui::EndMenu();
		}
	}

protected:
	DMC::DMCLayer* m_layer;
};

class VisualDMC : public LazyEngine::Application {
public:
	VisualDMC() {
		auto layer = new DMC::DMCLayer();
		// replace the original ImGuiLayer with the custom VisualImGuiLayer
		popOverlay(m_imguiLayer);
		delete m_imguiLayer;
		m_imguiLayer = new VisualImGuiLayer(layer);
		m_imguiLayer->enableDocking(true);
		pushOverlay(m_imguiLayer);
		pushLayer(layer);

		LAZYENGINE_INFO("Working Directory: {0}", std::filesystem::current_path());
	}

	~VisualDMC() {}

};

LazyEngine::Application *LazyEngine::createApplication() {
	return new VisualDMC();
}
