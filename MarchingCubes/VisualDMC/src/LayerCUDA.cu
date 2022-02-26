// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### LayerCUDA.cu ######################################################## //
// ### This file implements everything of LayerCUDA.h that is relying on ### //
// ### CUDA functionality.                                               ### //
// ######################################################################### //

#include "LayerCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <thrust/extrema.h>

namespace DMC {

	DMCImGui::DMCImGui()
		: m_currentAlgorithm()
		, m_fileDialog()
		, m_selectedFile("./Data-and-Results/Base-Problems/1-Edge.bin")
		, m_surfaceCases()
		, m_currentSurfaceCase()
		, m_showImplicitPopup(false)
		, m_showLoadGridPopup(false)
		, m_isoValue(900.f)
		, m_dimensions({ 128,128,128 })
		, m_gridRefinement(0)
		, m_addZeroBoundary(true)
		, m_lockDimensions(true)
		, m_useImplicitSurface(false)
		, m_updateInstantly(false)
		, m_renderingModes()
		, m_currentRenderingMode()
		, m_cameraControllers()
		, m_currentCameraController()
		, m_lineThickness(1.f)
		, m_selection(nullptr)
		// Data
		, m_grid()
		, m_mesh(nullptr)
	{
		initialize();
	}

	void DMCImGui::updateGrid() {
		if (m_useImplicitSurface) {
			m_grid.generateVolume(m_dimensions, m_currentSurfaceCase.second);
			m_dimensions = m_grid.getDimensions();
		}
		else {
			try {
				m_grid.readFromFile(m_selectedFile, m_gridRefinement, m_addZeroBoundary);
			}
			catch (std::exception& e) {
				throw e;
			}
			m_dimensions = m_grid.getDimensions();
		}
		m_mesh->setDefaultQuad();
	}


	void DMCImGui::imEdgeCases() {
		for (auto& edgeCase : m_edgeCases) {
			if (ImGui::MenuItem(edgeCase.first.c_str())) {
				m_grid.generateEdgeCase(edgeCase.second);
				m_dimensions = m_grid.getDimensions();
				m_mesh->setDefaultQuad();
			}
		}
	}

	bool DMCLayer::onMouseEvent(LazyEngine::MouseButtonPressedEvent& event) {
		if (event.getMouseButton() == LazyEngine::MouseButtonCode::Button_Right) {
			auto& cam = m_currentCameraController.second;
			// Create a pixel ray into the scene
			LazyEngine::CameraRay ray = cam->shootPixelRay(LazyEngine::Input::getMousePosition());

			if (m_selection != nullptr) {
				m_selection->shootCameraRay(ray);
			}

		}

		// Don't consume the event
		return false;
	}
}