// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### LayerCUDA.cpp ####################################################### //
// ### This file implements everything of LayerCUDA.h that is not        ### //
// ### relying on CUDA functionality to improve build-times.             ### //
// ######################################################################### //

#include "LayerCUDA.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

namespace DMC {

	// ##################################################################### //
	// ### DMCImGui ######################################################## //
	// ##################################################################### //

	void DMCImGui::initialize() {
		// create algorithms
		m_currentAlgorithm = std::make_unique<DualMarchingCubes>();

		// initialize surface names
		m_surfaceCases["Sphere"] = SurfaceCase::Sphere;
		m_surfaceCases["Torus"] = SurfaceCase::Torus;
		m_surfaceCases["Two-holed Torus"] = SurfaceCase::TwoHoledTorus;
		m_surfaceCases["Four-holed Torus"] = SurfaceCase::FourHoledTorus;
		m_surfaceCases["Genus Two"] = SurfaceCase::GenusTwo;
		m_surfaceCases["iWP"] = SurfaceCase::iWP;
		m_surfaceCases["pwHybrid"] = SurfaceCase::pwHybrid;
		m_surfaceCases["NeoVius"] = SurfaceCase::neovius;
		m_surfaceCases["Goursat"] = SurfaceCase::Goursat;
		m_surfaceCases["Steiner-Roman"] = SurfaceCase::SteinerRoman;
		m_currentSurfaceCase = *m_surfaceCases.find("NeoVius");

		// Initialize edge cases
		m_edgeCases["Non-Manifold Vertex 0"] = EdgeCase::NonManifoldVertex0;
		m_edgeCases["Non-Manifold Vertex 1"] = EdgeCase::NonManifoldVertex1;

		// initialize rendering modes
		m_renderingModes["Solid"] = MeshMode::Solid;
		m_renderingModes["Wireframe Triangles"] = MeshMode::WireframeTriangles;
		m_renderingModes["Wireframe Quads"] = MeshMode::WireframeQuads;
		m_renderingModes["Solid + Triangles"] = MeshMode::SolidAndWireframeTriangles;
		m_renderingModes["Solid + Quads"] = MeshMode::SolidAndWireframeQuads;
		m_renderingModes["Points"] = MeshMode::Points;
		m_currentRenderingMode = *m_renderingModes.find("Solid + Quads");

		// initialize Camera Controllers
		m_cameraControllers["Orbital"] = std::make_shared<LazyEngine::PerspectiveOrbitalCameraController>(16.f / 9.f, LazyEngine::Constants::PI * 0.5f);
		m_cameraControllers["Free Flight"] = std::make_shared<LazyEngine::PerspectiveFreeCameraController>(16.f / 9.f, LazyEngine::Constants::PI * 0.5f);
		m_currentCameraController = *m_cameraControllers.find("Orbital");
		if (LazyEngine::Input::getConnectedGamepadCount() > 0) {
			for (auto& cameraController : m_cameraControllers) {
				cameraController.second->setGamepad(LazyEngine::Input::getFirstGamepad());
			}
		}

		// Set initial Camera positions and view directions.
		for (auto& cameraController : m_cameraControllers) {
			cameraController.second->rotateBy(glm::vec3(1.f, 0.f, 0.f), 0.9f);
			cameraController.second->setPosition(glm::vec3(0.f, 20.f, 20.f));
		}
		std::dynamic_pointer_cast<LazyEngine::OrbitalCameraController>(m_cameraControllers["Orbital"])->setFocusPoint(glm::vec3(0.f));

		// Initialize the mesh of the viewport.
		m_mesh = std::make_shared<Mesh>();

		// initialize the numbers of non-manifold elements
		m_numNonManifoldEdges4 = 0;
		m_numNonManifoldEdges3 = 0;
		m_numNonManifoldVertices = 0;

		m_numQuads = 0;
		m_numQuadsWithNonManifoldEdges1 = 0;
		m_numQuadsWithNonManifoldEdges2 = 0;
		m_numQuadsWithNonManifoldEdges3 = 0;
		m_numQuadsWithNonManifoldEdges4 = 0;

		m_alphaValue = 1.f;

		// Initialize File-Browsers
		m_fileDialog = ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc);
		m_fileDialog.SetTitle("Load Uniform Grid");
		m_fileDialog.SetTypeFilters({ ".bin" });

		m_writeFileDialog = ImGui::FileBrowser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CloseOnEsc);
		m_writeFileDialog.SetTitle("Write Mesh as .obj");
		m_writeFileDialog.SetTypeFilters({ ".obj" });
	}

	void DMCImGui::setupDockspace() {
		ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
		ImGuiID dockspaceID = ImGui::GetID("MainDockSpace");

		ImGui::DockBuilderRemoveNode(dockspaceID);
		ImGui::DockBuilderAddNode(dockspaceID, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
		// split dockspace into three nodes
		// main View
		auto dockIDLeft = ImGui::DockBuilderSplitNode(dockspaceID, ImGuiDir_Left, 0.3f, nullptr, &dockspaceID);
		auto dockIDRight = ImGui::DockBuilderSplitNode(dockspaceID, ImGuiDir_Right, 0.5f, nullptr, &dockspaceID);

		auto dockIDVisualDMC = ImGui::DockBuilderSplitNode(dockIDLeft, ImGuiDir_Up, 0.3f, nullptr, &dockIDLeft);
		ImGui::DockBuilderDockWindow("Visual DMC", dockIDVisualDMC);
		auto dockIDOptionsBottom = ImGui::DockBuilderSplitNode(dockIDLeft, ImGuiDir_Down, 0.7f, nullptr, &dockIDLeft);
		ImGui::DockBuilderDockWindow("DMC+Post-Processes", dockIDLeft);

		ImGui::DockBuilderDockWindow("Performance", dockIDOptionsBottom);
		ImGui::DockBuilderDockWindow("Statistics", dockIDOptionsBottom);
		ImGui::DockBuilderDockWindow("Camera", dockIDOptionsBottom);
		ImGui::DockBuilderDockWindow("File", dockIDOptionsBottom);
		ImGui::DockBuilderDockWindow("Mesh", dockIDOptionsBottom);

		ImGui::DockBuilderDockWindow("Viewport", dockspaceID);

		ImGui::DockBuilderDockWindow("Current Selection", dockIDRight);

		ImGui::DockBuilderFinish(dockspaceID);
	}

	void DMCImGui::render() {
		static bool isFirstRender = true;
		if (isFirstRender) {
			setupDockspace();
			isFirstRender = false;
		}

		ImGui::Begin("Visual DMC");
		imRun();
		imIsoValue();
		ImGui::End();

		ImGui::Begin("DMC+Post-Processes");
		imDMCOptions();
		ImGui::End();

		ImGui::Begin("Performance");
		imPerformance();
		ImGui::End();

		ImGui::Begin("Statistics");
		imStatistics();
		ImGui::End();

		ImGui::Begin("File");
		imFile();
		ImGui::End();

		ImGui::Begin("Mesh");
		imMeshOptions();
		ImGui::End();

		ImGui::Begin("Camera");
		imCamera();
		ImGui::End();
		{
			if (m_showImplicitPopup) {
				ImGui::OpenPopup("Implicit");
				m_showImplicitPopup = false;
			}
			imImplicitSurfacePopups();
		}
		{
			if (m_showLoadGridPopup) {
				ImGui::OpenPopup("Load-Grid");
				m_showLoadGridPopup = false;
			}
			imLoadGridPopup();
		}

		if (m_selection != nullptr) {
			m_selection->displayImGuiOptions();
		}

		handleFileDialogs();

	}

	void DMCImGui::handleFileDialogs() {
		m_fileDialog.Display();
		m_writeFileDialog.Display();

		if (m_fileDialog.HasSelected()) {
			const std::string selection = m_fileDialog.GetSelected().string();
			m_selectedFile = selection;
			m_dimensions = m_grid.readDimensionsFromFile(m_selectedFile);
			m_useImplicitSurface = false;
			bool anExceptionOccured = false;
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				anExceptionOccured = true;
			}

			if (m_updateInstantly && !anExceptionOccured) {
				// open the selected file, then run the algorithm
				runAlgorithm();
			}

			m_fileDialog.ClearSelected();
		}

		if (m_writeFileDialog.HasSelected()) {
			const std::string selection = m_writeFileDialog.GetSelected().string();

			m_mesh->writeToFile(selection);

			m_writeFileDialog.ClearSelected();
		}
	}

	void DMCImGui::runAlgorithm() {
		MemoryMonitor::getInstance().resetMaxTotalByteCount();

		if (m_grid.getSizeInBytes() < 1) {
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				return;
			}
		}
		if (m_currentAlgorithm) {
			// reset performance static id, which is in charge of performance colors
			Performance::resetStaticId();
			m_lastPerformances = m_currentAlgorithm->run(m_isoValue, *m_mesh, m_grid);

			// set the new iso-value
			m_currentSurface.m_isoValue = m_isoValue;

			SelectionImGuiOptions options;
			if (m_selection != nullptr) options = m_selection->getOptions();

			if (m_currentAlgorithm) {
				auto& halfedgeMesh = m_currentAlgorithm->getHalfedgeMesh();
				m_selection = std::make_unique<Selection>(*m_mesh, halfedgeMesh, m_grid);
				m_selection->setOptions(options);
				m_selection->setOnFocusCamera(std::bind(&DMCImGui::focusPoint, this, std::placeholders::_1));
				m_selection->setGetIsoValue(std::bind(&DMCImGui::getIsoValue, this));

				m_numNonManifoldEdges4 = m_currentAlgorithm->getNumNonManifoldEdges4();
				m_numNonManifoldEdges3 = m_currentAlgorithm->getNumNonManifoldEdges3();
				m_numEdges = m_currentAlgorithm->getNumEdges();
				m_numNonManifoldVertices = m_currentAlgorithm->getNumNonManifoldVertices();

				m_numQuads = m_currentAlgorithm->getNumQuads();
				m_numQuadsWithNonManifoldEdges1 = m_currentAlgorithm->getNumQuadsWithNonManifoldEdges(1);
				m_numQuadsWithNonManifoldEdges2 = m_currentAlgorithm->getNumQuadsWithNonManifoldEdges(2);
				m_numQuadsWithNonManifoldEdges3 = m_currentAlgorithm->getNumQuadsWithNonManifoldEdges(3);
				m_numQuadsWithNonManifoldEdges4 = m_currentAlgorithm->getNumQuadsWithNonManifoldEdges(4);
			}
			else {
				m_selection = nullptr;
			}
		}
	}

	void DMCImGui::focusPoint(const glm::vec3& point) {
		if (point.x != point.x || point.y != point.y || point.z != point.z) return;
		for (auto& cam : m_cameraControllers) {
			auto pointer = std::dynamic_pointer_cast<LazyEngine::OrbitalCameraController>(cam.second);
			if (pointer) {
				pointer->setFocusPoint(point);
			}
			cam.second->lookAt(point, cam.second->getOrientation() * glm::vec3(0.f, 1.f, 0.f));
		}
	}

	void DMCImGui::imRun() {
		ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.05f, 0.95f, 0.7f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.05f, 0.95f, 0.8f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.05f, 0.95f, 0.95f));
		if (ImGui::Button("Run Algorithm")) {
			m_currentAlgorithm->setSplittingStep(SplittingStep::Ignore);
			runAlgorithm();
		}
		ImGui::PopStyleColor(3);

		ImGui::Checkbox("Update Instantly", &m_updateInstantly);
	}

	void DMCImGui::imIsoValue() {
		if (ImGui::CollapsingHeader("Iso-Value Options", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent();
			static float maxIsoValue = 1000.f;
			static float minIsoValue = 0.f;
			ImGui::InputFloat("Min Iso-Value", &minIsoValue);
			ImGui::InputFloat("Max Iso-Value", &maxIsoValue);
			if (ImGui::SliderFloat("Iso-Value", &m_isoValue, minIsoValue, maxIsoValue)) {
				if (m_updateInstantly) {
					runAlgorithm();
				}
			}
			ImGui::Unindent();
		}
	}

	void DMCImGui::imDMCOptions() {
		m_currentAlgorithm->renderImGui();
	}


	void DMCImGui::imPerformance() {
		static int currentSelection = -1;
		m_lastPerformances.renderChildrenUntilMatch(&currentSelection, m_lastPerformances.getTime());
	}

	void DMCImGui::imStatistics() {
		// Get the Memory consumption on GPU

		if (ImGui::Button("Run Performance Tests")) {
			performTestRuns();
		}
		if (ImGui::Button("Run all Performance Tests at once!")) {
			performAllTestRuns();
		}

		if (ImGui::CollapsingHeader("Memory Usage")) {
			MemoryMonitor::getInstance().displayGPUInfoImGui();
			ImGui::Separator();
			MemoryMonitor::getInstance().displayPercentagesImGui();
			ImGui::Separator();
			if (ImGui::CollapsingHeader("Memory Usage Details")) {
				ImGui::Indent();
				MemoryMonitor::getInstance().displayLargeButtonsImGui();
				ImGui::Unindent();
			}
			MemoryMonitor::getInstance().displayDetailsImGui();
		}

		if (ImGui::CollapsingHeader("Mesh Statistics")) {
			size_t vertexByteSize = m_mesh->getVertices().getSize();
			size_t vertexCount = vertexByteSize / sizeof(Vertex);

			ImGui::Text("GRID");
			ImGui::Text("#Cells:                     \t%9d", m_grid.getCellCount());
			ImGui::Text("x:                          \t%4d", m_grid.getDimensions().x);
			ImGui::Text("y:                          \t%4d", m_grid.getDimensions().y);
			ImGui::Text("z:                          \t%4d", m_grid.getDimensions().z);

			ImGui::Separator();
			ImGui::Text("EDGES");
			ImGui::Text("#Edges:                     \t%9d", m_numEdges);
			float edge4 = (static_cast<float>(m_numNonManifoldEdges4) / static_cast<float>(m_numEdges)) * 100.f;
			ImGui::Text("#Non-Manifold Edges (4):    \t%9d (%.2f%%)", m_numNonManifoldEdges4, edge4);
			float edge3 = (static_cast<float>(m_numNonManifoldEdges3) / static_cast<float>(m_numEdges)) * 100.f;
			ImGui::Text("#Non-Manifold Edges (3):    \t%9d (%.2f%%)", m_numNonManifoldEdges3, edge3);

			ImGui::Separator();
			ImGui::Text("VERTICES");
			ImGui::Text("#Vertices:                  \t%9d", vertexCount);
			float vertex = (static_cast<float>(m_numNonManifoldVertices) / static_cast<float>(vertexCount)) * 100.f;
			ImGui::Text("#Non-Manifold Vertices:     \t%9d (%.2f%%)", m_numNonManifoldVertices, vertex);

			ImGui::Separator();
			ImGui::Text("QUADS");
			ImGui::Text("#Quads:                     \t%9d", m_numQuads);
			float quad1 = (static_cast<float>(m_numQuadsWithNonManifoldEdges1) / static_cast<float>(m_numQuads)) * 100.f;
			ImGui::Text("#Quads w/ 1 NM-Edge:        \t%9d (%.2f%%)", m_numQuadsWithNonManifoldEdges1, quad1);
			float quad2 = (static_cast<float>(m_numQuadsWithNonManifoldEdges2) / static_cast<float>(m_numQuads)) * 100.f;
			ImGui::Text("#Quads w/ 2 NM-Edges:       \t%9d (%.2f%%)", m_numQuadsWithNonManifoldEdges2, quad2);
			float quad3 = (static_cast<float>(m_numQuadsWithNonManifoldEdges3) / static_cast<float>(m_numQuads)) * 100.f;
			ImGui::Text("#Quads w/ 3 NM-Edges:       \t%9d (%.2f%%)", m_numQuadsWithNonManifoldEdges3, quad3);
			float quad4 = (static_cast<float>(m_numQuadsWithNonManifoldEdges4) / static_cast<float>(m_numQuads)) * 100.f;
			ImGui::Text("#Quads w/ 4 NM-Edges:       \t%9d (%.2f%%)", m_numQuadsWithNonManifoldEdges4, quad4);

			if (ImGui::Button("Write Stats to Folder")) {
				const std::string directory = "./Statistics";
				auto currentFile = std::filesystem::path(m_selectedFile);
				std::string fileName = currentFile.filename().string();
				// only use the part of the file up to the first underscore or dot!
				std::string delimiter = "_.";
				std::string token = fileName.substr(0, fileName.find(delimiter));
				fileName = token + (m_addZeroBoundary ? "*" : "");
				// Append Grid Data

				writeMeshStatistics(directory, fileName);
				writePerformanceStatistics(directory, fileName);
			}
		}
	}

	void DMCImGui::imFile() {
		if (ImGui::Checkbox("Add Boundary Padding (Fixes 3-Face-Edges and Non-Manifold Vertices)", &m_addZeroBoundary)) {
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				return;
			}
			if (m_updateInstantly) {
				runAlgorithm();
			}
		}
		if (ImGui::SliderInt("Grid-Refinement", &m_gridRefinement, 0, 10)) {
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				return;
			}
			if (m_updateInstantly) {
				runAlgorithm();
			}
		}
	}

	void DMCImGui::imMeshOptions() {
		renderCombo<MeshMode>(
			"Rendering Mode", m_renderingModes, m_currentRenderingMode, [&](const std::pair<std::string, MeshMode>& selection) {
				m_currentRenderingMode = selection;
			});

		if (ImGui::SliderFloat("Line Thickness", &m_lineThickness, 1.f, 10.f)) {
			LazyEngine::RenderCommand::setLineWidth(m_lineThickness);
		}
		if (ImGui::SliderFloat("Alpha Value", &m_alphaValue, 0.f, 1.f)) {
			m_mesh->setAlphaValue(m_alphaValue);
		}
	}

	void DMCImGui::imCamera() {
		if (ImGui::Button("Reset Camera Focus Point")) {
			std::dynamic_pointer_cast<LazyEngine::OrbitalCameraController>(m_cameraControllers["Orbital"])->setFocusPoint(glm::vec3(0.f));
		}
		bool useOrbitalCamera = m_currentCameraController.first == m_cameraControllers.find("Orbital")->first;
		bool useFreeCamera = !useOrbitalCamera;
		if (ImGui::Checkbox("Orbital Camera", &useOrbitalCamera)) {
			const std::string orbitalCam = "Orbital";
			auto cam = std::dynamic_pointer_cast<LazyEngine::OrbitalCameraController>(m_cameraControllers[orbitalCam]);
			cam->setPosition(m_currentCameraController.second->getPosition());
			m_currentCameraController = *m_cameraControllers.find(orbitalCam);
		}
		if (ImGui::Checkbox("Free flight Camera", &useFreeCamera)) {
			const std::string freeFlightCam = "Free Flight";
			auto cam = m_cameraControllers.find(freeFlightCam)->second;
			cam->setOrientation(m_currentCameraController.second->getOrientation());
			cam->setPosition(m_currentCameraController.second->getPosition());
			m_currentCameraController = *m_cameraControllers.find(freeFlightCam);
		}

		ImGui::Separator();
		ImGui::Text("Position: [%.2f, %.2f, %.2f]", VECCC(m_currentCameraController.second->getPosition()));
		ImGui::Text("View-Direction: [%.2f, %.2f, %.2f]", VECCC(m_currentCameraController.second->getForwardDirection()));
	}


	void DMCImGui::imImplicitSurfaces() {
		bool wasClicked = false;
		for (auto& surface : m_surfaceCases) {
			if (ImGui::MenuItem(surface.first.c_str())) {
				m_currentSurfaceCase = surface;
				wasClicked = true;
			}
		}
		m_showImplicitPopup = wasClicked;
	}

	void DMCImGui::imImplicitSurfacePopups() {
		static glm::ivec3 gridDimensions = glm::ivec3(128, 128, 128);
		if (ImGui::BeginPopupModal("Implicit")) {
			if (ImGui::Checkbox("Lock Dimensions", &m_lockDimensions)) {
				// set all dimensions equal
				gridDimensions.y = gridDimensions.x;
				gridDimensions.z = gridDimensions.x;
			}
			if (m_lockDimensions) {
				// Display only one slider, that adjusts all three values simultaneously
				if (ImGui::SliderInt("Grid Dimensions", &gridDimensions.x, 5, 512)) {
					gridDimensions.y = gridDimensions.x;
					gridDimensions.z = gridDimensions.x;
				}
			}
			else {
				// Display three sliders, one for each dimension
				ImGui::SliderInt3("Grid Dimensions", glm::value_ptr(gridDimensions), 5, 512);
			}

			if (ImGui::Button("OK", ImVec2(120, 0))) {
				m_useImplicitSurface = true;
				m_dimensions = gridDimensions;
				bool anExceptionOccured = false;
				try {
					updateGrid();
				}
				catch (std::exception& e) {
					// Do nothing, actually.
					anExceptionOccured = true;
				}

				if (m_updateInstantly && !anExceptionOccured) {
					runAlgorithm();
				}

				ImGui::CloseCurrentPopup();
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}

	void DMCImGui::imLoadGridPopup() {
		if (ImGui::BeginPopupModal("Load-Grid")) {
			ImGui::Checkbox("Add Boundary Padding (Fixes 3-Face-Edges and Non-Manifold Vertices)", &m_addZeroBoundary);
			ImGui::SliderInt("Grid-Refinement", &m_gridRefinement, 0, 10);

			if (ImGui::Button("OK", ImVec2(120, 0))) {
				m_useImplicitSurface = false;
				m_fileDialog.Open();

				ImGui::CloseCurrentPopup();
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}

	void DMCImGui::performAllTestRuns() {
		std::vector<std::pair<std::string, float>> testFiles = {
			{"./data/Big Data/Volumes/Baby_ushort_256_256_98.bin", 147.5f},
			{"./data/Big Data/Volumes/Bruce_ushort_256_256_156.bin", 147.5f},
			{"./data/Big Data/Volumes/Angio_ushort_384_512_80.bin", 47.f},
			{"./data/Big Data/MRI-Head_ushort_256_256_256.bin", 105.f},
			{"./data/Big Data/Volumes/Carp_ushort_256_256_512.bin", 1500.f},
			{"./data/Big Data/Volumes/CT-Abdomen_ushort_512_512_147.bin", 80.f},
			{"./data/Big Data/Volumes/cenovix_ushort_512_512_361.bin", 900.f},
			{"./data/Big Data/head_ushort_512_512_641.bin", 900.f},
			{"./data/Big Data/mecanix_ushort_512_512_743.bin", 1200.f},
			{"./data/Big Data/Volumes/becken_ushort_512_512_1047.bin", 900.f},
		};

		const float originalIsoValue = m_isoValue;
		const std::string originalFile = m_selectedFile;
		for (const auto& testFile : testFiles) {
			m_isoValue = testFile.second;
			m_selectedFile = testFile.first;
			performTestRuns();
		}
		m_selectedFile = originalFile;
		m_isoValue = originalIsoValue;
	}

	void DMCImGui::performTestRuns() {
		const std::string directory = "./Statistics/";
		auto currentFile = std::filesystem::path(m_selectedFile);
		std::string fileName = currentFile.filename().string();
		// only use the part of the file up to the first underscore or dot!
		std::string delimiter = "_.";
		std::string token = fileName.substr(0, fileName.find(delimiter));
		fileName = token + (m_addZeroBoundary ? "*" : "");

		// Disable Post Processing
		const bool originalZeroBoundaryFlag = m_addZeroBoundary;
		m_currentAlgorithm->setSplittingStep(SplittingStep::None);
		const bool wasOriginallySplitting = m_currentAlgorithm->getSplitNonManifoldHalfedges();
		m_currentAlgorithm->getSplitNonManifoldHalfedges() = false;
		// First run the algorithm without any fixes
		{
			m_addZeroBoundary = false;
			bool anExceptionOccured = false;
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				anExceptionOccured = true;
			}
			if (!anExceptionOccured) {
				runAlgorithm();
				writeMeshStatistics(directory, fileName);
				writePerformanceStatistics(directory, fileName);
			}
		}

		// Second, run the algorithm with an enlarged grid
		{
			m_addZeroBoundary = true;
			bool anExceptionOccured = false;
			try {
				updateGrid();
			}
			catch (std::exception& e) {
				anExceptionOccured = true;
			}
			if (!anExceptionOccured) {
				runAlgorithm();
				writeMeshStatistics(directory, fileName);
				writePerformanceStatistics(directory, fileName);
			}
		}

		// Third, run the algorithm with all fixes
		{
			m_currentAlgorithm->setSplittingStep(SplittingStep::All);
			m_currentAlgorithm->getSplitNonManifoldHalfedges() = true;
			runAlgorithm();
			writeMeshStatistics(directory, fileName);
			writePerformanceStatistics(directory, fileName);
		}

		// reset manual splitting step and zero boundary flag
		m_currentAlgorithm->setSplittingStep(SplittingStep::Ignore);
		m_addZeroBoundary = originalZeroBoundaryFlag;
		m_currentAlgorithm->getSplitNonManifoldHalfedges() = wasOriginallySplitting;
	}

	void DMCImGui::writeMeshStatistics(const std::string& directory, const std::string& fileName) {
		size_t vertexByteSize = m_mesh->getVertices().getSize();
		size_t vertexCount = vertexByteSize / sizeof(Vertex);

		// Append Grid Data
		{
			std::ofstream output(directory + "Statistics.json", std::ios_base::app);
			output << "{\n"
				<< "\t\"fileName\": " << "\"" << fileName << "\",\n"
				<< "\t\"dimensions\": {\n"
				<< "\t\t\"x\": " << m_grid.getDimensions().x << ",\n"
				<< "\t\t\"y\": " << m_grid.getDimensions().y << ",\n"
				<< "\t\t\"z\": " << m_grid.getDimensions().z << "\n"
				<< "\t},\n"
				<< "\t\"numCells\": " << (m_grid.getDimensions().x - 1) * (m_grid.getDimensions().y - 1) * (m_grid.getDimensions().z - 1) << ",\n"
				<< "\t\"isoValue\": " << m_isoValue << ",\n"

				<< "\t\"edges\": {\n"
				<< "\t\t\"numEdges\": " << m_numEdges << ",\n"
				<< "\t\t\"3-Edges\": " << m_numNonManifoldEdges3 << ",\n"
				<< "\t\t\"4-Edges\": " << m_numNonManifoldEdges4 << "\n"
				<< "\t},\n"

				<< "\t\"vertices\": {\n"
				<< "\t\t\"numVertices\": " << vertexCount << ",\n"
				<< "\t\t\"nonManifoldVertices\": " << m_numNonManifoldVertices << "\n"
				<< "\t},\n"

				<< "\t\"quads\": {\n"
				<< "\t\t\"numQuads\": " << m_numQuads << ",\n"
				<< "\t\t\"1-Quads\": " << m_numQuadsWithNonManifoldEdges1 << ",\n"
				<< "\t\t\"2-Quads\": " << m_numQuadsWithNonManifoldEdges2 << ",\n"
				<< "\t\t\"3-Quads\": " << m_numQuadsWithNonManifoldEdges3 << ",\n"
				<< "\t\t\"4-Quads\": " << m_numQuadsWithNonManifoldEdges4 << "\n"
				<< "\t}\n"

				<< "}," << std::endl;
		}
	}

	void DMCImGui::writePerformanceStatistics(const std::string& directory, const std::string& fileName) {
		std::ofstream output(directory + "Performances.json", std::ios_base::app);
		output << "{\n"
			<< "\t\"file\": " << "\"" << fileName << "\"" << ",\n"
			<< "\t\"isoValue\": " << m_isoValue << ",\n"
			<< "\t\"performance\":\n"
			<< m_lastPerformances.toString(1) << ",\n"
			<< "\t\"maxMemoryUsage\": " << MemoryMonitor::getInstance().getMaxTotalByteCount() << "\n"
			<< "},"
			<< std::endl;

	}

	// ##################################################################### //
	// ### DMCLayer ######################################################## //
	// ##################################################################### //

	DMCLayer::DMCLayer()
		: LazyEngine::Layer("DMCLayer")
		, DMCImGui()
	{
		// empty
	}

	DMCLayer::~DMCLayer() {
		// empty
	}

	void DMCLayer::onAttach() {
		// nothing
	}


	void DMCLayer::onImGuiRender() {
		render();
	}

	void DMCLayer::onEvent(LazyEngine::Event& event) {
		LazyEngine::EventDispatcher dispatcher(event);
		// Disconnect a gamepad before handling the camera's events
		dispatcher.dispatch<LazyEngine::GamepadConnectionEvent>(LAZYENGINE_BIND_EVENT_FUNC(DMCLayer::onGamepadConnection));
		dispatcher.dispatch<LazyEngine::MouseButtonPressedEvent>(LAZYENGINE_BIND_EVENT_FUNC(DMCLayer::onMouseEvent));

		for (auto& cameraController : m_cameraControllers) {
			// as these include WindowResize-Events, every camera controller must receive them
			cameraController.second->onEvent(event);
		}
	}

	bool DMCLayer::onGamepadConnection(LazyEngine::GamepadConnectionEvent& event) {
		if (event.hasConnected()) {
			auto gamepad = LazyEngine::Input::getGamepad(event.getGamepadID());
			for (auto& cameraController : m_cameraControllers) {
				cameraController.second->setGamepad(gamepad);
			}
			LAZYENGINE_INFO("Gamepad connected to App!");
		}
		else {
			LAZYENGINE_INFO("Gamepad disconnected from App!");
		}

		// don't consume the event
		return false;
	}

	void DMCLayer::onUpdate(const LazyEngine::TimeStep& deltaTime) {
		m_currentCameraController.second->onUpdate(deltaTime);
	}

	void DMCLayer::onRender() {
		// Clear the Framebuffer
		const float color = 0.1f;
		LazyEngine::RenderCommand::setClearColor({ color, color, color, 1.f });
		LazyEngine::RenderCommand::clear();

		// now render the scene
		LazyEngine::Renderer::beginScene(m_currentCameraController.second->getCamera());

		switch (m_currentRenderingMode.second) {
		case MeshMode::Solid:
			m_mesh->render();
			break;
		case MeshMode::WireframeTriangles:
			m_mesh->renderWireframeTriangles();
			break;
		case MeshMode::WireframeQuads:
			m_mesh->renderWireframeQuads();
			break;
		case MeshMode::SolidAndWireframeTriangles:
			m_mesh->render();
			m_mesh->renderWireframeTriangles(true);
			break;
		case MeshMode::SolidAndWireframeQuads:
			m_mesh->render();
			m_mesh->renderWireframeQuads(true);
			break;
		case MeshMode::Points:
			m_mesh->renderPoints();
			break;
		default:
			m_mesh->render();
		}

		if (m_selection != nullptr) {
			m_selection->renderHighlights();
		}

		LazyEngine::Renderer::endScene();
	}


	void DMCLayer::renderMenu() {
		if (ImGui::MenuItem("Load Grid")) {
			m_showLoadGridPopup = true;
		}
		if (ImGui::BeginMenu("Generate Implicit Surface")) {
			imImplicitSurfaces();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Generate Edge Case")) {
			imEdgeCases();
			ImGui::EndMenu();
		}
		if (ImGui::MenuItem("Save Mesh as .obj")) {
			m_writeFileDialog.Open();
		}
		if (m_selection != nullptr) {
			m_selection->displayImGuiMenu();
		}
	}

}
