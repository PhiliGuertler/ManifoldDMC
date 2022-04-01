#pragma once

// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### LayerCUDA.h ######################################################### //
// ### Defines the LazyEngine-Layer that is responsible for UI and       ### //
// ### Viewport of Dual Marching Cubes and its Post-Processes.           ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui-filebrowser/imfilebrowser.h>

#include "DMC/Mesh.h"
#include "DMC/DualMarchingCubes.h"

#include "Selection.h"

namespace DMC {

	// Defines Render-Modes of the Mesh in the Viewport.
	enum MeshMode: int {
		Solid = 0,
		WireframeTriangles = 1,
		WireframeQuads = 2,
		SolidAndWireframeTriangles = 3,
		SolidAndWireframeQuads = 4,
		Points = 5
	};

	// Defines the currently loaded Surface (or rather its grid's data)
	struct SelectedSurface {
		float m_isoValue = 0.f;
		
		bool m_usesFile = true;
		SurfaceCase m_surfaceCase;
		std::string m_fileName = "";
	};

	// Takes care of all mutable Data that is supposed to be modifiable using the GUI.
	class DMCImGui {
	public:
		DMCImGui();

		void render();

		inline const UniformGridHost<float>& getGrid() const {
			return m_grid;
		}

		inline Mesh& getMesh() {
			return *m_mesh;
		}

		inline std::shared_ptr<LazyEngine::CameraController> getCameraController() {
			return m_currentCameraController.second;
		}

		inline MeshMode getRenderingMode() const {
			return m_currentRenderingMode.second;
		}

	protected:

		void imRun();

		void imIsoValue();
		void imDMCOptions();

		void imPerformance();
		void imStatistics();
		void imFile();
		void imMeshOptions();

		void imLightingOptions();
		void imCamera();

		void imImplicitSurfaces();
		void imImplicitSurfacePopups();

		void imEdgeCases();

		void setupDockspace();
		
		void imLoadGridPopup();

		/**
		 *	Renders a ImGui Dropdown Combo.
		 *	@param optionName: The name and label of the ImGui-Element (must be unique)
		 *	@param options: A list of <label, value> pairs. 
		 *		The label is the part that will be selectable in the dropdown, while the <label, value> pair 
		 *		will be forwarded to the callback function on selection.
		 *	@param selectedOption: The currently selected option's label.
		 *	@param onSelection: A callback function that will be executed when a selection occurs.
		 *		It takes the option pair (label, value) as its argument.
		 */
		template <typename T>
		inline void renderCombo(const std::string& optionName, const std::map<std::string, T>& options, const std::string& selectedOption, const std::function<void(const std::pair<std::string, T>&)>& onSelection) {
			// find the selected option in the options array
			auto iter = options.find(selectedOption);
			if (iter == options.end()) {
				ImGui::Text("Unknown Option selected!");
				return;
			}
			if (ImGui::BeginCombo(optionName.c_str(), iter->first.c_str())) {
				// The dropdown is displayed, display all items
				for (auto& i : options) {
					bool isSelected = i.first == selectedOption;
					if (ImGui::Selectable(i.first.c_str(), isSelected)) {
						// handle the selection
						onSelection(i);
					}
					// highlight the currently selected item
					if (isSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
		}

		/**
		 *	Renders a combo (aka Dropdown Selection Widget)
		 *	@param optionName: The name and label of the widget (must be unique)
		 *  @param options: An array of selectable options
		 *  @param selectedOption: The currently selected option
		 *  @param onSelection: A callback function that will be called with the index of the new selection if a selection happens.
		 */
		template <typename T>
		inline void renderCombo(const std::string& optionName, const std::map<std::string, T>& options, const std::pair<std::string, T>& selectedOption, const std::function<void(const std::pair<std::string, T>&)>& onSelection) {
			return renderCombo(optionName, options, selectedOption.first, onSelection);
		}

		// --- Algorithm Specifics --- //
		/**
		 *	Renders the Stats of the currently selected Algorithm
		 */
		//void renderAlgorithmStats();
		/**
		 *	Renders the Algorithm's last performance
		 */
		//void renderAlgorithmPerformance();
		/**
		 *	Renders Algorithm Specific Options of the currently selected Algorithm
		 */
		//void renderAlgorithmOptions();

		// --- Grid-Value Specifics --- //
		/**
		 *	Renders a file selection screen
		 */
		//void renderFileSelection();

		/**
		 *	Renders a implicit function selection screen
		 */
		//void renderSurfaceSelection();
		/**
		 *	Renders Gridoptions, e.g. The dimensions
		 */
		//void renderGridOptions();


		// --- Rendering Specifics --- //
		/**
		 *	Renders rendering options, e.g. wireframe/solid selection
		 */
		//void renderRenderingOptions();
		/**
		 *	Renders rendering stats, e.g. the camera's position
		 */
		//void renderRenderingStats();

		// --- Mesh Specifics --- //
		//void renderMeshOptions();


		// --- File Dialogs --- //
		void handleFileDialogs();

		void updateGrid();
		void runAlgorithm();

		void performAllTestRuns();
		void performTestRuns();
		void writeMeshStatistics(const std::string& directory, const std::string& fileName);
		void writePerformanceStatistics(const std::string& directory, const std::string& fileName);

	protected:
		void initialize();

		void focusPoint(const glm::vec3& point);

		inline float getIsoValue() {
			return m_isoValue;
		}

	protected:
		
		// --- Algorithms --- //
		std::unique_ptr<DualMarchingCubes> m_currentAlgorithm;

		// --- File Selection --- //
		ImGui::FileBrowser m_fileDialog;
		std::string m_selectedFile;

		ImGui::FileBrowser m_writeFileDialog;
		std::string m_selectedOutputFile;

		// --- Implicit Surfaces --- //
		std::map<std::string, SurfaceCase> m_surfaceCases;
		std::pair<std::string, SurfaceCase> m_currentSurfaceCase;

		std::map<std::string, EdgeCase> m_edgeCases;

		bool m_showImplicitPopup;
		bool m_showLoadGridPopup;

		// --- Grid Options --- //
		float m_isoValue;
		glm::ivec3 m_dimensions;
		int m_gridRefinement;
		bool m_addZeroBoundary;
		// If true, all three dimensions will be adjusted at the same time
		bool m_lockDimensions;
		// If true, an implicit surface will be used, otherwise a file dialog appears
		bool m_useImplicitSurface;
		// If true, slider updates are applied immediatley to the grid and the current algorithm will be rerun
		bool m_updateInstantly;

		// --- Rendering Options --- //
		std::map<std::string, MeshMode> m_renderingModes;
		std::pair<std::string, MeshMode> m_currentRenderingMode;

		std::map<std::string, std::shared_ptr<LazyEngine::CameraController>> m_cameraControllers;
		std::pair<std::string, std::shared_ptr<LazyEngine::CameraController>> m_currentCameraController;

		float m_lineThickness;
		float m_alphaValue;
		
		// --- Lighting Options --- //
		glm::vec2 m_lightOffset;
		bool m_snapLightingToCamera;

		// --- Viewport Selection --- //
		std::unique_ptr<Selection> m_selection;

		// --- Data --- //
		UniformGridHost<float> m_grid;
		// This Mesh will be rendered in the viewport!
		std::shared_ptr<Mesh> m_mesh;

		SelectedSurface m_currentSurface;

		Performance m_lastPerformances;

		// The amount of edges that connect 4 faces
		int m_numNonManifoldEdges4;
		// The amount of edges that connect 3 faces
		int m_numNonManifoldEdges3;
		// The total amount of edges
		int m_numEdges;

		// The amount of Vertices that are not part of exactly one face-fan
		int m_numNonManifoldVertices;

		int m_numQuads;
		int m_numQuadsWithNonManifoldEdges1;
		int m_numQuadsWithNonManifoldEdges2;
		int m_numQuadsWithNonManifoldEdges3;
		int m_numQuadsWithNonManifoldEdges4;

		// outputting statistics
		bool m_runTests;
	};
	
	/**
	 *	The actual LazyEngine-Layer that will be displaying GUI and Viewport.
	 */
	class DMCLayer : public LazyEngine::Layer, DMCImGui {
	public:
		DMCLayer();
		virtual ~DMCLayer();

		virtual void onAttach() override;

		virtual void onUpdate(const LazyEngine::TimeStep& deltaTime) override;

		virtual void onRender() override;
		virtual void onImGuiRender() override;

		virtual void onEvent(LazyEngine::Event& event) override;

		void renderMenu();

	protected:
		bool onGamepadConnection(LazyEngine::GamepadConnectionEvent& event);
		bool onMouseEvent(LazyEngine::MouseButtonPressedEvent& event);

	};

}