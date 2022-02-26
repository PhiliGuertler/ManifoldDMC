#pragma once

// ######################################################################### //
// ### Author: Philipp Gürtler ############################################# //
// ### Selection.h ######################################################### //
// ### Defines a class that handles viewport selections. Using the mouse ### //
// ######################################################################### //

#include <LazyEngine/LazyEngine.h>
#include <imgui.h>
#include <imgui-filebrowser/imfilebrowser.h>

#include "DMC/Mesh.h"
#include "DMC/Halfedges/HalfedgeMesh.h"
#include "DMC/UniformGrid.h"

namespace DMC {

	enum class SelectionMode {
		Vertex,
		Face,
	};

	/**
	 *	contains all ImGui-User-editable values of Selection
	 */
	struct SelectionImGuiOptions {
		// Selection
		SelectionMode selectionMode = SelectionMode::Vertex;
		// Vertex-Selection
		float vertexSphereSize = 0.03f; // In screen-space coordinates
		// Halfedge-Selection
		float halfedgeWidth = 0.3f; // In world-space coordinates
		// Cells
		bool useCellRadius = false;
		int cellRadius = 1;
		glm::ivec3 selectionMin = glm::ivec3(0, 0, 0);
		glm::ivec3 selectionMax = glm::ivec3(0, 0, 0);
		bool isRenderCells = false;
		// Asymptotic Decider
		bool displayCellWallIntersections = false;
		// File Output
		std::string outputFile = "./output/testOutput.bin";
	};

	/**
	 *	Takes care of selecting vertices, (half)edges and faces
	 */
	class Selection {
	public:
		Selection(Mesh& mesh, HalfedgeMesh& halfedgeMesh, UniformGridHost<float>& grid);
		virtual ~Selection();

		inline void setSelectionMode(const SelectionMode& mode) {
			m_options.selectionMode = mode;
		}

		inline void setCellRadius(int value) {
			m_options.cellRadius = value;
		}

		inline void setIsRenderCells(bool value) {
			m_options.isRenderCells = value;
		}

		void setOptions(const SelectionImGuiOptions& options);

		inline SelectionImGuiOptions getOptions() const {
			return m_options;
		}

		inline void setOnFocusCamera(const std::function<void(const glm::vec3&)>& callback) {
			m_updateCameras = callback;
		}

		inline void setGetIsoValue(const std::function<float()>& callback) {
			m_getIsoValue = callback;
		}

		/**
		 *	Renders ImGui options
		 */
		void displayImGuiOptions();

		void displayImGuiMenu();

		/**
		 *	Renders Highlights in the 3D-Viewport
		 */
		void renderHighlights();

		/**
		 *	Performs a selection by shooting a camera-ray into the scene
		 */
		void shootCameraRay(const LazyEngine::CameraRay& ray);

		/**
		 *	Selects the first vertex that is marked as non-manifold that can be found.
		 */
		void selectRandomNonManifoldMarkedVertex();

		void selectRandomNonManifoldMarkedSingleVertex();

	protected:
		// Selection Methods //
		void selectVertex(const LazyEngine::CameraRay& ray);
		void selectFace(const LazyEngine::CameraRay& ray);

		// Imgui Subsections //
		void imGuiVertex();
		void imGuiHalfedge();
		void imGuiFace();
		void imGuiCell();
		void imGuiSelectionOptions();

		void renderFileSelectionForSaving();

		void updateFaceInfoForSelectedVertex();
		void updateHalfedgeInfoForSelectedVertex();
		
		void updateInfoForSelectedHalfedgeID();
		void updateInfoForSelectedFaceID();

		Halfedge getHalfedgeOfID(HalfedgeID id);

		void extractVertexData(VertexID index);

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

		/**
		 *	Creates a Grid mesh around the currently selected cell.
		 *	@param radius: The amount of neighboring cells that should be added.
		 *		example:
		 *		1: only the cell itself will be used
		 *		2: the cell and all 26 of its direct neighbors will be used
		 *		...
		 */
		void createGridWireframe();

		void createGridWireframeUsingDimensions();
	protected:
		// The Mesh on which the selection occurs
		Mesh& m_mesh;
		// The HalfedgeMesh of m_mesh
		HalfedgeMesh& m_halfedgeMesh;
		// The Uniform Grid
		UniformGridHost<float>& m_grid;

		// Callbacks
		std::function<void(const glm::vec3&)> m_updateCameras;
		std::function<const glm::mat4& ()> m_getWorldToScreen;
		std::function<float()> m_getIsoValue;

		// ------------------ //
		// --- Selections --- //
		// Vertex
		VertexID m_selectedVertexID;
		Vertex m_selectedVertex;
		Flags m_selectedVertexFlags;

		// Halfedge
		HalfedgeID m_selectedHalfedgeID;
		Halfedge m_selectedHalfedge;
		std::vector<HalfedgeInfo> m_possibleHalfedges;

		// Face
		FaceID m_selectedFaceID;
		FaceInfo m_selectedFace;
		LazyEngine::Ref<LazyEngine::IndexBuffer> m_selectedFaceIndexBuffer;
		std::vector<FaceInfo> m_possibleFaces;

		// Cell
		size_t m_selectedCellID;
		glm::ivec3 m_selectedCellIndex3D;
		std::vector<std::vector<std::vector<float>>> m_selectedGridValues;
		glm::ivec3 m_selectedGridDimensions;

		// ------------------- //
		// --- File-Output --- //
		ImGui::FileBrowser m_fileDialog;

		// ----------------- //
		// --- Rendering --- //
		std::unique_ptr<Mesh> m_cellMesh;
		std::unique_ptr<Mesh> m_cellWallsMesh;

		// --------------- //
		// --- Options --- //
		SelectionImGuiOptions m_options;
		//
		std::map<std::string, SelectionMode> m_selectionModeOptions;
		std::pair<std::string, SelectionMode> m_currentSelectionMode;
	};

}