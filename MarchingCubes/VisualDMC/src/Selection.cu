// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### Selection.cu ######################################################## //
// ### Implements Selection.h                                            ### //
// ######################################################################### //

#include "Selection.h"

#include <thrust/extrema.h>

#include "DMC/PostProcesses/ManifoldCreator.h"
#include "DMC/Utils/Intersections.h"

namespace DMC {

	// ##################################################################### //
	// ### Device Code ##################################################### //
	// ##################################################################### //

	/**
	 *	Finds all halfedges that originate at a specific vertex.
	 */
	__global__ void findHalfedgesOfVertex(
		HalfedgeDeviceVector halfedges,
		InfoDeviceVector<HalfedgeInfo> output,
		VertexID origin
	) {
		LazyEngine::CUDAInfo info;
		int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		Halfedge halfedge = halfedges[threadId];
		if (halfedge.getOriginVertexID() == origin) {
			output.addInfo({ threadId, halfedge });
		}
	}

	/**
	 *	Finds all faces that are containing a specific vertex.
	 */
	__global__ void findFacesOfVertex(
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		InfoDeviceVector<FaceInfo> output,
		VertexID vertex
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		Halfedge halfedge = halfedges[threadId];
		if (halfedge.getOriginVertexID() == vertex) {
			HalfedgeID faceOrigin = faces.getFirstHalfedgeID(halfedge.getFace());
			output.addInfo({ halfedge.getFace(), faces.getAttributes(halfedge.getFace()), faceOrigin });
		}
	}

	__device__ inline glm::ivec3 computeIndex3D(int linearIndex, glm::ivec3 dimensions) {
		glm::ivec3 indexOffset;
		indexOffset.x = linearIndex % dimensions.x;
		indexOffset.y = (linearIndex / dimensions.x) % dimensions.y;
		indexOffset.z = linearIndex / (dimensions.x * dimensions.y);
		return indexOffset;
	}

	__device__ inline glm::ivec3 computeLocalIndexOffset(unsigned int threadId, int maxDimension) {
		glm::ivec3 indexOffset;
		indexOffset.x = threadId % maxDimension;
		indexOffset.y = (threadId / maxDimension) % maxDimension;
		indexOffset.z = threadId / (maxDimension * maxDimension);
		return indexOffset;
	}

	__device__ inline glm::ivec3 computeLocalIndex(glm::ivec3 index3D, int maxDimension, unsigned int threadId) {
		// compute the vertex's index3D
		glm::ivec3 indexOffset = computeLocalIndexOffset(threadId, maxDimension);
		glm::ivec3 globalIndex = index3D + indexOffset;

		return globalIndex;
	}

	__device__ inline bool isOutOfBounds(glm::ivec3 index3D, glm::ivec3 dimensions) {
		bool result = false;
		result |= (index3D.x < 0 || index3D.y < 0 || index3D.z < 0);
		result |= (index3D.x >= dimensions.x || index3D.y >= dimensions.y || index3D.z >= dimensions.z);
		return result;
	}

	__device__ inline uint32_t linearizeIndex3D(glm::ivec3 index3D, glm::ivec3 dimensions) {
		return static_cast<uint32_t>((index3D.z * dimensions.x * dimensions.y) + (index3D.y * dimensions.x) + index3D.x);
	}

	__device__ inline uint32_t linearizeIndex3D(glm::ivec3 index3D, glm::ivec3 offset, glm::ivec3 dimensions) {
		glm::ivec3 index = index3D + offset;
		return static_cast<uint32_t>((index.z * dimensions.x * dimensions.y) + (index.y * dimensions.x) + index.x);
	}

	/**
	 *	Generates cell-vertices of a cell
	 */
	__global__ void createCellVertices(LazyEngine::DataView<Vertex> vertices, glm::ivec3 index3D, int radius, UniformGrid<float> grid) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.size()) return;

		glm::ivec3 globalIndex = computeLocalIndex(index3D, radius * 2, threadId);
		globalIndex -= radius - 1;
		glm::ivec3 dimensions = grid.getDimensions();

		bool earlyOut = isOutOfBounds(globalIndex, dimensions);

		// don't leave the grid boundaries
		if (earlyOut) {
			return;
		}

		// finally compute the vertex's position
		Vertex vertex;
		vertex.position = grid.getOrigin() + glm::vec3(globalIndex) * grid.getDeltas();
		vertex.normal = glm::vec3(0.f, 1.f, 0.f);

		// display x-y-z-coordinates with the coloring!
		glm::ivec3 selectionDimensions = glm::ivec3(radius*2);
		glm::ivec3 offset = computeIndex3D(threadId, selectionDimensions);
		glm::vec3 indexColor = glm::vec3(offset) / glm::vec3(selectionDimensions);
		vertex.color = glm::vec4(indexColor, 1.f);

		vertices[threadId] = vertex;
	}

	__global__ void createCellVerticesUsingDimensions(LazyEngine::DataView<Vertex> vertices, glm::ivec3 minIndex, glm::ivec3 maxIndex, UniformGrid<float> grid) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.size()) return;

		// compute this thread's offset from the minIndex
		glm::ivec3 selectionDimensions = (maxIndex - minIndex) + 1;
		glm::ivec3 offset = computeIndex3D(threadId, selectionDimensions);
		// compute the corresponding 3D-index in the original grid
		glm::ivec3 index3D = minIndex + offset;

		glm::ivec3 gridDimensions = grid.getDimensions();
		//printf("%d: selection-dims: [%d,%d,%d], offset: [%d,%d,%d] ==> index3D: [%d,%d,%d], grid: [%d,%d,%d]\n", threadId, VECCC(selectionDimensions), VECCC(offset), VECCC(index3D), VECCC(gridDimensions));
		bool earlyOut = isOutOfBounds(index3D, gridDimensions);
		if (earlyOut) {
			// This vertex would be out of bounds of the original grid. Ignore it.
			return;
		}

		// Compute the vertex's position
		Vertex vertex;
		vertex.position = grid.getOrigin() + glm::vec3(index3D) * grid.getDeltas();
		vertex.normal = glm::vec3(0.f, 1.f, 0.f);
		// display x-y-z-coordinates with the coloring!
		glm::vec3 dimensionsFactors = glm::vec3(offset) / glm::vec3(maxIndex - minIndex);
		vertex.color = glm::vec4(dimensionsFactors, 1.f);

		vertices[threadId] = vertex;
	}

	__device__ void createCellTriangles(LazyEngine::DataView<uint32_t> indices, int index, glm::ivec3 localOffset, glm::ivec3 localGridDimensions) {
		// Quad 1:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 0), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);

		// Quad 2:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 1), localGridDimensions);

		// Quad 3:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 1), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);

		// Quad 4:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 0), localGridDimensions);

		// Quad 5:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 1), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 0, 0), localGridDimensions);

		// Quad 6:
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 0), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);

		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(1, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 1), localGridDimensions);
		indices[index++] = linearizeIndex3D(localOffset + glm::ivec3(0, 1, 0), localGridDimensions);
	}

	__global__ void createCellIndicesUsingDimensions(LazyEngine::DataView<uint32_t> indices, glm::ivec3 minIndex, glm::ivec3 maxIndex, UniformGrid<float> grid) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		unsigned int cellId = threadId;
		if (threadId * (6 * 2 * 3) >= indices.size()) return;

		glm::ivec3 gridDimensions = grid.getDimensions();

		// handle an entire cell
		// compute this thread's offset from the minIndex
		glm::ivec3 selectionDimensions = (maxIndex - minIndex);
		glm::ivec3 offset = computeIndex3D(threadId, selectionDimensions);
		// compute the corresponding 3D-index in the original grid
		glm::ivec3 index3D = minIndex + offset;

		int index = cellId * 6 * 2 * 3;

		glm::ivec3 localIndex = index3D;
		for (int x = 0; x < 2; ++x) {
			for (int y = 0; y < 2; ++y) {
				for (int z = 0; z < 2; ++z) {
					if (isOutOfBounds(localIndex + glm::ivec3(x, y, z), gridDimensions)) {
						for (int i = 0; i < 6 * 3 * 2; ++i) {
							indices[index + i] = 0;
						}
						return;
					}
				}
			}
		}

		createCellTriangles(indices, index, offset, selectionDimensions + 1);
	}

	__global__ void createCellIndices(LazyEngine::DataView<uint32_t> indices, glm::ivec3 index3D, int radius, UniformGrid<float> grid) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		unsigned int cellId = threadId;
		if (threadId * (6 * 2 * 3) >= indices.size()) return;


		glm::ivec3 gridDimensions = grid.getDimensions();

		// handle an entire cell
		glm::ivec3 localOffset = computeLocalIndexOffset(cellId, radius * 2 - 1);
		int cheatedCellId = cellId + localOffset.y + (radius * 2 * 2 - 1) * localOffset.z;
		int index = cellId * 6 * 2 * 3;

		glm::ivec3 localIndex = computeLocalIndex(index3D, radius * 2, cheatedCellId);
		localIndex -= radius - 1;
		for (int x = 0; x < 2; ++x) {
			for (int y = 0; y < 2; ++y) {
				for (int z = 0; z < 2; ++z) {
					if (isOutOfBounds(localIndex + glm::ivec3(x, y, z), gridDimensions)) {
						for (int i = 0; i < 6 * 3 * 2; ++i) {
							indices[index + i] = 0;
						}
						return;
					}
				}
			}
		}

		glm::ivec3 localGridDimensions = glm::ivec3(radius * 2);

		localOffset = computeLocalIndexOffset(cheatedCellId, radius * 2);

		createCellTriangles(indices, index, localOffset, localGridDimensions);
	}

	__global__ void getFaceIndices(FaceID faceId, HalfedgeDeviceVector halfedges, HalfedgeFaceDeviceVector faces, LazyEngine::DataView<uint32_t> output) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.size()) return;
		if (threadId != faceId) return;

		HalfedgeID initialHalfedge = faces.getFirstHalfedgeID(threadId);
		// get the halfedges that make up the quad
		glm::ivec4 quad;
		Halfedge currentHalfedge = halfedges[initialHalfedge];
		for (int i = 0; i < 4; ++i) {
			quad[i] = currentHalfedge.getOriginVertexID();
			currentHalfedge = halfedges[currentHalfedge.getNext()];
		}

		// create index layout (for two triangles)
		output[0] = quad.x;
		output[1] = quad.y;
		output[2] = quad.z;

		output[3] = quad.z;
		output[4] = quad.w;
		output[5] = quad.x;
	}


	// ##################################################################### //
	// ### Host Code ####################################################### //
	// ##################################################################### //

	Selection::Selection(Mesh& mesh, HalfedgeMesh& halfedgeMesh, UniformGridHost<float>& grid)
		: m_mesh(mesh)
		, m_halfedgeMesh(halfedgeMesh)
		, m_grid(grid)
		// Selections
		// Vertex Selection
		, m_selectedVertexID(-1)
		, m_selectedVertex()
		, m_selectedVertexFlags()
		// Halfedge Selection
		, m_selectedHalfedgeID(-1)
		, m_possibleHalfedges()
		// Face Selection
		, m_selectedFaceID(-1)
		, m_selectedFace()
		, m_possibleFaces()
		// Cell Selection
		, m_selectedCellID(static_cast<size_t>(-1))
		, m_selectedCellIndex3D({ -1,-1,-1 })
		, m_selectedGridValues()
		, m_selectedGridDimensions({ -1,-1,-1 })
		// File Output
		, m_fileDialog()
		// Rendering
		, m_cellMesh(nullptr)
		, m_cellWallsMesh(nullptr)
		// Options
		, m_options()
		, m_selectionModeOptions()
		, m_currentSelectionMode("Vertex", SelectionMode::Vertex)
	{
		m_cellMesh = std::make_unique<Mesh>();
		m_cellWallsMesh = std::make_unique<Mesh>();
		setOptions(m_options);
		m_selectionModeOptions["Vertex"] = SelectionMode::Vertex;
		m_selectionModeOptions["Face"] = SelectionMode::Face;

		m_cellMesh->setScale(mesh.getScale());
		m_cellWallsMesh->setScale(mesh.getScale());

		m_fileDialog = ImGui::FileBrowser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CloseOnEsc);
		m_fileDialog.SetTitle("Save Selection");
		m_fileDialog.SetTypeFilters({ ".bin" });
	}

	void Selection::setOptions(const SelectionImGuiOptions& options) {
		m_options = options;
		m_mesh.getVertexMaterial()->setScreenSpaceSize(glm::vec2(m_options.vertexSphereSize));
		m_mesh.getVertexMaterial()->setUseScreenSpaceSize(true);
		switch (m_options.selectionMode) {
		case SelectionMode::Vertex:
			m_currentSelectionMode = { "Vertex", SelectionMode::Vertex };
			break;
		case SelectionMode::Face:
			m_currentSelectionMode = { "Face", SelectionMode::Face };
			break;
		}
	}

	Selection::~Selection() {
		// empty
	}

	// ##################################################################### //
	// ### Selection: Viewport Rendering ################################### //
	// ##################################################################### //

	void Selection::renderHighlights() {
		if (m_selectedFaceID != -1) {
			LazyEngine::RenderCommand::disableDepthTest();
			{
				thrust::device_vector<uint32_t> indices(6);
				getFaceIndices ToCudaArgs(m_halfedgeMesh.getHalfedgeFaces().getDataView()) (m_selectedFaceID, m_halfedgeMesh.getHalfedges(), m_halfedgeMesh.getHalfedgeFaces(), indices);
				std::vector<uint32_t> indexData(6);
				thrust::copy(indices.begin(), indices.end(), indexData.begin());
				m_mesh.setHighlightIndices(indexData);
				m_mesh.renderHighlightSolid();
			}
			LazyEngine::RenderCommand::enableDepthTest();
		}

		if (m_selectedVertexID != -1) {
			LazyEngine::RenderCommand::disableDepthTest();
			m_mesh.getVertexMaterial()->setScreenSpaceSize(glm::vec2(m_options.vertexSphereSize));
			{
				std::vector<uint32_t> indices = { static_cast<uint32_t>(m_selectedVertexID) };
				m_mesh.setHighlightIndices(indices);
				m_mesh.renderHighlightPoints();
			}
			LazyEngine::RenderCommand::enableDepthTest();
		}

		if (m_selectedHalfedgeID != -1) {
			LazyEngine::RenderCommand::disableDepthTest();
			{
				// get the vertex ids of the concerned halfedge
				Halfedge a, b;
				auto halfedges = m_halfedgeMesh.getHalfedges().getDataView();
				thrust::copy(halfedges.begin() + m_selectedHalfedgeID, halfedges.begin() + m_selectedHalfedgeID + 1, &a);
				thrust::copy(halfedges.begin() + m_selectedHalfedge.getNext(), halfedges.begin() + m_selectedHalfedge.getNext() + 1, &b);
				std::vector<uint32_t> indices = { static_cast<uint32_t>(a.getOriginVertexID()), static_cast<uint32_t>(b.getOriginVertexID()) };
				m_mesh.setHighlightIndices(indices);
				m_mesh.renderHighlightLines();
			}
			LazyEngine::RenderCommand::enableDepthTest();
		}

		if (m_selectedCellID != static_cast<size_t>(-1) && m_options.isRenderCells) {
			m_cellMesh->renderWireframeQuads();
		}

#if 0
		if (m_options.displayCellWallIntersections) {
			m_cellWallsMesh->render();
		}
#endif
	}

	// ##################################################################### //
	// ### Selection: ImGui Rendering ###################################### //
	// ##################################################################### //

	void Selection::displayImGuiOptions() {
		ImGui::Begin("Current Selection");
		imGuiSelectionOptions();
		imGuiVertex();
		imGuiHalfedge();
		imGuiFace();
		imGuiCell();
		ImGui::End();

		m_fileDialog.Display();

		if (m_fileDialog.HasSelected()) {
			const std::string selection = m_fileDialog.GetSelected().string();
			m_options.outputFile = selection;

			if (m_selectedCellID != static_cast<size_t>(-1)) {
				glm::ivec3 minCorner = m_selectedCellIndex3D;
				glm::ivec3 maxCorner = m_selectedCellIndex3D;
				if (m_options.useCellRadius) {
					minCorner -= (m_options.cellRadius - 1);
					maxCorner += m_options.cellRadius;
				}
				else {
					minCorner = m_selectedCellIndex3D - m_options.selectionMin;
					maxCorner = m_selectedCellIndex3D + m_options.selectionMax;
				}
				m_grid.writeSelectionToFile(m_options.outputFile, minCorner, maxCorner);
			}

			m_fileDialog.ClearSelected();
		}
	}

	void Selection::displayImGuiMenu() {
		if (m_options.isRenderCells && m_selectedCellID != static_cast<size_t>(-1)) {
			if (ImGui::MenuItem("Save selected Grid as .bin")) {
				m_fileDialog.Open();
			}
		}
	}


	static inline void pushHighlightButton() {
		ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.36f, 0.8f, 0.8f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.36f, 0.9f, 0.9f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.36f, 1.f, 1.f));
	}

	static inline void popHighlightButton() {
		ImGui::PopStyleColor(3);
	}

	void Selection::renderFileSelectionForSaving() {
		ImGui::PushID(12345678);
		ImGui::InputText("", const_cast<char*>(m_options.outputFile.c_str()), m_options.outputFile.size());
		ImGui::PopID();
		ImGui::SameLine();
		if (ImGui::Button("Browse")) {
			m_fileDialog.Open();
		}
	}

	void Selection::imGuiSelectionOptions() {
		if (ImGui::CollapsingHeader("Selection Options")) {
			ImGui::Indent();
			
			renderCombo<SelectionMode>("SelectionMode", m_selectionModeOptions, m_currentSelectionMode, [&](const std::pair<std::string, SelectionMode>& newSelection) {
				m_currentSelectionMode = newSelection;
				m_options.selectionMode = newSelection.second;
			});

			ImGui::Unindent();
		}
	}

	void Selection::imGuiVertex() {
		if (ImGui::CollapsingHeader("Vertices")) {
			ImGui::Indent();

			if (ImGui::CollapsingHeader("Vertex Rendering Options")) {
				ImGui::Indent();
				auto& material = m_mesh.getVertexMaterial();
				if (ImGui::SliderFloat("Billboard Size", &m_options.vertexSphereSize, 0.f, 1.f, "%.2f", 3.f)) {
					material->setScreenSpaceSize(glm::vec2(m_options.vertexSphereSize));
				}
				ImGui::Unindent();
			}

			if (m_selectedVertexID != -1) {
				if (ImGui::Button("Focus Selected Vertex")) {
					glm::vec3 worldPosition = m_mesh.modelToWorld(m_selectedVertex.position);
					LAZYENGINE_INFO("Selected Vertex: [{0},{1},{2}], WorldPosition: [{3},{4},{5}]", VECCC(m_selectedVertex.position), VECCC(worldPosition));
					m_updateCameras(worldPosition);
				}
				if (ImGui::Button("Reset Camera Focus")) {
					m_updateCameras(glm::vec3(0.f,0.f,0.f));
				}
			}
			
			if (ImGui::CollapsingHeader("Manual Vertex Selection")) {
				ImGui::Indent();
				static int manualVertex = 0;
				ImGui::InputInt("VertexID", &manualVertex);
				if (ImGui::Button("Select Manual Vertex")) {
					if (manualVertex > -1 && manualVertex < m_halfedgeMesh.getHalfedgeVertices().getFlagsDataView().size()) {
						m_selectedVertexID = manualVertex;
						extractVertexData(m_selectedVertexID);
					}
				}
				ImGui::Unindent();
			}

			ImGui::Text("Position: [%.4f, %.4f, %.4f]", m_selectedVertex.position.x, m_selectedVertex.position.y, m_selectedVertex.position.z);
			ImGui::Text("Normal: [%.2f, %.2f, %.2f]", m_selectedVertex.normal.x, m_selectedVertex.normal.y, m_selectedVertex.normal.z);
			ImGui::Text("Color: [%.2f, %.2f, %.2f, %.2f]", m_selectedVertex.color.r, m_selectedVertex.color.g, m_selectedVertex.color.b, m_selectedVertex.color.a);
			ImGui::Text("ID: %d", m_selectedVertexID);

			if(ImGui::CollapsingHeader("Vertex-Flags", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::Indent();
				bool isNonManifold = m_selectedVertexFlags.isNonManifold();
				bool isBoundary = m_selectedVertexFlags.isBoundary();
				bool isWeird = m_selectedVertexFlags.isWeird();
				bool hasMultipleFans = m_selectedVertexFlags.hasMultipleFans();
				ImGui::Checkbox("isNonManifold", &isNonManifold);
				ImGui::Checkbox("isBoundary", &isBoundary);
				ImGui::Checkbox("isWeird", &isWeird);
				ImGui::Checkbox("hasMultipleFans", &hasMultipleFans);
				ImGui::Unindent();
			}

			if (ImGui::Button("Select Random Non-Manifold Edge-Vertex")) {
				selectRandomNonManifoldMarkedVertex();
			}
			if (ImGui::Button("Select Random Non-Manifold Single Vertex")) {
				selectRandomNonManifoldMarkedSingleVertex();
			}

			ImGui::Unindent();
		}
	}

	void Selection::imGuiHalfedge() {
		if (ImGui::CollapsingHeader("Halfedges")) {
			ImGui::Indent();

			if (ImGui::CollapsingHeader("Halfedge Rendering Options")) {
				ImGui::Indent();
				if (ImGui::SliderFloat("Halfedge Width", &m_options.halfedgeWidth, 0.f, 2.f, "%.3f", 3.f)) {
					auto& material = m_mesh.getLineMaterial();
					material->setLineWidth(m_options.halfedgeWidth);
				}
				ImGui::Unindent();
			}

			int index = 0;
			if (ImGui::Button("Unselect Halfedge")) {
				m_selectedHalfedgeID = -1;
				m_selectedHalfedge = Halfedge();
			}
			if (ImGui::CollapsingHeader("Manual Halfedge-Selection")){
				ImGui::Indent();
				static int manualHalfedge = 0;
				ImGui::InputInt("HalfedgeID", &manualHalfedge);
				if (ImGui::Button("Select Manual Halfedge")) {
					if (-1 < manualHalfedge && manualHalfedge < m_halfedgeMesh.getHalfedges().size()) {
						m_selectedHalfedgeID = manualHalfedge;
						updateInfoForSelectedHalfedgeID();
					}
				}
				ImGui::Unindent();
			}
			ImGui::Text("Selected Halfedge: %d", m_selectedHalfedgeID);
			std::stringstream ss;
			ss << "Origin Vertex " << m_selectedHalfedge.getOriginVertexID();
			if (ImGui::Button(ss.str().c_str())) {
				m_selectedVertexID = m_selectedHalfedge.getOriginVertexID();
				extractVertexData(m_selectedVertexID);

				auto gridStats = m_grid.getGridStats();
				m_selectedCellIndex3D = gridStats.getCellIndex(m_selectedVertex.position);
				m_selectedCellID = gridStats.getGlobalIndex(m_selectedCellIndex3D);
			}
			{
				std::stringstream ss;
				ss << "Face ID: " << m_selectedHalfedge.getFace();
				ImGui::PushID(80085);
				if (ImGui::Button(ss.str().c_str())) {
					m_selectedFaceID = m_selectedHalfedge.getFace();
					updateInfoForSelectedHalfedgeID();
					updateInfoForSelectedFaceID();
				}
				ImGui::PopID();
			}
			{
				ImGui::SameLine();
				std::stringstream ss;
				ss << "Next ID: " << m_selectedHalfedge.getNext();
				ImGui::PushID(80086);
				if (ImGui::Button(ss.str().c_str())) {
					m_selectedHalfedgeID = m_selectedHalfedge.getNext();
					updateInfoForSelectedHalfedgeID();
				}
				ImGui::PopID();
			}
			{
				ImGui::SameLine();
				std::stringstream ss;
				ss << "Twin ID: " << m_selectedHalfedge.getTwin();
				ImGui::PushID(80087);
				if (ImGui::Button(ss.str().c_str())) {
					m_selectedHalfedgeID = m_selectedHalfedge.getTwin();
					updateInfoForSelectedHalfedgeID();
				}
				ImGui::PopID();
			}

			ImGui::Separator();

			for (auto& info : m_possibleHalfedges) {
				if (index & 0x1) ImGui::SameLine();
				std::stringstream ss;
				ss << "Halfedge-ID: " << info.id;

				bool useHighlightColors = info.id == m_selectedHalfedgeID;
				if (useHighlightColors) {
					pushHighlightButton();
				}

				if (ImGui::Button(ss.str().c_str())) {
					m_selectedHalfedgeID = info.id;
					m_selectedHalfedge = info.halfedge;
				}

				if (useHighlightColors) {
					popHighlightButton();
				}

				++index;
			}
			ImGui::Unindent();
		}
	}

	void Selection::imGuiFace() {
		if (ImGui::CollapsingHeader("Faces")) {
			ImGui::Indent();
			if (m_selectedFaceID != -1) {
				ImGui::Text("Selected Face: %d", m_selectedFaceID);
				ImGui::Text("Origin Halfedge: %d", m_selectedFace.initialHalfedge);
				bool isNonManifold = m_selectedFace.attributes.isNonManifold();
				ImGui::Checkbox("non-Manifold", &isNonManifold);
			}
			int index = 0;
			for (auto& info : m_possibleFaces) {
				if (index & 0x1) ImGui::SameLine();
				std::stringstream ss;
				ss << "Face-ID: " << info.id;

				bool isSelectedFace = info.id == m_selectedFaceID;
				if (isSelectedFace) {
					pushHighlightButton();
				}

				if (ImGui::Button(ss.str().c_str())) {
					m_selectedFaceID = info.id;
					m_selectedFace = info;
				}

				if (isSelectedFace) {
					popHighlightButton();
				}

				++index;
			}
			ImGui::Unindent();
		}
	}

	void Selection::imGuiCell() {
		if (ImGui::CollapsingHeader("Cells")) {
			ImGui::Indent();

			if (ImGui::CollapsingHeader("Cell Rendering Options")) {
				ImGui::Indent();
				ImGui::Checkbox("Render Cells", &m_options.isRenderCells);
				ImGui::Checkbox("Use Cell Radius", &m_options.useCellRadius);
				if (m_options.useCellRadius) {
					if (ImGui::SliderInt("Cell-Radius", &m_options.cellRadius, 1, 25)) {
						createGridWireframe();
					}
				}
				else {
					if (ImGui::SliderInt3("Cells Min Index", glm::value_ptr(m_options.selectionMin), 0, 25)) {
						createGridWireframeUsingDimensions();
					}
					if (ImGui::SliderInt3("Cells Max Index", glm::value_ptr(m_options.selectionMax), 0, 25)) {
						createGridWireframeUsingDimensions();
					}
				}
				ImGui::Unindent();
			}

			ImGui::Text("Cell-Index: %d", m_selectedCellID);
			ImGui::Text("Cell-Index 3D: [%d, %d, %d]", m_selectedCellIndex3D.x, m_selectedCellIndex3D.y, m_selectedCellIndex3D.z);

			{
				ImGui::Text("Selection Dimensions: [%d, %d, %d]", m_selectedGridDimensions.x, m_selectedGridDimensions.y, m_selectedGridDimensions.z);
				for (int z = 0; z < m_selectedGridValues.size(); ++z) {
					const auto& currentPlane = m_selectedGridValues[z];
					for (int y = 0; y < currentPlane.size(); ++y) {
						const auto& currentLine = currentPlane[y];
						for (int x = 0; x < currentLine.size(); ++x) {
							ImGui::Text("%.2f ", currentLine[x]);
							if (x < currentLine.size() - 1) {
								ImGui::SameLine();
							}
						}
					}
					ImGui::Separator();
				}
			}



			ImGui::Unindent();
		}
	}

	// ##################################################################### //
	// ### Selection: Viewport Selection ################################### //
	// ##################################################################### //

	void Selection::shootCameraRay(const LazyEngine::CameraRay& ray) {
		switch (m_options.selectionMode) {
		case SelectionMode::Vertex:
			selectVertex(ray);
			break;
		case SelectionMode::Face:
			selectFace(ray);
			break;
		}
	}

	struct IsFlagNonManifoldSet {
		HostDevice bool operator()(const Flags& flags) {
			return flags.isNonManifold();
		}
	};

	void Selection::selectRandomNonManifoldMarkedVertex() {
		auto vertexDataView = m_halfedgeMesh.getHalfedgeVertices().getFlagsDataView();
		auto iter = thrust::find_if(vertexDataView.begin(), vertexDataView.end(), IsFlagNonManifoldSet());
		VertexID index = thrust::distance(vertexDataView.begin(), iter);
		if (iter == vertexDataView.end()) {
			index = -1;
		}
		extractVertexData(index);
	}

	struct IsSingleNonManifoldVertex {
		HostDevice bool operator()(const Flags& flags) {
			return !flags.isNonManifold() && flags.hasMultipleFans();
		}
	};

	void Selection::selectRandomNonManifoldMarkedSingleVertex() {
		auto vertexDataView = m_halfedgeMesh.getHalfedgeVertices().getFlagsDataView();
		auto iter = thrust::find_if(vertexDataView.begin(), vertexDataView.end(), IsSingleNonManifoldVertex());
		VertexID index = thrust::distance(vertexDataView.begin(), iter);
		if (iter == vertexDataView.end()) {
			LAZYENGINE_WARN("No single non-manifold vertex found!");
			index = -1;
		}
		extractVertexData(index);
	}

	void Selection::extractVertexData(VertexID index) {
		m_selectedVertexID = index;
		if (m_selectedVertexID == -1 || m_selectedVertexID >= m_halfedgeMesh.getHalfedgeVertices().getFlagsDataView().size()) return;

		// extract Vertex Flags
		auto flagsDataView = m_halfedgeMesh.getHalfedgeVertices().getFlagsDataView();
		thrust::copy(flagsDataView.begin() + m_selectedVertexID, flagsDataView.begin() + m_selectedVertexID + 1, &m_selectedVertexFlags);

		// extract Vertex Position, Normal, etc.
		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(m_mesh.getVertices());
		auto dataView = vertexInterop.getMapping();
		thrust::copy(dataView.begin() + m_selectedVertexID, dataView.begin() + m_selectedVertexID + 1, &m_selectedVertex);

		// update the highlight-index for rendering
		std::vector<uint32_t> indices = { static_cast<uint32_t>(m_selectedVertexID) };
		m_mesh.setHighlightIndices(indices);

		updateFaceInfoForSelectedVertex();
	}

	void Selection::selectVertex(const LazyEngine::CameraRay& ray) {

		const float radius = 1.f;

		// map the vertex buffer into cuda-memory-space
		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(m_mesh.getVertices());
		auto dataView = vertexInterop.getMapping();

		// create a device vector that will contain every intersection result
		MonitoredThrustBuffer<CollisionRayData> intersections(dataView.size(), "Intersections");

		// intersect the ray with each vertex
		const auto transform = m_mesh.getModelToWorld();
		Intersector::intersectRayWithScene(ray, dataView, intersections, transform, radius);

		// find the closest vertex to the ray and copy it to Host memory
		CollisionRayData minElement = *thrust::min_element(intersections.begin(), intersections.end(), CollisionRayData());

		if (minElement.distance != minElement.distance || minElement.distance > 0.f) {
			m_selectedVertexID = -1;
			m_selectedCellID = static_cast<size_t>(-1);
			m_selectedVertex.position = glm::vec3(0.f);
			m_selectedVertex.normal = glm::vec3(0.f);
			m_selectedVertex.color = glm::vec4(0.f);

			m_mesh.setHighlightIndices(std::vector<uint32_t>());
		}
		else {
			// This is a valid point, extract data concerning it
			extractVertexData(minElement.vertexID);


			auto gridStats = m_grid.getGridStats();
			m_selectedCellIndex3D = gridStats.getCellIndex(m_selectedVertex.position);
			m_selectedCellID = gridStats.getGlobalIndex(m_selectedCellIndex3D);

			if (m_options.useCellRadius) {
				createGridWireframe();
			}
			else {
				createGridWireframeUsingDimensions();
			}

			// get all halfedges originating in the selected vertex
			updateHalfedgeInfoForSelectedVertex();
			// get the faces that are associated with the selected vertex
			updateFaceInfoForSelectedVertex();

		}
	}

	void Selection::selectFace(const LazyEngine::CameraRay& ray) {
		// use the halfedge-mesh's face-buffer

		// map the vertex buffer into cuda-memory-space
		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(m_mesh.getVertices());
		auto dataView = vertexInterop.getMapping();

		// create a device vector that will contain every intersection result
		MonitoredThrustBuffer<CollisionRayQuad> intersections(m_halfedgeMesh.getHalfedgeFaces().size(), "Intersections-Quads");

		// intersect the ray with each vertex
		const auto transform = m_mesh.getModelToWorld();
		Intersector::intersectRayWithQuads(ray, m_halfedgeMesh.getHalfedgeFaces(), m_halfedgeMesh.getHalfedges(), dataView, intersections, transform);

		// find the closest quad to the ray and copy it to Host memory
		CollisionRayQuad minElement = *thrust::min_element(intersections.begin(), intersections.end(), CollisionRayQuad());

		LAZYENGINE_INFO("MinElement: FaceID={0}, distanceFromOrigin={1}", minElement.faceID, minElement.distanceFromOrigin);
		if (minElement.distanceFromOrigin != minElement.distanceFromOrigin || minElement.distanceFromOrigin < 0.f) {
			m_selectedFaceID = -1;
			m_selectedFace = FaceInfo();
		}
		else {
			// This is a valid face, extract data concerning it
			m_selectedFaceID = minElement.faceID;
			updateInfoForSelectedFaceID();
		}
	}

	// ##################################################################### //
	// ### Selection: Helper Methods ####################################### //
	// ##################################################################### //

	void Selection::updateFaceInfoForSelectedVertex() {

		InfoHostVector<FaceInfo> faceInfo(100);
		auto dataView = m_halfedgeMesh.getHalfedges().getDataView();
		findFacesOfVertex ToCudaArgs(dataView) (m_halfedgeMesh.getHalfedges(), m_halfedgeMesh.getHalfedgeFaces(), faceInfo, m_selectedVertexID);

		m_possibleFaces.resize(faceInfo.size());
		if (faceInfo.size() > 0) {
			auto faceInfoDataView = faceInfo.getDataView();
			thrust::copy(faceInfoDataView.begin(), faceInfoDataView.begin() + faceInfo.size(), m_possibleFaces.begin());
		}
	}

	void Selection::updateHalfedgeInfoForSelectedVertex() {
		InfoHostVector<HalfedgeInfo> halfedgeInfo(100);
		auto dataView = m_halfedgeMesh.getHalfedges().getDataView();
		findHalfedgesOfVertex ToCudaArgs(dataView) (m_halfedgeMesh.getHalfedges(), halfedgeInfo, m_selectedVertexID);

		m_possibleHalfedges.resize(halfedgeInfo.size());
		auto halfedgeInfoDataView = halfedgeInfo.getDataView();
		thrust::copy(halfedgeInfoDataView.begin(), halfedgeInfoDataView.begin() + halfedgeInfo.size(), m_possibleHalfedges.begin());

	}

	void Selection::updateInfoForSelectedFaceID() {
		m_selectedFace.id = m_selectedFaceID;
		if (m_selectedFaceID == -1) {
			return;
		}
		auto faceAttributesDataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
		auto faceFirstHalfedgeDataView = m_halfedgeMesh.getHalfedgeFaces().getFirstHalfedgeIDDataView();
		thrust::copy(faceAttributesDataView.begin() + m_selectedFaceID, faceAttributesDataView.begin() + m_selectedFaceID + 1, &m_selectedFace.attributes);
		thrust::copy(faceFirstHalfedgeDataView.begin() + m_selectedFaceID, faceFirstHalfedgeDataView.begin() + m_selectedFaceID + 1, &m_selectedFace.initialHalfedge);
	}

	void Selection::updateInfoForSelectedHalfedgeID() {
		if (m_selectedHalfedgeID == -1) {
			m_selectedHalfedge = Halfedge();
			return;
		}
		// fetch the halfedge pointed to by the selected halfedge id
		m_selectedHalfedge = getHalfedgeOfID(m_selectedHalfedgeID);

		InfoHostVector<HalfedgeInfo> halfedgeInfo(100);
		auto dataView = m_halfedgeMesh.getHalfedges().getDataView();
		findHalfedgesOfVertex ToCudaArgs(dataView) (m_halfedgeMesh.getHalfedges(), halfedgeInfo, m_selectedHalfedge.getOriginVertexID());

		m_possibleHalfedges.resize(halfedgeInfo.size());
		auto halfedgeInfoDataView = halfedgeInfo.getDataView();
		thrust::copy(halfedgeInfoDataView.begin(), halfedgeInfoDataView.begin() + halfedgeInfo.size(), m_possibleHalfedges.begin());
	}

	Halfedge Selection::getHalfedgeOfID(HalfedgeID id) {
		Halfedge result;
		auto& halfedges = m_halfedgeMesh.getHalfedges();
		if (id > -1 && id < halfedges.size()) {
			thrust::copy(halfedges.getDataView().begin() + id, halfedges.getDataView().begin() + id + 1, &result);
		}
		return result;
	}

	void Selection::createGridWireframeUsingDimensions() {
		if (m_selectedCellID == static_cast<size_t>(-1)) return;

		// compute the min-corner index
		glm::ivec3 minCorner = m_selectedCellIndex3D;
		minCorner -= m_options.selectionMin;

		// compute the max-corner index
		glm::ivec3 maxCorner = m_selectedCellIndex3D + 1;
		maxCorner += m_options.selectionMax;

		glm::ivec3 cellSelectionSize = maxCorner - minCorner;

		// compute the amount of vertices needed
		int numVertices = cellSelectionSize.x + 1;
		numVertices *= cellSelectionSize.y + 1;
		numVertices *= cellSelectionSize.z + 1;

		// 6 quads per cell, aka 6 * two triangles
		int numIndices = cellSelectionSize.x * cellSelectionSize.y * cellSelectionSize.z;
		numIndices *= 6 * 2 * 3;

		LAZYENGINE_INFO("Cells: [{0},{1},{2}], numVertices: {3}, numIndices: {4}", VECCC(cellSelectionSize), numVertices, numIndices);

		m_cellMesh->resize(numVertices * sizeof(Vertex), numIndices);

		{
			auto& vertexBuffer = m_cellMesh->getVertices();
			LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(vertexBuffer);
			auto dataView = vertexInterop.getMapping();

			createCellVerticesUsingDimensions ToCudaArgs(dataView) (dataView, minCorner, maxCorner, m_grid);
		}
		{
			auto& indexBuffer = m_cellMesh->getIndices();
			LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(indexBuffer);
			auto dataView = indexInterop.getMapping();

			createCellIndicesUsingDimensions ToCudaArgs(dataView) (dataView, minCorner, maxCorner, m_grid);
		}

		m_selectedGridValues.clear();
		m_selectedGridDimensions = m_grid.extractSubData(m_selectedGridValues, minCorner, maxCorner);

	}

	void Selection::createGridWireframe() {
		if (m_selectedCellID == static_cast<size_t>(-1)) return;

		int numVertices = 2 * m_options.cellRadius;
		numVertices = numVertices * numVertices * numVertices;

		// 6 quads per cell
		int numIndices = 2 * m_options.cellRadius - 1;
		numIndices = numIndices * numIndices * numIndices;
		numIndices *= 6 * 2 * 3;

		m_cellMesh->resize(numVertices * sizeof(Vertex), numIndices);

		{
			auto& vertexBuffer = m_cellMesh->getVertices();
			LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(vertexBuffer);
			auto dataView = vertexInterop.getMapping();

			createCellVertices ToCudaArgs(dataView) (dataView, m_selectedCellIndex3D, m_options.cellRadius, m_grid);
		}
		{
			auto& indexBuffer = m_cellMesh->getIndices();
			LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(indexBuffer);
			auto dataView = indexInterop.getMapping();

			createCellIndices ToCudaArgs(dataView) (dataView, m_selectedCellIndex3D, m_options.cellRadius, m_grid);
		}

		glm::ivec3 minCorner = m_selectedCellIndex3D - (m_options.cellRadius - 1);
		glm::ivec3 maxCorner = m_selectedCellIndex3D + m_options.cellRadius;
		m_selectedGridValues.clear();
		m_selectedGridDimensions = m_grid.extractSubData(m_selectedGridValues, minCorner, maxCorner);
	}


}