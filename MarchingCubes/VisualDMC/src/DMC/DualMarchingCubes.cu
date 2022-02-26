// ######################################################################### //
// ### Author: Philipp G�rtler ############################################# //
// ### DualMarchingCubes.cu ################################################ //
// ### Implements the DualMarchingCubes Algorithm, that is based heavily ### //
// ### on the implementation of DMC from https://github.com/rogrosso/tmc ### //
// ######################################################################### //

#include "DualMarchingCubes.h"

#include "DeviceHashTables.h"
#include "DeviceVectors.h"
#include "CellIntersection.h"

#include "Halfedges/HalfedgeMesh.h"

#include <thrust/sequence.h>
#include <thrust/extrema.h>

namespace DMC {

	// ##################################################################### //
	// ### CUDA Constants ################################################## //
	// ##################################################################### //

	__constant__ unsigned short c_edgePatterns[256];
	__constant__ char c_trianglePatterns[256 * 16];
	__constant__ char c_triangleAmbiguities[256];
	__constant__ char c_polygonPatterns[256 * 17];


	// ##################################################################### //
	// ### Dual Marching Cubes Kernels ##################################### //
	// ##################################################################### //

	 /**
	  *	performs the dual marching cubes method.
	  */
	__global__ void performDualMarchingCubes(float isoValue, UniformGrid<float> uniformGrid, QuadrilateralDeviceHashTable hashTable, VertexDeviceVector vertices, bool enableSurfaceToVertexIteration) {
		LazyEngine::CUDAInfo info;
		auto threadId = info.getGlobalThreadId();
		if (threadId >= uniformGrid.getCellCount()) return;

		glm::ivec3 index3D = uniformGrid.get3DIndexFromGlobalIndex(threadId);
		glm::ivec3 gridSize = uniformGrid.getDimensions();
		if (index3D.x >= (gridSize.x - 1) || index3D.y >= (gridSize.y - 1) || index3D.z >= (gridSize.z - 1)) {
			// don't handle the last planes in each dimension
			return;
		}

		float scalars[8];
		uniformGrid.evaluateCellCornerValues(scalars, index3D);

		// check which case this is
		unsigned int caseNumber = 0x0;
		for (int i = 0; i < 8; ++i) {
			caseNumber |= ((scalars[i] >= isoValue) ? 0x1 : 0x0) << i;
		}

		// early out if this is a case with nothing to do
		if (caseNumber == 0 || caseNumber == 255) {
			return;
		}

		// perform CellIntersection's sliceP-method
		CellIntersection intersection(isoValue, caseNumber, index3D, uniformGrid, hashTable, vertices, scalars, c_polygonPatterns, c_triangleAmbiguities, MarchingCubes::MC_AMBIGUOUS, enableSurfaceToVertexIteration);
		intersection.sliceP();
	}

	/**
	 *	This kernel could need a lot of optimization!
	 */
	__global__ void mapQuadrilaterals(QuadrilateralDeviceHashTable hashTable, QuadrilateralDeviceVector quadrilaterals) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;

		if (hashTable.isBucketEmpty(threadId)) return;

		glm::ivec4 quad = hashTable.getQuad(threadId);
		if (quad.x < 0 || quad.y < 0 || quad.z < 0 || quad.w < 0) {
			return;
		}
		uint32_t color = hashTable.getColor(threadId);
		quadrilaterals.addQuadrilateral(quad, color);
	}

	// ##################################################################### //
	// ### MarchingCubes ################################################### //
	// ##################################################################### //

	MarchingCubes::MarchingCubes()
	{
		// upload the tables to the CUDA Constants
		checkCudaErrors(cudaMemcpyToSymbol(c_edgePatterns, (void*)s_edgePatterns.data(), s_edgePatterns.size() * sizeof(unsigned short)));
		checkCudaErrors(cudaMemcpyToSymbol(c_trianglePatterns, (void*)s_trianglePatterns.data(), s_trianglePatterns.size() * sizeof(char)));
		checkCudaErrors(cudaMemcpyToSymbol(c_triangleAmbiguities, (void*)s_triangleAmbiguities.data(), s_triangleAmbiguities.size() * sizeof(char)));
		checkCudaErrors(cudaMemcpyToSymbol(c_polygonPatterns, (void*)s_polygonPatterns.data(), s_polygonPatterns.size() * sizeof(char)));
	}

	MarchingCubes::~MarchingCubes() {
		// empty
	}

	// ##################################################################### //
	// ### DualMarchingCubes ############################################### //
	// ##################################################################### //

	static const IndexType MAX_ANYTHING = 20'000'000;

	DualMarchingCubes::DualMarchingCubes()
		: MarchingCubes()
		// Counters
		, m_numVertices(0)
		, m_numQuads(0)
		// DMC Options
		, m_enableSurfaceToVertexIteration(true)
		// CUDA Buffers (for DMC)
		, m_hashTable(nullptr)
		, m_vertices(nullptr)
		, m_quads(nullptr)
		// Post-Process Datastructures
		, m_halfedgeMesh(nullptr)
		, m_manifoldCreator(nullptr)
		, m_programmaticSplittingStep(SplittingStep::Ignore)
		// Post-Process Options
		, m_simplifyP3x3yColor(false)
		, m_simplifyP3x3yOld(false)
		, m_simplifyP3333(false)
		, m_splitNonManifoldHalfedges(true)
		, m_detectNonManifoldVertices(true)
	{
		m_manifoldCreator = std::make_unique<ManifoldCreator>();
	}

	DualMarchingCubes::~DualMarchingCubes() {
		// De-allocate cuda buffers explicitly
		m_hashTable = nullptr;
		m_vertices = nullptr;
		m_quads = nullptr;
	}

	Performance DualMarchingCubes::initializeBuffers() {
		Performance performance = { "DMC Buffer Initialization" };
		initHashtable(performance);
		initQuads(performance);
		initSharedVertexList(performance);
		return performance;
	}

	void DualMarchingCubes::renderImGui() {
		if (ImGui::CollapsingHeader("DMC Options")) {
			ImGui::Checkbox("Enable Vertex-To-Surface-Iteration", &m_enableSurfaceToVertexIteration);
		}
		if (ImGui::CollapsingHeader("DMC Post-Processing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent();
			if (ImGui::CollapsingHeader("Mesh-Simplifications")) {
				ImGui::Indent();
				ImGui::Checkbox("Simplify p3x3y Color-based", &m_simplifyP3x3yColor);
				ImGui::Checkbox("Simplify p3x3y Old", &m_simplifyP3x3yOld);
				ImGui::Checkbox("Simplify p3333", &m_simplifyP3333);
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Non-Manifoldness")) {
				ImGui::Indent();
				ImGui::Checkbox("Split Non-Manifold Halfedges", &m_splitNonManifoldHalfedges);

				if (m_manifoldCreator != nullptr && m_splitNonManifoldHalfedges) {
					m_manifoldCreator->imguiOptions();
				}
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("DMC Debugging Options")) {
				ImGui::Indent();
				if (m_manifoldCreator != nullptr) {
					m_manifoldCreator->imguiDebug();
				}
				ImGui::Checkbox("Detect Non Manifold Vertices", &m_detectNonManifoldVertices);
				ImGui::Unindent();
			}
			ImGui::Unindent();
		}
	}

	int DualMarchingCubes::getNumNonManifoldEdges4() const {
		if (m_halfedgeMesh != nullptr) {
			return m_halfedgeMesh->getNumNonManifoldEdges4();
		}
		return -1;
	}

	int DualMarchingCubes::getNumNonManifoldEdges3() const {
		if (m_halfedgeMesh != nullptr) {
			return m_halfedgeMesh->getNumNonManifoldEdges3();
		}
		return -1;
	}

	int DualMarchingCubes::getNumEdges() const {
		if (m_halfedgeMesh != nullptr) {
			return m_halfedgeMesh->getNumEdges();
		}
		return -1;
	}


	int DualMarchingCubes::getNumNonManifoldVertices() const {
		if (m_halfedgeMesh != nullptr && m_detectNonManifoldVertices) {
			return m_manifoldCreator->getNumNonManifoldVertices();
		}
		return -1;
	}

	int DualMarchingCubes::getNumQuadsWithNonManifoldEdges(int numManifoldEdges) const {
		if (m_halfedgeMesh != nullptr) {
			return m_halfedgeMesh->getNumQuadsWithNonManifoldEdges(numManifoldEdges);
		}
		return -1;
	}

	Performance DualMarchingCubes::run(float isoValue, Mesh& mesh, const UniformGridHost<float>& grid) {
		m_halfedgeMesh = nullptr;

		// Keep track of the computation times
		Performance performances = { "Total Execution Time" };

		LAZYENGINE_WARN("Starting DMC. IsoValue: {0}", isoValue);

		// Get the problem size
		size_t numCells = grid.getCellCount();

		// ################################################################# //
		// ### Compute Quadrilaterals ###################################### //

		{
			Performance initialization = { "Initialization of Buffers" };
			// Step 1: Allocate Hash-Table with 2 * 10^6 entries/ Clear existing hash table
			initHashtable(initialization);

			// Step 2: Allocate Vertex Memory
			initSharedVertexList(initialization);

			// Step 3: Allocate Quadrilaterals
			initQuads(initialization);

			performances.addChild(initialization);
		}

		{
			Performance dmcPerformance = { "DMC" };

			// Step 4: Compute the Iso-surface
			computeIsoSurface(dmcPerformance, grid, isoValue);

			// Step 5: Compute a shared vertex list for a quadrilateral mesh.
			// The indices of the quadrilateral vertices have to be mapped to a global vertex index in the vertex array
			m_numVertices = m_vertices->size();
			if (m_numVertices == 0) {
				LAZYENGINE_ERROR("DMC: No Vertices have been produced!");
				performances.addChild(dmcPerformance);
				return performances;
			}

			// Step 5.1: Map the quadrilateral indices
			mapQuads(dmcPerformance);

			// release the memory of the QuadrilateralHostHashTable
			{
				PerformanceTimer timer("Freeing the QuadHashTable", dmcPerformance);
				m_hashTable.reset(nullptr);
			}

			performances.addChild(dmcPerformance);
		}
		// ### Post Processing starts here ### //
		runPostProcesses(performances, grid, isoValue, mesh);

		// Step 6: Update the mesh's buffers
		if (!m_manifoldCreator->rendersInterestingAreas()) {
			createMesh(performances, mesh, grid);
		}

		// ### Deallocate remaining Buffers ### //
		{
			PerformanceTimer timer("Freeing Vertices", performances);
			m_vertices.reset(nullptr);
		}

		return performances;
	}

	void DualMarchingCubes::initHashtable(Performance& performances) {
		PerformanceTimer timer("Hashtable initialization", performances);
		if (m_hashTable == nullptr) {
			m_hashTable = std::make_unique<QuadrilateralHostHashTable>(MAX_ANYTHING);
		}
		m_hashTable->clear();
	}

	void DualMarchingCubes::initSharedVertexList(Performance& performances) {
		PerformanceTimer timer("Shared Vertex List allocation", performances);
		if (m_vertices == nullptr) {
			m_vertices = std::make_unique<VertexHostVector>(MAX_ANYTHING);
		}
		m_vertices->clear();
	}

	void DualMarchingCubes::initQuads(Performance& performances) {
		PerformanceTimer timer("Quadrilaterals List allocation", performances);
		if (m_quads == nullptr) {
			m_quads = std::make_unique<QuadrilateralHostVector>(MAX_ANYTHING);
		}
		m_quads->clear();
	}

	void DualMarchingCubes::computeIsoSurface(Performance& performances, const UniformGridHost<float>& grid, float isoValue) {
		PerformanceTimer timer("Dual Marching Standalone", performances);
		auto dataView = grid.getDeviceGrid().getDataView();
		performDualMarchingCubes ToCudaArgs(dataView) (isoValue, grid, *m_hashTable, *m_vertices, m_enableSurfaceToVertexIteration);
	}

	void DualMarchingCubes::mapQuads(Performance& performances) {
		PerformanceTimer timer("Quad Mapping", performances);

		auto dataView = m_hashTable->getDataView();
		mapQuadrilaterals ToCudaArgs(dataView) (*m_hashTable, *m_quads);

		m_numQuads = m_quads->size();
	}

	void DualMarchingCubes::createMesh(Performance& performances, Mesh& mesh, const UniformGridHost<float>& grid) {
		PerformanceTimer timer("Mesh Construction", performances);
		updateMesh(mesh, grid);
	}

	void DualMarchingCubes::runPostProcesses(Performance& performances, const UniformGridHost<float>& grid, float isoValue, Mesh& mesh) {

		// ######################################################## //
		// ### Post-Process 1: Generate Halfedge Data Structure ### //
		{
			m_halfedgeMesh = std::make_unique<HalfedgeMesh>();

			auto initPerformance = m_halfedgeMesh->initialize(m_numQuads, m_numVertices, *m_quads);

			m_manifoldCreator->setHalfedgeMesh(m_halfedgeMesh.get());

			performances.addChild(initPerformance);
		}

		// ######################################################## //
		// ### Post-Process 2: Mesh Simplification ################ //
		if (m_simplifyP3x3yColor) {
			auto p = m_halfedgeMesh->simplify3X3Y(*m_vertices, *m_quads);
			performances.addChild(p);
		}

		if (m_simplifyP3x3yOld) {
			auto p = m_halfedgeMesh->simplify3X3YOld(*m_vertices, *m_quads);
			performances.addChild(p);
		}

		if (m_simplifyP3333) {
			auto p = m_halfedgeMesh->simplify3333(*m_vertices, *m_quads);
			performances.addChild(p);
		}

		{
			PerformanceTimer timer("Freeing DMC Quads", performances);
			m_quads.reset(nullptr);
		}

		/*if (m_splitNonManifoldHalfedges) */{
			auto p = splitNonManifoldHalfedges(grid, isoValue, mesh);
			performances.addChild(p);
		}

		if (m_detectNonManifoldVertices) {
			auto p = m_manifoldCreator->detectNonManifoldVertices();
			performances.addChild(p);
		}
	}

	void DualMarchingCubes::updateMesh(Mesh& mesh, const UniformGridHost<float>& grid) {
		size_t numVertices = m_vertices->size() * sizeof(Vertex);
		size_t numIndices = m_halfedgeMesh->getHalfedgeFaces().size() * 6;
		if (numIndices == 0 || numVertices == 0) {
			return;
		}
		mesh.resize(numVertices, numIndices);
		{
			cudaTestMemoryLeakScope();
			LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(mesh.getVertices());
			LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(mesh.getIndices());

			auto vertexDataView = vertexInterop.getMapping();
			auto indexDataView = indexInterop.getMapping();

			m_halfedgeMesh->toVertexAndIndexBuffer(*m_vertices, vertexDataView, indexDataView);
		}

		// update mesh scaling (So that all meshes will use approximately the same world-space)
		glm::vec3 scaling = glm::vec3(10.f);
		glm::vec3 origin = glm::vec3(grid.getOrigin());
		float minOrigin = std::min(origin.x, std::min(origin.y, origin.z));
		scaling /= minOrigin;
		mesh.setScale(scaling);
	}

	Performance DualMarchingCubes::splitNonManifoldHalfedges(const UniformGridHost<float>& grid, float isoValue, Mesh& debugMesh) {
		Performance performances = { "Post-Process: Non-Manifold-Splitting" };
		{
			PerformanceTimer timer("Fix Halfedge-Twin Connections", performances);
			m_manifoldCreator->fixManifoldTwins(*m_vertices, grid, isoValue);
		}
		if (m_splitNonManifoldHalfedges) {
			Performance split = m_manifoldCreator->splitNonManifoldHalfedgesVersion3(*m_vertices, grid, isoValue, debugMesh, m_programmaticSplittingStep);
			performances.addChild(split);
		}
		return performances;
	}
}
