#include "ManifoldSplitter.h"

#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include "../Utils/Interpolation.h"
#include "../Utils/Intersections.inl"
#include "../Utils/AsymptoticDecider.h"
#include "../CellIntersection.h"

namespace DMC {

	// ##################################################################### //
	// ### MergeList ####################################################### //
	// ##################################################################### //

	DeviceMergeList::DeviceMergeList(const LazyEngine::DataView<glm::ivec2>& mergeVertexIDs, const LazyEngine::DataView<int>& numMergeVertices)
		: m_mergeElementIDs(mergeVertexIDs)
		, m_numMergeElements(numMergeVertices)
	{
		// empty
	}

	__device__ MergePartners DeviceMergeList::getAllPartnersOf(int vertexID) {
		MergePartners result;
		result.numPartners = getNumPartners(vertexID);
		if (result.numPartners <= 1) {
			// This element doesn't have any merge-partners.
			return result;
		}

		// get the element's partners
		for (int i = 0; i < result.numPartners - 1; ++i) {
			result.partners[i] = getMergeElement(vertexID, i);
		}

		// check the partner's partners.
		for (int i = 0; i < result.numPartners-1; ++i) {
			// first, get the partner itself
			int partner = result.partners[i];
			// get the amount of partners of the current partner
			int numPartnersPartners = getNumPartners(partner);
			// Now check for all of the partner's partners if they are contained in the result.
			for (int k = 0; k < numPartnersPartners-1; ++k) {
				// get the partner's partner
				int partnersPartner = getMergeElement(partner, k);
				// check if it is contained in the result
				bool isContained = vertexID == partnersPartner;
				for (int z = 0; z < result.numPartners-1; ++z) {
					isContained |= result.partners[z] == partnersPartner;
				}
				// If the element is not contained, add it to the resulting partners
				if (!isContained) {
					result.partners[result.numPartners - 1] = partnersPartner;
					++(result.numPartners);
				}
			}
		}

		return result;
	}


	HostMergeList::HostMergeList(size_t numElements)
		: d_mergeElementIDs(numElements)
		, d_numMergeElements(numElements)
	{
		initialize();
	}

	void HostMergeList::initialize() {
		thrust::fill(d_mergeElementIDs.begin(), d_mergeElementIDs.end(), glm::ivec2(INVALID, INVALID));
		// each vertex should be merged with itself, obviously
		thrust::fill(d_numMergeElements.begin(), d_numMergeElements.end(), 1);
		// create a device variant
		m_deviceList = std::unique_ptr<DeviceMergeList>(new DeviceMergeList(d_mergeElementIDs, d_numMergeElements));
	}

	// ##################################################################### //
	// ### ManifoldSplitter ################################################ //
	// ##################################################################### //

	SplittingStep ManifoldSplitter::s_debugStep = SplittingStep::All;
	bool ManifoldSplitter::s_renderInterestingAreas = false;
	bool ManifoldSplitter::s_copyDataBack = true;

	ManifoldSplitter::ManifoldSplitter(const UniformGridHost<float>& grid, HalfedgeMesh& halfedgeMesh, VertexHostVector& vertices)
		: m_grid(grid)
		, m_halfedgeMesh(halfedgeMesh)
		, m_vertices(vertices)
	{
		// empty
	}

	Performance ManifoldSplitter::run(float isoValue, Mesh& debugMesh) {
		Performance totalPerformance = { "ManifoldSplitter::run" };
		int i = 0;

		std::unique_ptr<WorkingCopyHalfedge> tmpHalfedges;
		std::unique_ptr<WorkingCopyVertex> tmpVertices;
		std::unique_ptr<WorkingCopyFace> tmpFaces;

		// Subdivisions might cause new non-manifold edges, that are automatically twinned-up correctly.
		// Solve them iteratively
		while (true) {
			std::stringstream ss;
			ss << "Iteration " << i;
			Performance performances = { ss.str() };

#ifdef LAZYENGINE_DEBUG
			LAZYENGINE_INFO("Iteration {0}", i);
#endif

			// find out how many faces have to be subdivided
			size_t numSubdivisions;
			{
				PerformanceTimer timer("Count Non Manifold Faces", performances);
				numSubdivisions = countNonManifoldFaces();
			}

			// If no subdivisions have to be made, return
			if (numSubdivisions == 0) {
				break;
			}

			{
				PerformanceTimer timer("Tmp-Buffer initialization", performances);
				// each face consists of 4 existing and up to 16 new halfedges when subdivided
				{
					const int numHalfedges = (16 + 4) * numSubdivisions;
					tmpHalfedges = std::make_unique<WorkingCopyHalfedge>(numHalfedges, m_halfedgeMesh.getHalfedges().size());
				}

				// each quad contains up to 4 original vertices (some of them are obviously shared with their neighbors.
				// each subdivision will create up to 4 new ones
				{
					const int numVertices = (4 + 4) * numSubdivisions;
					// Set the remove-flag to true by default. This makes it easier to mark duplicate vertices after copying.
					tmpVertices = std::make_unique<WorkingCopyVertex>(numVertices, m_vertices.size(), Flags::RemoveFlag());
				}

				// Each subdivision will run on one face and will create up to 4 new faces
				{
					const int numFaces = (1 + 4) * numSubdivisions;
					tmpFaces = std::make_unique<WorkingCopyFace>(numFaces, m_halfedgeMesh.getHalfedgeFaces().size());
				}
				cudaCheckError();
			}

			{
				Performance performance = createWorkingCopies(*tmpFaces, *tmpHalfedges, *tmpVertices);
				performances.addChild(performance);
			}

			{
				PerformanceTimer timer("Free Halfedge-HashTable", totalPerformance);
				m_halfedgeMesh.freeHashTable();
			}

			if (s_debugStep >= SplittingStep::SubdivideQuads) {
				PerformanceTimer timer("Subdivide Quads", performances);
				subdivideQuads(*tmpVertices, *tmpHalfedges, *tmpFaces);
			}

			if (s_debugStep >= SplittingStep::CollapseNonManifoldEdges) {
				Performance performance = collapseNonManifoldEdges(*tmpVertices, *tmpHalfedges, *tmpFaces);
				performances.addChild(performance);
			}

			if (s_debugStep >= SplittingStep::MoveVerticesToSurface) {
				PerformanceTimer timer("Move Vertices to Surface", performances);
				moveVerticesToSurface(*tmpVertices, isoValue);
			}

			if (s_copyDataBack || s_debugStep >= SplittingStep::EnableIterativeRefinement) {
				Performance performance = copyWorkingCopiesBack(*tmpVertices, *tmpHalfedges, *tmpFaces);
				performances.addChild(performance);
				if (!s_renderInterestingAreas) {
					PerformanceTimer timer("Deallocate WorkingCopies", performances);
					tmpHalfedges.reset(nullptr);
					tmpVertices.reset(nullptr);
					tmpFaces.reset(nullptr);
				}
			}

			{
				// Refresh the halfedge mesh and the non-manifold flags.
				m_halfedgeMesh.initializeHashTable();
				Performance performance = m_halfedgeMesh.refreshHashTable();
				performances.addChild(performance);
			}

			totalPerformance.addChild(performances);

			++i;

			if (!(s_debugStep >= SplittingStep::EnableIterativeRefinement) || !s_copyDataBack) break;
		}

		if (s_renderInterestingAreas && tmpVertices != nullptr && tmpHalfedges != nullptr && tmpFaces != nullptr) {
			LAZYENGINE_INFO("Render Interesting Areas");
			PerformanceTimer timer("Update Debug-Mesh", totalPerformance);
			updateDebugMesh(*tmpVertices, *tmpHalfedges, *tmpFaces, debugMesh);
		}

		return totalPerformance;
	}

	void ManifoldSplitter::imguiOptions() {
		if (ImGui::CollapsingHeader("ManifoldSplitter Options", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent();
			std::vector<std::string> steps = {
				"Subdivide Quads",
				"Collapse Non Manifold Edges",
				"Merge Vertices",
				"Update Halfedge Twins",
				"Move Vertices to Surface",
				"Enable Iterative Refinement"
			};

			if (ImGui::Button("Enable all steps")) {
				s_debugStep = SplittingStep::All;
			}
			for (int i = 0; i < steps.size(); ++i) {
				if (s_debugStep < i) break;
				SplittingStep currentStep = static_cast<SplittingStep>(i + 1);
				bool isChecked = s_debugStep >= currentStep;
				if (ImGui::Checkbox(steps[i].c_str(), &isChecked)) {
					if (!isChecked) {
						// enable the current step's predecessor
						s_debugStep = static_cast<SplittingStep>(i);
					}
					else {
						// enable the current step
						s_debugStep = currentStep;
					}
				}
			}
			ImGui::Unindent();
		}
	}

	void ManifoldSplitter::imguiDebug() {
		if (ImGui::CollapsingHeader("ManifoldSplitter Debugging")) {
			ImGui::Indent();
			ImGui::Checkbox("Copy Data Back", &s_copyDataBack);
			ImGui::Checkbox("Render Interesting Areas", &s_renderInterestingAreas);
			ImGui::Unindent();
		}
	}

	// ##################################################################### //
	// ### Helpers ######################################################### //
	// ##################################################################### //

	struct CountNonManifoldFaces {
		__device__ inline bool operator()(const HalfedgeFace& face) {
			return face.isNonManifold();
		}
	};

	size_t ManifoldSplitter::countNonManifoldFaces() const {
		auto dataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
		size_t count = thrust::count_if(dataView.begin(), dataView.end(), CountNonManifoldFaces());
		return count;
	}


	// ##################################################################### //
	// ### createWorkingCopies ############################################# //

	Performance ManifoldSplitter::createWorkingCopies(WorkingCopyFace& faces, WorkingCopyHalfedge& halfedges, WorkingCopyVertex& vertices) {
		Performance performance = { "Create Working Copies" };
		{
			PerformanceTimer timer("Copy Vertices to tmp", performance);
			copyOriginalVertices(vertices);
		}

		{
			PerformanceTimer timer("Copy Halfedges to tmp", performance);
			copyOriginalHalfedges(halfedges);
		}

		{
			PerformanceTimer timer("Copy Faces to tmp", performance);
			copyOriginalFaces(faces, halfedges);
		}

		{
			PerformanceTimer timer("Update Halfedge References", performance);
			updateHalfedgeReferences(vertices, halfedges, faces);
		}

		{
			PerformanceTimer timer("Update Flags", performance);
			updateFlags(halfedges, vertices);
		}

		return performance;
	}


	// ##################################################################### //
	// ### CopyOriginalVertices ############################################ //

	__global__ void copyVerticesToWorkingCopy(
		HalfedgeFaceDeviceVector originalFaces,
		HalfedgeDeviceVector originalHalfedges,
		VertexDeviceVector originalVertices,
		DeviceWorkingCopyVertex output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= originalFaces.size()) return;

		// Check if the current face is non-manifold. If not, early out.
		HalfedgeFace face = originalFaces.getAttributes(threadId);
		if (!face.isNonManifold()) return;

		// copy each vertex of this face
		HalfedgeID firstHalfedge = originalFaces.getFirstHalfedgeID(threadId);
		Halfedge currentHalfedge = originalHalfedges[firstHalfedge];

		for (int i = 0; i < 4; ++i) {
			// get the vertexID of the current halfedge
			VertexID currentVertex = currentHalfedge.getOriginVertexID();
			glm::vec3 position = originalVertices.getPosition(currentVertex);
			glm::vec3 normal = originalVertices.getNormal(currentVertex);

			// insert the vertex into the working copy
			auto copyIndex = output.dataVector.addVertex(position, normal);

			// set up mappings from and to the working copy
			output.copyToOriginal[copyIndex] = currentVertex;
			output.originalToCopy[currentVertex] = copyIndex;

			// update the current halfedge to the next one in the face
			currentHalfedge = originalHalfedges[currentHalfedge.getNext()];
		}
	}

	__global__ void markDuplicateVertices(
		DeviceWorkingCopy<VertexDeviceVector> vertices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.originalToCopy.size()) return;

		LookUpType copyIndex = vertices.originalToCopy[threadId];
		if (copyIndex == INVALID) return;

		// keep this vertex
		vertices.flags[copyIndex].unsetRemoveFlag();
	}

	void ManifoldSplitter::copyOriginalVertices(WorkingCopyVertex& vertices) {
		{
			auto dataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
			// check the vertices of each face. Only non-manifold faces will be processed.
			copyVerticesToWorkingCopy ToCudaArgs(dataView) (
				m_halfedgeMesh.getHalfedgeFaces(),
				m_halfedgeMesh.getHalfedges(),
				m_vertices,
				vertices
				);
			cudaCheckError();
		}
		{
			LazyEngine::DataView<LookUpType> dataView = *vertices.originalToCopy;
			// check the mapping for each original vertex. Writing this index was racy, so the last written vertex will stay.
			markDuplicateVertices ToCudaArgs(dataView) (
				vertices
				);
			cudaCheckError();
		}
	}


	// ##################################################################### //
	// ### CopyOriginalHalfedges ########################################### //

	__global__ void copyHalfedgesToWorkingCopy(
		HalfedgeFaceDeviceVector originalFaces,
		HalfedgeDeviceVector originalHalfedges,
		DeviceWorkingCopyHalfedge output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= originalFaces.size()) return;

		// check if the current face is actually non-manifold. Only proceed if so.
		HalfedgeFace face = originalFaces.getAttributes(threadId);
		if (!face.isNonManifold()) return;

		// copy each halfedge of this face
		HalfedgeID currentHalfedgeID = originalFaces.getFirstHalfedgeID(threadId);
		for (int i = 0; i < 4; ++i) {
			Halfedge currentHalfedge = originalHalfedges[currentHalfedgeID];

			// Insert the modified halfedge into the working copy
			auto copyIndex = output.dataVector.addHalfedge(currentHalfedge);
			output.copyToOriginal[copyIndex] = currentHalfedgeID;
			output.originalToCopy[currentHalfedgeID] = copyIndex;

			currentHalfedgeID = currentHalfedge.getNext();
		}
	}

	void ManifoldSplitter::copyOriginalHalfedges(WorkingCopyHalfedge& halfedges) {
		auto dataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
		// check the halfedges of each face. Only non-manifold faces will be processed.
		copyHalfedgesToWorkingCopy ToCudaArgs(dataView) (
			m_halfedgeMesh.getHalfedgeFaces(),
			m_halfedgeMesh.getHalfedges(),
			halfedges
			);
		cudaCheckError();
	}


	// ##################################################################### //
	// ### CopyOriginalFaces ############################################### //

	__global__ void copyFacesToWorkingCopy(
		HalfedgeFaceDeviceVector originalFaces,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= originalFaces.size()) return;

		// check if the current face is non-manifold
		HalfedgeFace face = originalFaces.getAttributes(threadId);
		if (!face.isNonManifold()) return;

		// copy the face
		HalfedgeID firstHalfedge = originalFaces.getFirstHalfedgeID(threadId);
		// map the halfedge to the copied index
		firstHalfedge = halfedges.originalToCopy[firstHalfedge];

		auto copyIndex = output.dataVector.pushFace(firstHalfedge, face);
		output.copyToOriginal[copyIndex] = threadId;
		output.originalToCopy[threadId] = copyIndex;
	}

	void ManifoldSplitter::copyOriginalFaces(WorkingCopyFace& faces, WorkingCopyHalfedge& halfedges) {
		auto dataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
		// only copy non-manifold halfedges
		copyFacesToWorkingCopy ToCudaArgs(dataView) (
			m_halfedgeMesh.getHalfedgeFaces(),
			halfedges,
			faces
			);
		cudaCheckError();
	}


	// ##################################################################### //
	// ### updateHalfedgeReferences ######################################## //

	__global__ void updateWorkingCopyHalfedges(
		DeviceWorkingCopyVertex vertices,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace faces
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		Halfedge& currentHalfedge = halfedges.dataVector[threadId];

		// look up the copied vertex id
		currentHalfedge.setOriginVertexID(vertices.originalToCopy[currentHalfedge.getOriginVertexID()]);
		// look up the copied face id
		currentHalfedge.setFace(faces.originalToCopy[currentHalfedge.getFace()]);
		// look up the copied next id
		currentHalfedge.setNext(halfedges.originalToCopy[currentHalfedge.getNext()]);
		// special care for the twin, as it was mapped to -twin-2 during the copy kernel.
		HalfedgeID originalTwin = currentHalfedge.getTwin();
		if (originalTwin < 0) {
			// This twin is and was INVALID from the start. Don't look it up.
			currentHalfedge.setTwin(originalTwin);
		}
		else {
			// This twin was originally set. Check if there is a copied counterpart for it.
			HalfedgeID mappedTwinID = halfedges.originalToCopy[originalTwin];

			if (mappedTwinID == INVALID) {
				// This Twin was not copied into the working copy.
				// map the twin halfedge to -twin - 2.
				// This maps -1 -> -1, 0 -> -2, 1 -> -3, ..., which means INVALID stays INVALID
				// Later on when copying back the halfedges, it will be easy to distinguish original twins that need to be restored and new twins.
				// Original twins will be negative and uneaqual to INVALID and can be restored by applying -mapped-2.
				// New twins will be 0 or positive.
				mappedTwinID = -originalTwin - 2;
				currentHalfedge.setTwin(mappedTwinID);
			}
			else {
				// This twin was copied into the working copy.
				currentHalfedge.setTwin(mappedTwinID);
			}
		}
	}

	void ManifoldSplitter::updateHalfedgeReferences(
		WorkingCopyVertex& vertices,
		WorkingCopyHalfedge& halfedges,
		WorkingCopyFace& faces)
	{
		// Currently, faces and halfedges in the working copies are still pointing to the original indices.
		// This function maps the indices to their copied counterparts.
		auto dataView = halfedges.dataVector.getDataView();
		updateWorkingCopyHalfedges ToCudaArgs(dataView) (
			vertices,
			halfedges,
			faces
			);
		cudaCheckError();
	}


	// ##################################################################### //
	// ### updateFlags ##################################################### //
	
	__global__ void markHalfedgesAsNonManifold(
		HalfedgeDeviceHashTable hashTable,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyVertex vertices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;

		// check if the current entry of the hashtable is empty
		if (hashTable.isBucketEmpty(threadId)) return;

		// get the hash-table's entries
		glm::ivec4 edges = hashTable.getHalfedges(threadId);
		bool isNonManifold = true;
		for (int i = 0; i < 4; ++i) {
			// The edge is non-manifold if all halfedges point to a valid index
			isNonManifold &= edges[i] != HashTable::INVALID_INDEX;
		}
		if (!isNonManifold) return;

		// set the non-manifold flag for each copied counterpart of the halfedges
		for (int i = 0; i < 4; ++i) {
			HalfedgeID halfedgeID = halfedges.originalToCopy[edges[i]];
			Halfedge halfedge = halfedges.dataVector[halfedgeID];
			halfedges.flags[halfedgeID].setNonManifoldFlag();
			vertices.flags[halfedge.getOriginVertexID()].setNonManifoldFlag();
		}
	}

	void ManifoldSplitter::updateFlags(WorkingCopyHalfedge& halfedges, WorkingCopyVertex& vertices) {
		// set halfedges to be non-manifold if they are mentioned in halfedgeMesh's non-manifold hash-table
		auto dataView = m_halfedgeMesh.getHashTable().getDataView();

		markHalfedgesAsNonManifold ToCudaArgs(dataView) (
			m_halfedgeMesh.getHashTable(), halfedges, vertices
		);
	}


	// ##################################################################### //
	// ### subdivideQuads ################################################## //

	__device__ inline void connectHalfedgeRing(HalfedgeID* halfedgeIDs, HalfedgeDeviceVector& halfedges) {
		for (int i = 0; i < 4; ++i) {
			int nextIndex = (i + 1) & 0x3;
			Halfedge& edge = halfedges[halfedgeIDs[i]];
			edge.setNext(halfedgeIDs[nextIndex]);
		}
	}

	__device__ inline void createHalfedgeRingFromVertexIDs(VertexID* corners, HalfedgeDeviceVector& halfedges, HalfedgeID *outHalfedges) {
		for (int i = 0; i < 4; ++i) {
			// create a new halfedge
			Halfedge edge;
			edge.setOriginVertexID(corners[i]);
			outHalfedges[i] = halfedges.addHalfedge(edge);
		}
		connectHalfedgeRing(outHalfedges, halfedges);
	}

	__device__ inline void replaceFaceIDOfHalfedges(HalfedgeID* halfedgeIDs, FaceID faceID, HalfedgeDeviceVector& halfedges, HalfedgeFaceDeviceVector& faces) {
		for (int i = 0; i < 4; ++i) {
			halfedges[halfedgeIDs[i]].setFace(faceID);
		}
		faces.getFirstHalfedgeID(faceID) = halfedgeIDs[0];
	}

	__device__ inline void createBoundaryQuads(
		VertexID *originalQuadVertices,
		HalfedgeID *originalHalfedges,
		VertexID *innerQuadVertices,
		HalfedgeID outFaces[4][4],
		HalfedgeDeviceVector& halfedges,
		HalfedgeFaceDeviceVector& faces
	) {
		int previous = 3;
		for (int i = 0; i < 4; ++i) {
			// get the vertex addresses of the quad corners in the correct order
			VertexID quadVertices[4] = {
				originalQuadVertices[i],
				innerQuadVertices[i],
				innerQuadVertices[previous],
				originalQuadVertices[previous]
			};

			for (int k = 0; k < 3; ++k) {
				// create a new halfedge
				Halfedge edge;
				edge.setOriginVertexID(quadVertices[k]);
				outFaces[i][k] = halfedges.addHalfedge(edge);
			}
			outFaces[i][3] = originalHalfedges[previous];

			connectHalfedgeRing(outFaces[i], halfedges);
			FaceID newFace = faces.pushFace(outFaces[i][0]);
			replaceFaceIDOfHalfedges(outFaces[i], newFace, halfedges, faces);

			previous = i;
		}
	}

	__device__ inline void swapHalfedgeMappings(HalfedgeID a, HalfedgeID b, DeviceWorkingCopyHalfedge& halfedges) {
		// first, swap the copy to original values
		LookUpType aOriginal = halfedges.copyToOriginal[a];
		LookUpType bOriginal = halfedges.copyToOriginal[b];
		halfedges.copyToOriginal[a] = bOriginal;
		halfedges.copyToOriginal[b] = aOriginal;

		// next, swap the originalToCopy of the halfedge that existed in the first place
		bool aOriginalExists = aOriginal != INVALID;
		bool bOriginalExists = bOriginal != INVALID;
		if (aOriginalExists && !bOriginalExists) {
			halfedges.originalToCopy[aOriginal] = b;
		}
		else if (!aOriginalExists && !bOriginalExists) {
			halfedges.originalToCopy[bOriginal] = a;
		}
		else {
			printf("%s: Unexpected State: aOriginalExists: %d, bOriginalExists: %d\n", __FUNCTION__, aOriginalExists, bOriginalExists);
		}
	}

	__global__ void performSubdivideQuads(
		HalfedgeFaceDeviceVector originalFaces,
		DeviceWorkingCopyVertex vertices,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace faces
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= originalFaces.size()) return;

		// check if the original face is non-manifold
		HalfedgeFace originalFaceAttributes = originalFaces.getAttributes(threadId);
		if (!originalFaceAttributes.isNonManifold()) return;

		// get the working copy counterpart's ID of the original face
		LookUpType faceID = faces.originalToCopy[threadId];

		// create 4 new vertices like in the old version
		// place the vertices at thirds inside of the quad
		glm::vec2 uvCoordinates[4] = {
			{0.33f, 0.33f},
			{0.67f, 0.33f},
			{0.67f, 0.67f},
			{0.33f, 0.67f},
		};

		// get all halfedges, halfedgeIDs, vertices and vertexIDs of the current face
		HalfedgeID firstHalfedge = faces.dataVector.getFirstHalfedgeID(faceID);
		// fetch all halfedgeIDs and halfedges
		HalfedgeID faceHalfedgeIDs[4];
		Halfedge faceHalfedges[4];
		halfedges.dataVector.fetchFaceInfo(firstHalfedge, faceHalfedgeIDs, faceHalfedges);
		
		// fetch the vertex ids, positions and normals
		VertexID vertexIDs[4];
		Vertex vertexPositions[4];
		for (int i = 0; i < 4; ++i) {
			vertexIDs[i] = faceHalfedges[i].getOriginVertexID();
			vertexPositions[i].position = vertices.dataVector.getPosition(vertexIDs[i]);
			vertexPositions[i].normal = vertices.dataVector.getNormal(vertexIDs[i]);
		}

		// interpolate the new vertex positions and normals
		Vertex subdivisionVertices[4];
		for (int i = 0; i < 4; ++i) {
			subdivisionVertices[i] = Interpolation<Vertex>::interpolateBilinearly(vertexPositions, uvCoordinates[i]);
		}

		// add the subdivision vertices to the shared vertex list
		VertexID innerQuadVertexAddresses[4];
		for (int i = 0; i < 4; ++i) {
			innerQuadVertexAddresses[i] = vertices.dataVector.addVertex(subdivisionVertices[i].position, subdivisionVertices[i].normal);
			// don't remove the vertex when copying back
			vertices.flags[innerQuadVertexAddresses[i]].unsetRemoveFlag();
		}

		// create halfedges that connect these new vertices
		// create the inner quad
		HalfedgeID innerHalfedgeFace[4];
		createHalfedgeRingFromVertexIDs(innerQuadVertexAddresses, halfedges.dataVector, innerHalfedgeFace);
		replaceFaceIDOfHalfedges(innerHalfedgeFace, faceID, halfedges.dataVector, faces.dataVector);

		// create the 4 outer quads, that each use one edge of the original quad
		HalfedgeID outerHalfedgeFaces[4][4];
		createBoundaryQuads(vertexIDs, faceHalfedgeIDs, innerQuadVertexAddresses, outerHalfedgeFaces, halfedges.dataVector, faces.dataVector);

		// connect the newly created halfedges to their correct twins
		int previous = 3;
		for (int i = 0; i < 4; ++i) {
			halfedges.dataVector.setAsTwins(outerHalfedgeFaces[i][2], outerHalfedgeFaces[previous][0]);
			halfedges.dataVector.setAsTwins(outerHalfedgeFaces[i][1], innerHalfedgeFace[previous]);
			
			previous = i;
		}

		// swap halfedge ids to enable the deletion of previously defined halfedges. These halfedges will then map to one of the center quad edges.
		for (int i = 0; i < 4; ++i) {
			HalfedgeID outer = faceHalfedgeIDs[i];
			HalfedgeID inner = innerHalfedgeFace[i];
			swapHalfedgeMappings(outer, inner, halfedges);
		}
	}

	void ManifoldSplitter::subdivideQuads(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces) {
		// run the subdivision on every quad of the original mesh. The working-copy faces will be modified/extended, so running on them would lead to race-conditions.
		auto dataView = m_halfedgeMesh.getHalfedgeFaces().getDataView();
		performSubdivideQuads ToCudaArgs(dataView) (
			m_halfedgeMesh.getHalfedgeFaces(),
			vertices,
			halfedges,
			faces
		);
		cudaCheckError();
	}


	// ##################################################################### //
	// ### collapseNonManifoldEdges ######################################## //

	__device__ void markVerticesAsMergable(VertexID a, VertexID b, DeviceMergeList& mergeList) {
		auto index = atomicAdd(&mergeList.getNumPartners(a), 1);
		if (index > 5) {
			printf("Added more than four vertex-neighbors: %d\n", index-1);
			return;
		}
		mergeList.getMergeElement(a, index - 1) = b;
	}

	__device__ void markHalfedgesAsMergeable(HalfedgeID a, HalfedgeID b, DeviceMergeList& mergeList) {
		auto index = atomicAdd(&mergeList.getNumPartners(a), 1);
		if (index > 5) {
			printf("Added more than four halfedge neighbors: %d\n", index-1);
			return;
		}
		mergeList.getMergeElement(a, index - 1) = b;
	}

	__device__ void collapseHalfedge(
		HalfedgeID halfedge,
		DeviceWorkingCopyHalfedge& halfedges,
		DeviceWorkingCopyFace& faces,
		DeviceMergeList& vertexMergeList,
		DeviceMergeList& halfedgeMergeList
	) {
		// Collect the Halfedges of halfedge's face and its twin's face
		HalfedgeID currentFaces[2][4];
		Halfedge currentHalfedge = halfedges.dataVector[halfedge];
		currentFaces[0][0] = halfedge;
		Halfedge currentTwin = halfedges.dataVector[currentHalfedge.getTwin()];
		currentFaces[1][0] = currentHalfedge.getTwin();
		for (int i = 1; i < 4; ++i) {
			// walk over both faces in opposite directions to find opposing vertices.
			currentFaces[0][i] = currentHalfedge.getNext();
			currentFaces[1][4 - i] = currentTwin.getNext();

			currentHalfedge = halfedges.dataVector[currentHalfedge.getNext()];
			currentTwin = halfedges.dataVector[currentTwin.getNext()];
		}

		// Mark the elements as removable
		for (int face = 0; face < 2; ++face) {
			for (int i = 0; i < 4; ++i) {
				// set the halfedges of the face as removable
				halfedges.flags[currentFaces[face][i]].setRemoveFlag();
			}
			// set the face itself as removable
			FaceID currentFaceID = halfedges.dataVector[currentFaces[face][0]].getFace();
			faces.flags[currentFaceID].setRemoveFlag();
		}

		// Combine Vertices that will have to merge in the MergeList
		// namely face[0]'s 2nd and face[1]'s 1st vertex have to be merged,
		// as well as face[0]'s 3rd and face[1]'s 2nd vertex.
		{
			// merge vertices 2-1
			currentHalfedge = halfedges.dataVector[currentFaces[0][2]];
			currentTwin = halfedges.dataVector[currentFaces[1][1]];
			// update currentHalfedge's vertex first
			markVerticesAsMergable(currentHalfedge.getOriginVertexID(), currentTwin.getOriginVertexID(), vertexMergeList);
			// now update currentTwin's vertex
			markVerticesAsMergable(currentTwin.getOriginVertexID(), currentHalfedge.getOriginVertexID(), vertexMergeList);
		}

		{
			// merge vertices 3-2	
			currentHalfedge = halfedges.dataVector[currentFaces[0][3]];
			currentTwin = halfedges.dataVector[currentFaces[1][2]];
			// update currentHalfedge's vertex first
			markVerticesAsMergable(currentHalfedge.getOriginVertexID(), currentTwin.getOriginVertexID(), vertexMergeList);
			// now update currentTwin's vertex
			markVerticesAsMergable(currentTwin.getOriginVertexID(), currentHalfedge.getOriginVertexID(), vertexMergeList);
		}

		// mark the halfedges as mergeable too
		{
			for (int i = 1; i < 4; ++i) {
				currentHalfedge = halfedges.dataVector[currentFaces[0][i]];
				currentTwin = halfedges.dataVector[currentFaces[1][i]];
				markHalfedgesAsMergeable(currentFaces[0][i], currentTwin.getTwin(), halfedgeMergeList);
				markHalfedgesAsMergeable(currentTwin.getTwin(), currentFaces[0][i], halfedgeMergeList);
				
				markHalfedgesAsMergeable(currentFaces[1][i], currentHalfedge.getTwin(), halfedgeMergeList);
				markHalfedgesAsMergeable(currentHalfedge.getTwin(), currentFaces[1][i], halfedgeMergeList);
			}
		}
	}

	__global__ void performCollapseNonManifoldHalfedges(
		DeviceWorkingCopyFace faces,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceMergeList vertexMergeList,
		DeviceMergeList halfedgeMergeList
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		// Only continue if the current halfedge is indeed non manifold.
		Flags halfedgeFlags = halfedges.flags[threadId];
		if (!halfedgeFlags.isNonManifold()) return;

		// Check the halfedge's twin's ID. If the twin has the lower ID, it will handle the collapse.
		HalfedgeID twin = halfedges.dataVector[threadId].getTwin();
		if (twin == INVALID) {
			printf("%s: The non-manifold halfedge's twin is undefined!\n", __FUNCTION__);
			return;
		}
		else if (twin < 0) {
			printf("%s: The non-manifold halfedge's twin is not part of the WorkingCopy!\n", __FUNCTION__);
			return;
		}


		// leave the collapse to the twin if its ID is smaller.
		if (twin < threadId) return;
		
		// actually collapse the halfedge by removing both of its adjacent faces and their halfedges.
		collapseHalfedge(threadId, halfedges, faces, vertexMergeList, halfedgeMergeList);
	}

	/**
	 *	Returns true if the caller can bail, because this is not a special case at all.
	 */
	__device__ bool sortHalfedgesAsNextOnes(int numHalfedges, HalfedgeID *toBeSorted, DeviceWorkingCopyHalfedge& halfedges) {
		// Sort the collected HalfedgeIDs, so that the first one of them has no predecessor that is also non-manifold
		// After that, the sorting defines true neighborhoods of halfedges.
		int sortedHalfedges[4] = { -1, -1, -1, -1 };
		// first, look for a halfedge that has no next halfedge contained in the toBeSorted array.
		for (int i = 0; i < numHalfedges; ++i) {
			Halfedge currentHalfedge = halfedges.dataVector[toBeSorted[i]];
			bool hasNext = false;
			for (int k = 0; k < numHalfedges; ++k) {
				hasNext |= toBeSorted[k] == currentHalfedge.getNext();
			}
			if (hasNext) continue;
			if (sortedHalfedges[0] != -1 && numHalfedges == 2) {
				// This Quad has two non-manifold-edges on opposite sides. It can be ignored entirely.
				return true;
			}
			sortedHalfedges[0] = i;
		}

		if (sortedHalfedges[0] == -1) {
			printf("%s: Somehow, there was no first halfedge found!\n", __FUNCTION__);
			return true;
		}

		// now find the halfedge that lists the latest added halfedge as next
		// This starts with the halfedge that is not referenced as next to any of the other halfedges
		HalfedgeID currentNext = toBeSorted[sortedHalfedges[0]];
		int currentNextIndex = 1;
		for (int k = 0; k < numHalfedges; ++k) {
			// try this for each halfedge.
			for (int i = 0; i < numHalfedges; ++i) {
				// compare the next-ids for each halfedge to currentNext
				HalfedgeID currentHalfedgeID = toBeSorted[i];
				Halfedge currentHalfedge = halfedges.dataVector[currentHalfedgeID];
				if (currentHalfedge.getNext() == currentNext) {
					// This halfedge is the predecessor to currentNext.
					sortedHalfedges[currentNextIndex] = i;
					++currentNextIndex;
					currentNext = currentHalfedgeID;
					break;
				}
			}
		}

		if (sortedHalfedges[numHalfedges - 1] == -1) {
			printf("%s: Not all Halfedges have been sorted!\n", __FUNCTION__);
			return true;
		}

		// now sort the halfedges by their sorting index
		HalfedgeID tmp[4] = { -1, -1, -1, -1 };
		for (int i = numHalfedges - 1; i >= 0; --i) {
			tmp[(numHalfedges - 1) - i] = toBeSorted[sortedHalfedges[i]];
		}

		// output the sorted halfedges to toBeSorted
		for (int i = 0; i < 4; ++i) {
			toBeSorted[i] = tmp[i];
		}

		return false;
	}

	__global__ void mergeVertices(
		DeviceWorkingCopyVertex vertices,
		DeviceMergeList vertexMergeList
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.dataVector.size()) return;

		// get all partners of this halfedge's vertex
		VertexID vertex = threadId;
		MergePartners partners = vertexMergeList.getAllPartnersOf(vertex);

		constexpr int maxNumPartners = (sizeof(partners.partners) / sizeof(*partners.partners));

		if (partners.numPartners == 1) {
			// The vertex has no merge-partners except for itself.
			return;
		}
		else if (partners.numPartners < 1 || partners.numPartners > maxNumPartners) {
			glm::vec3 vertexPosition = vertices.dataVector.getPosition(vertex);
			printf("%d: [%.2f,%.2f,%.2f] Invalid number of partners: %d\n", vertex, VECCC(vertexPosition), partners.numPartners);
			return;
		}

		// check if this vertex's partners have a smaller id. If they do, leave the computations to them.
		VertexID smallestCorner = vertex;

		// Get this vertex's position and normal.
		glm::vec3 vertexPosition = vertices.dataVector.getPosition(vertex);
		glm::vec3 vertexNormal = vertices.dataVector.getNormal(vertex);

		bool vertexIsObsolete = false;
		for (int i = 0; i < partners.numPartners - 1; ++i) {
			VertexID partner = partners.partners[i];
			vertexPosition += vertices.dataVector.getPosition(partner);
			vertexNormal += vertices.dataVector.getNormal(partner);
			
			if (partner < smallestCorner) {
				vertexIsObsolete = true;
				smallestCorner = partner;
			}
		}

		if (vertexIsObsolete) {
			// this vertex can be removed, it will be replaced with smallestCorner.
			vertices.flags[vertex].setRemoveFlag();
			return;
		}

		// average out the vertex position and normal
		vertexPosition /= static_cast<float>(partners.numPartners);
		vertexNormal /= static_cast<float>(partners.numPartners);
		vertices.dataVector.getPosition(smallestCorner) = vertexPosition;
		vertices.dataVector.getNormal(smallestCorner) = vertexPosition;
	}

	__global__ void updateHalfedgeVertexIDs(
		DeviceWorkingCopyHalfedge halfedges,
		DeviceMergeList vertexMergeList
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		// get the halfedge's vertex
		Halfedge& halfedge = halfedges.dataVector[threadId];
		VertexID vertexID = halfedge.getOriginVertexID();

		// get all partners of the vertex
		MergePartners vertexPartners = vertexMergeList.getAllPartnersOf(vertexID);
		// If there are no merge-partners, return early.
		if (vertexPartners.numPartners == 1) return;
		// get the smallest vertexID of the partners
		VertexID smallestVertex = vertexID;
		for (int i = 0; i < vertexPartners.numPartners - 1; ++i) {
			VertexID partner = vertexPartners.partners[i];
			if (partner < smallestVertex) {
				smallestVertex = partner;
			}
		}

		// update the origin vertex id to the smallest vertex
		halfedge.setOriginVertexID(smallestVertex);
	}

	__global__ void mergeHalfedges(
		DeviceWorkingCopyHalfedge halfedges,
		DeviceMergeList halfedgeMergeList
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		// check if the current halfedge is flagged for removal
		Flags halfedgeFlags = halfedges.flags[threadId];
		if (halfedgeFlags.isRemoveFlagSet()) return;

		// get the partners of this halfedge
		MergePartners partners = halfedgeMergeList.getAllPartnersOf(threadId);
		
		if (partners.numPartners == 1) {
			// This halfedge doesn't need to be merged
			return;
		}

		// search for a valid twin.
		HalfedgeID validTwin = INVALID;
		for (int i = 0; i < partners.numPartners - 1; ++i) {
			HalfedgeID partner = partners.partners[i];
			// get the partner's twin
			HalfedgeID twin = halfedges.dataVector[partner].getTwin();
			// get the twin's flags
			Flags twinFlags = halfedges.flags[twin];
			if (!twinFlags.isRemoveFlagSet()) {
				validTwin = twin;
				break;
			}
		}

		// There should always only be one valid partner left.
		if (validTwin == INVALID) {
#ifdef LAZYENGINE_DEBUG
			printf("%s: No Valid Twin found!\n", __FUNCTION__);
#endif
			return;
		}

		// only let the lower of the two ids update the twin-state
		if (validTwin < threadId) return;

		halfedges.dataVector.setAsTwins(threadId, validTwin);
	}

	Performance ManifoldSplitter::collapseNonManifoldEdges(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces) {
		Performance performance = { "collapse non-manifold Edges" };
		
		// contains the vertices that should be merged together
		std::unique_ptr<HostMergeList> mergeListVertices;
		// contains the halfedges that should become new twins
		std::unique_ptr<HostMergeList> mergeListHalfedges;
		{
			PerformanceTimer timer("Vertex Merge List Initialization", performance);
			// create a list that will keep track of vertex mergers.
			mergeListVertices = std::make_unique<HostMergeList>(vertices.dataVector.size());
			mergeListHalfedges = std::make_unique<HostMergeList>(halfedges.dataVector.size());
			cudaCheckError();
		}

		if (s_debugStep >= SplittingStep::CollapseNonManifoldEdges) {
			PerformanceTimer timer("Delete and Mark mergeable Vertices and Halfedges", performance);
			// collapse each halfedge that is marked as non-manifold
			auto dataView = halfedges.dataVector.getDataView();
			performCollapseNonManifoldHalfedges ToCudaArgs(dataView) (
				faces, halfedges, *mergeListVertices, *mergeListHalfedges
			);
			cudaCheckError();
		}

		if (s_debugStep >= SplittingStep::MergeVertices) {
			PerformanceTimer timer("Merge Vertices", performance);
			// Run this on the vertices
			auto dataView = vertices.dataVector.getDataView();
			mergeVertices ToCudaArgs(dataView) (
				vertices, *mergeListVertices
			);
			cudaCheckError();
		}

		if (s_debugStep >= SplittingStep::MergeVertices) {
			PerformanceTimer timer("Update Halfedge VertexIDs", performance);
			// do this for every halfedge
			auto dataView = halfedges.dataVector.getDataView();
			updateHalfedgeVertexIDs ToCudaArgs(dataView) (
				halfedges, *mergeListVertices
			);
			cudaCheckError();
		}

		if (s_debugStep >= SplittingStep::UpdateHalfedgeTwins) {
			PerformanceTimer timer("Update Halfedge-Twins", performance);
			auto dataView = halfedges.dataVector.getDataView();
			mergeHalfedges ToCudaArgs(dataView) (
				halfedges, *mergeListHalfedges
			);
			cudaCheckError();
		}

		return performance;
	}


	// ##################################################################### //
	// ### moveVerticesToSurface ########################################### //

	__global__ void performMoveVerticesToSurface(
		DeviceWorkingCopyVertex vertices,
		UniformGrid<float> grid,
		float isoValue
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.dataVector.size()) return;
		if (vertices.flags[threadId].isRemoveFlagSet()) return;

		// compute the cell-index of the vertex
		glm::vec3 point = vertices.dataVector.getPosition(threadId);
		glm::ivec3 index3D = grid.getIndex3DFromPosition(point);
		// get the cell-corner values at that index
		float scalars[8];
		grid.evaluateCellCornerValues(scalars, index3D);
		
		// now iterate to the surface
		VertexToSurfaceIterator iterator(isoValue, scalars);
		
		// transform point local coordinates [0^3, 1^3] using the cell boundaries
		glm::vec3 minCell = grid.getOrigin() + (glm::vec3(index3D) * grid.getDeltas());
		glm::vec3 maxCell = grid.getOrigin() + (glm::vec3(index3D+1) * grid.getDeltas());
		point = (point - minCell) / (maxCell - minCell);

		iterator.movePointToSurface(point);

		// transform the point to world-space coordinated
		point = grid.getOrigin() + (glm::vec3(index3D) + point) * grid.getDeltas();
		vertices.dataVector.getPosition(threadId) = point;
	}

	void ManifoldSplitter::moveVerticesToSurface(WorkingCopyVertex& vertices, float isoValue) {
		auto dataView = vertices.dataVector.getDataView();
		performMoveVerticesToSurface ToCudaArgs(dataView) (
			vertices,
			m_grid,
			isoValue
		);
		cudaCheckError();
	}


	// ##################################################################### //
	// ### copyWorkingCopiesBack ########################################### //

	__global__ void copyVerticesBack(
		DeviceWorkingCopyVertex vertices,
		VertexDeviceVector output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.dataVector.size()) return;

		// check the vertex flags. Ignore this vertex if it is marked as removed.
		Flags flags = vertices.flags[threadId];
		if (flags.isRemoveFlagSet()) return;

		// check the vertex mapping
		LookUpType originalVertex = vertices.copyToOriginal[threadId];
		glm::vec3 position = vertices.dataVector.getPosition(threadId);
		glm::vec3 normal = vertices.dataVector.getNormal(threadId);
		if (originalVertex == INVALID) {
			// This vertex didn't exist in the original mesh. Create a new one
			originalVertex = output.addVertex(position, normal);
			vertices.copyToOriginal[threadId] = originalVertex;
			//vertices.originalToCopy[originalVertex] = threadId;
		}
		else {
			// This vertex exists in the output. Override its position and normal.
			output.getPosition(originalVertex) = position;
			output.getNormal(originalVertex) = normal;
		}
	}

	__global__ void copyHalfedgesBack(
		DeviceWorkingCopyHalfedge halfedges,
		HalfedgeDeviceVector output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		// Check the halfedge flags. Ignore this halfedge if it is marked as removed.
		Flags flags = halfedges.flags[threadId];
		if (flags.isRemoveFlagSet()) return;

		// check the halfedge mapping
		Halfedge halfedge = halfedges.dataVector[threadId];
		LookUpType originalHalfedge = halfedges.copyToOriginal[threadId];
		if (originalHalfedge == INVALID) {
			// This halfedge didn't exist in the original mesh. Create a new one.
			originalHalfedge = output.addHalfedge(halfedge);
			halfedges.copyToOriginal[threadId] = originalHalfedge;
			//halfedges.originalToCopy[originalHalfedge] = threadId;
		}
		else {
			// This halfedge exists in the output. Override it
			output[originalHalfedge] = halfedge;
		}
	}

	__global__ void copyFacesBack(
		DeviceWorkingCopyFace faces,
		DeviceWorkingCopyHalfedge halfedges,
		HalfedgeFaceDeviceVector output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.dataVector.size()) return;

		// Check the face flags. Ignore this face if it is marked as removed.
		Flags flags = faces.flags[threadId];
		if (flags.isRemoveFlagSet()) return;

		// check the face mapping
		LookUpType originalFace = faces.copyToOriginal[threadId];
		HalfedgeID firstHalfedge = halfedges.copyToOriginal[faces.dataVector.getFirstHalfedgeID(threadId)];
		HalfedgeFace attributes = faces.dataVector.getAttributes(threadId);
		if (originalFace == INVALID) {
			// This face doesn't exist in the output. Create a new one.
			originalFace = output.pushFace(firstHalfedge, attributes);
			faces.copyToOriginal[threadId] = originalFace;
			//faces.originalToCopy[originalFace] = threadId;
		}
		else {
			// This face existed in the original mesh. Override it.
			output.getFirstHalfedgeID(originalFace) = firstHalfedge;
			output.getAttributes(originalFace) = attributes;
		}
	}

	__global__ void updateHalfedgesBack(
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace faces,
		DeviceWorkingCopyVertex vertices,
		HalfedgeDeviceVector output
	) {
		LazyEngine::CUDAInfo info;
		unsigned  int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		Flags flags = halfedges.flags[threadId];
		if (flags.isRemoveFlagSet()) return;

		Halfedge& halfedge = output[halfedges.copyToOriginal[threadId]];
		halfedge.setNext(halfedges.copyToOriginal[halfedge.getNext()]);
		halfedge.setFace(faces.copyToOriginal[halfedge.getFace()]);
		halfedge.setOriginVertexID(vertices.copyToOriginal[halfedge.getOriginVertexID()]);
		// The twin is a special case.
		HalfedgeID twin = halfedge.getTwin();
		if (twin < 0) {
			// map the twin to its original id.
			twin = -twin - 2;
			// The twin is now either INVALID or a previously valid id.
			halfedge.setTwin(twin);
			if (twin != INVALID) {
				// The twin is not invalid. Also update its twin.
				output[twin].setTwin(halfedges.copyToOriginal[threadId]);
			}
		}
		else {
			twin = halfedges.copyToOriginal[twin];
			halfedge.setTwin(twin);
		}
	}

	Performance ManifoldSplitter::copyWorkingCopiesBack(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces) {
		Performance performance = { "Copy Working Copies Back" };

		{
			PerformanceTimer timer("Free obsolete buffers", performance);
			vertices.originalToCopy.reset(new MonitoredThrustBuffer<LookUpType>(1, "Vertices-OriginalToCopy"));
			halfedges.originalToCopy.reset(new MonitoredThrustBuffer<LookUpType>(1, "Halfedges-OriginalToCopy"));
			faces.originalToCopy.reset(new MonitoredThrustBuffer<LookUpType>(1, "Faces-OriginalToCopy"));
			cudaCheckError();
		}

		{
			PerformanceTimer timer("Resize HalfedgeMesh-Buffers", performance);
#ifdef LAZYENGINE_DEBUG
			LAZYENGINE_INFO("HalfedgeMesh before Resizing: Faces {0}/{1}, Halfedges {2}/{3}, Vertices {4}/{5}, Shared Vertices: {6}/{7}",
				m_halfedgeMesh.getHalfedgeFaces().size(), m_halfedgeMesh.getHalfedgeFaces().capacity(),
				m_halfedgeMesh.getHalfedges().size(), m_halfedgeMesh.getHalfedges().capacity(),
				m_halfedgeMesh.getHalfedgeVertices().size(), m_halfedgeMesh.getHalfedgeVertices().capacity(),
				m_vertices.size(), m_vertices.capacity()
			);
#endif
			// Vertices are set to 20'000'000 capacity, so they don't need any resizing.
			
			// Reserve a bit more memory than actually necessary, just to be safe.
			int numFaces = m_halfedgeMesh.getHalfedgeFaces().size() + faces.dataVector.size();
			m_halfedgeMesh.getHalfedgeFaces().resize(numFaces);
			cudaCheckError();

			int numHalfedges = m_halfedgeMesh.getHalfedges().size() + halfedges.dataVector.size();
			m_halfedgeMesh.getHalfedges().resize(numHalfedges);
			cudaCheckError();

			int numVertices = m_vertices.size() + vertices.dataVector.size();
			m_halfedgeMesh.getHalfedgeVertices().resize(numVertices);
			cudaCheckError();

#ifdef LAZYENGINE_DEBUG
			LAZYENGINE_INFO("HalfedgeMesh after Resizing: Faces {0}/{1}, Halfedges {2}/{3}, Vertices {4}/{5}, Shared Vertices: {6}/{7}",
				m_halfedgeMesh.getHalfedgeFaces().size(), m_halfedgeMesh.getHalfedgeFaces().capacity(),
				m_halfedgeMesh.getHalfedges().size(), m_halfedgeMesh.getHalfedges().capacity(),
				m_halfedgeMesh.getHalfedgeVertices().size(), m_halfedgeMesh.getHalfedgeVertices().capacity(),
				m_vertices.size(), m_vertices.capacity()
			);
#endif
		}

		{
			PerformanceTimer timer("Copy Vertices back", performance);
			auto dataView = vertices.dataVector.getDataView();
			copyVerticesBack ToCudaArgs(dataView) (
				vertices,
				m_vertices
			);
			cudaCheckError();
		}

		{
			PerformanceTimer timer("Copy Halfedges back", performance);
			auto dataView = halfedges.dataVector.getDataView();
			copyHalfedgesBack ToCudaArgs(dataView) (
				halfedges,
				m_halfedgeMesh.getHalfedges()
			);
			cudaCheckError();
		}

		{
			PerformanceTimer timer("Copy Faces back", performance);
			auto dataView = faces.dataVector.getDataView();
			copyFacesBack ToCudaArgs(dataView) (
				faces,
				halfedges,
				m_halfedgeMesh.getHalfedgeFaces()
			);
			cudaCheckError();
		}

		{
			PerformanceTimer timer("Update Halfedge References", performance);
			auto dataView = halfedges.dataVector.getDataView();
			updateHalfedgesBack ToCudaArgs(dataView) (
				halfedges,
				faces,
				vertices,
				m_halfedgeMesh.getHalfedges()
			);
			cudaCheckError();
		}

		return performance;
	}


	// ##################################################################### //
	// ### updateDebugMesh ################################################# //

	__global__ void insertVertices(
		DeviceWorkingCopyVertex vertices,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace faces,
		LazyEngine::DataView<Vertex> vertexBuffer
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.dataVector.size()) return;

		// If the halfedge is marked as removable, skip it.
		Flags flags = halfedges.flags[threadId];
		if (flags.isRemoveFlagSet()) {
			return;
		}

		// Get the current halfedge
		Halfedge halfedge = halfedges.dataVector[threadId];

		// get the halfedge's origin vertex
		Vertex vertex;
		vertex.position = vertices.dataVector.getPosition(halfedge.getOriginVertexID());
		vertex.normal = vertices.dataVector.getNormal(halfedge.getOriginVertexID());

		// compute a color depending on the vertex's flags
		float hue = 180.f;
		if (vertices.flags[halfedge.getOriginVertexID()].isWeird()) {
			hue = 97.f;
		}
		if (vertices.flags[halfedge.getOriginVertexID()].isNonManifold()) {
			hue = 0.f;
		}
		vertex.color = glm::vec4(LazyEngine::Color::HSVtoRGB(hue, 0.8f, 0.8f), 1.f);

		// Output the vertex to the vertex buffer
		vertexBuffer[halfedge.getOriginVertexID()] = vertex;
	}

	__global__ void insertIndices(
		DeviceWorkingCopyVertex vertices,
		DeviceWorkingCopyHalfedge halfedges,
		DeviceWorkingCopyFace faces,
		LazyEngine::DataView<uint32_t> indexBuffer
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.dataVector.size()) return;

		// If the current face is marked as removable, skip it.
		Flags faceFlags = faces.flags[threadId];
		if (faceFlags.isRemoveFlagSet()) {
			return;
		}

		// get the vertexIDs of the current face
		HalfedgeID firstHalfedge = faces.dataVector.getFirstHalfedgeID(threadId);
		NeighborVertices corners = halfedges.dataVector.getFaceVertices(firstHalfedge);

		int i = 0;
		// write triangle 1
		indexBuffer[6 * threadId + (i++)] = corners.x;
		indexBuffer[6 * threadId + (i++)] = corners.y;
		indexBuffer[6 * threadId + (i++)] = corners.z;
		// write triangle 2
		indexBuffer[6 * threadId + (i++)] = corners.z;
		indexBuffer[6 * threadId + (i++)] = corners.w;
		indexBuffer[6 * threadId + (i++)] = corners.x;
	}

	void ManifoldSplitter::updateDebugMesh(WorkingCopyVertex& vertices, WorkingCopyHalfedge& halfedges, WorkingCopyFace& faces, Mesh& debugMesh) {
		size_t vertexBytes = vertices.dataVector.size() * sizeof(Vertex);
		size_t numIndices = faces.dataVector.size() * 6;
		debugMesh.resize(vertexBytes, numIndices);

		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(debugMesh.getVertices());
		LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(debugMesh.getIndices());

		auto vertexBuffer = vertexInterop.getMapping();
		auto indexBuffer = indexInterop.getMapping();

		{
			auto dataView = halfedges.dataVector.getDataView();
			insertVertices ToCudaArgs(dataView) (
				vertices, halfedges, faces, vertexBuffer
			);
			cudaCheckError();
		}

		{
			auto dataView = faces.dataVector.getDataView();
			insertIndices ToCudaArgs(dataView) (
				vertices, halfedges, faces, indexBuffer
			);
			cudaCheckError();
		}

	}

}