#include "ManifoldCreator.h"

#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include "../Utils/Interpolation.h"
#include "../Utils/Intersections.inl"
#include "../Utils/AsymptoticDecider.h"
#include "../Utils/Utilities.h"
#include "../CellIntersection.h"

namespace DMC {

	// ##################################################################### //
	// ### Fix Non-Manifold-Edge Connections ############################### //
	// ##################################################################### //

	struct OffsetDirections {
		glm::ivec3 xOffset;
		glm::ivec3 yOffset;
	};

	struct VertexPositionCell {
		glm::vec3 position;
		glm::ivec3 cellIndex3D;
		VertexID vertexID;
	};

	/**
	 *	contains two vertices that are connected via an edge.
	 *	vertex0 will always be the vertex of the cell with a lower linear index
	 */
	struct EdgePositions {
		VertexPositionCell vertex0;
		VertexPositionCell vertex1;
	};

	__device__ inline bool checkNudgedPosition(glm::vec3 vertex0, glm::vec3 offset0, glm::vec3 vertex1, glm::vec3 offset1, UniformGrid<float>& grid) {
		glm::vec3 nudgedPosition0 = vertex0 + offset0;
		glm::ivec3 nudgedCell0 = grid.getIndex3DFromPosition(nudgedPosition0);

		glm::vec3 nudgedPosition1 = vertex1 + offset1;
		glm::ivec3 nudgedCell1 = grid.getIndex3DFromPosition(nudgedPosition1);

		glm::ivec3 difference = nudgedCell0 - nudgedCell1;
		int sum = abs(difference.x) + abs(difference.y) + abs(difference.z);
		return sum == 1;
	}

	__device__ inline glm::vec3 iterationToOffset(int iteration) {
		// interpret iteration as binary.
		glm::vec3 result = glm::ivec3(0.f, 0.f, 0.f);
		result.x = (iteration & BIT(0)) ? 1.f : 0.f;
		result.y = (iteration & BIT(1)) ? 1.f : 0.f;
		result.z = (iteration & BIT(2)) ? 1.f : 0.f;

		// negate x if BIT 3 is set
		result.x = (iteration & BIT(3)) ? -result.x : result.x;
		// negate y if BIT 4 is set
		result.y = (iteration & BIT(4)) ? -result.y : result.y;
		// negate z if BIT 5 is set
		result.z = (iteration & BIT(5)) ? -result.z : result.z;

		result *= LazyEngine::Constants::EPSILON;

		return result;
	}

	__device__ inline EdgePositions computeEdgeVertices(HalfedgeDeviceVector& halfedges, VertexDeviceVector& vertices, Halfedge halfedge, UniformGrid<float>& grid) {

		// Get the two vertices that make up the edge
		glm::vec3 vertex0 = vertices.getPosition(halfedge.getOriginVertexID());
		HalfedgeID next = halfedge.getNext();
		if (next < 0 || next >= halfedges.size()) {
			printf("%s:%d Next is Invalid: %d\n", __FUNCTION__, __LINE__, next);
		}
		glm::vec3 vertex1 = vertices.getPosition(halfedges[next].getOriginVertexID());

		EdgePositions edge;
		glm::ivec3 vertex0Cell = grid.getIndex3DFromPosition(vertex0);
		glm::ivec3 vertex1Cell = grid.getIndex3DFromPosition(vertex1);

		// Whoops, it seems that sometimes these cells are no direct neigbors!
		// This is caused by vertices that are very close to the cell-boundaries.
		// Try nudging the vertices slightly in xyz-directions to force neighboring cells.
		for (int i = 0; i < 0x40; ++i) {
			// nudge vertex 0
			glm::vec3 offset0 = iterationToOffset(i);
			glm::vec3 offset1 = offset0;
			int finalK = -1;
			for (int k = 0; k < 0x40; ++k) {
				// nudge vertex 1
				offset1 = iterationToOffset(k);
				if (checkNudgedPosition(vertex0, offset0, vertex1, offset1, grid)) {
					finalK = k;
					break;
				}
			}
			if (finalK != -1) {
				vertex0 = vertex0 + offset0;
				vertex0Cell = grid.getIndex3DFromPosition(vertex0);

				vertex1 = vertex1 + offset1;
				vertex1Cell = grid.getIndex3DFromPosition(vertex1);
				break;
			}
		}

		if (grid.getGlobalIndex(vertex0Cell) < grid.getGlobalIndex(vertex1Cell)) {
			edge.vertex0.position = vertex0;
			edge.vertex0.cellIndex3D = vertex0Cell;
			edge.vertex0.vertexID = halfedge.getOriginVertexID();
			edge.vertex1.position = vertex1;
			edge.vertex1.cellIndex3D = vertex1Cell;
			edge.vertex1.vertexID = halfedges[next].getOriginVertexID();
		}
		else {
			edge.vertex0.position = vertex1;
			edge.vertex0.cellIndex3D = vertex1Cell;
			edge.vertex0.vertexID = halfedges[next].getOriginVertexID();
			edge.vertex1.position = vertex0;
			edge.vertex1.cellIndex3D = vertex0Cell;
			edge.vertex1.vertexID = halfedge.getOriginVertexID();
		}
		return edge;
	}

	__device__ inline glm::ivec3 computeCellWallNormal(glm::ivec3 cellA, glm::ivec3 cellB) {
		return glm::max(cellA, cellB) - glm::min(cellA, cellB);

	}

	__device__ inline OffsetDirections computeOffsetDirections(glm::ivec3 cellA, glm::ivec3 cellB) {
		glm::ivec3 cellWallNormal = computeCellWallNormal(cellA, cellB);
		OffsetDirections result;
		if (cellWallNormal.x == 0) {
			result.xOffset = glm::ivec3(1, 0, 0);
			if (cellWallNormal.y == 0) {
				result.yOffset = glm::ivec3(0, 1, 0);
			}
			else {
				result.yOffset = glm::ivec3(0, 0, 1);
			}
		}
		else {
			result.xOffset = glm::ivec3(0, 1, 0);
			result.yOffset = glm::ivec3(0, 0, 1);
		}
		return result;
	}

	__device__ inline void computeCellWallValues(float output[4], glm::ivec3 cellA, glm::ivec3 cellB, UniformGrid<float>& grid, const OffsetDirections& offsets) {
		// use the index3D of a cell-point between the two cells
		glm::ivec3 minIndex = glm::max(cellA, cellB);
		// Always start at the smallest index3D, the walk around starting with the xOffset direction
		output[0] = grid[minIndex];
		output[1] = grid[minIndex + offsets.xOffset];
		output[2] = grid[minIndex + offsets.xOffset + offsets.yOffset];
		output[3] = grid[minIndex + offsets.yOffset];
	}

	__device__ inline void computeCellWallVertices(glm::vec3 output[4], glm::ivec3 cellA, glm::ivec3 cellB, UniformGrid<float>& grid, const OffsetDirections& offsets) {
		// use the index3D of a cell-point between the two cells
		glm::ivec3 minIndex = glm::max(cellA, cellB);
		// Always start at the smallest index3D, the walk around starting with the xOffset direction
		output[0] = grid.getOrigin() + grid.getDeltas() * glm::vec3(minIndex);
		output[1] = grid.getOrigin() + grid.getDeltas() * glm::vec3(minIndex + offsets.xOffset);
		output[2] = grid.getOrigin() + grid.getDeltas() * glm::vec3(minIndex + offsets.xOffset + offsets.yOffset);
		output[3] = grid.getOrigin() + grid.getDeltas() * glm::vec3(minIndex + offsets.yOffset);
	}

	__device__ inline bool isHashTableEntryNonManifold(unsigned int index, HalfedgeDeviceVector& halfedges, HalfedgeDeviceHashTable& hashTable) {
		glm::ivec4 edges = hashTable.getHalfedges(index);
		return (
			edges.x != -1 &&
			edges.y != -1 &&
			edges.z != -1 &&
			edges.w != -1
		);
	}

	__device__ inline glm::vec2 downgradeVec3(glm::vec3 point, OffsetDirections offsets) {
		bool omitX = offsets.xOffset.x == 0 && offsets.yOffset.x == 0;
		bool omitY = offsets.xOffset.y == 0 && offsets.yOffset.y == 0;

		if (omitX) {
			return { point.y, point.z };
		}
		else if (omitY) {
			return { point.x, point.z };
		}
		return { point.x, point.y };
	}

	__device__ inline void mapHalfedgeToCellWallCorners(
		HalfedgeID outSortedHalfedges[4],
		glm::vec3 planeCorners[4],
		glm::ivec4 halfedgeIDs,
		HalfedgeDeviceVector& halfedges,
		VertexDeviceVector& vertices,
		OffsetDirections offsets
	) {
		// This operates in 2 dimensions only. Omit the perpendicular dimension of the cell-wall
		glm::vec2 minWall = downgradeVec3(planeCorners[0], offsets);
		glm::vec2 maxWall = downgradeVec3(planeCorners[2], offsets);

		for (int i = 0; i < 4; ++i) {
			HalfedgeID currentHalfedgeID = halfedgeIDs[i];
			Halfedge halfedge = halfedges[currentHalfedgeID];

			// get the first of the two vertices of this face that is not part of the non-manifold edge
			halfedge = halfedges[halfedge.getNext()];
			halfedge = halfedges[halfedge.getNext()];
			VertexID vertex = halfedge.getOriginVertexID();

			glm::vec3 vertexPosition = vertices.getPosition(vertex);
			glm::vec2 vertex2D = downgradeVec3(vertexPosition, offsets);

			// find the direction into which the vertex is extruding from the cell-wall
			glm::vec2 minOffset = vertex2D - minWall;
			glm::vec2 maxOffset = vertex2D - maxWall;

			bool isBottom = minOffset.y < 0.f;
			bool isRight = maxOffset.x > 0.f;
			bool isTop = maxOffset.y > 0.f;
			bool isLeft = minOffset.x < 0.f;

			if (isBottom) outSortedHalfedges[0] = currentHalfedgeID;
			if (isRight) outSortedHalfedges[1] = currentHalfedgeID;
			if (isTop) outSortedHalfedges[2] = currentHalfedgeID;
			if (isLeft) outSortedHalfedges[3] = currentHalfedgeID;
		}
	}

	__device__ void reconnectHalfedgesUsingIDs(HalfedgeDeviceVector& halfedges, HalfedgeID halfedgeIDs[4], bool connect01) {
		if (connect01) {
			// connect sortedHalfedge 0 and sortedHalfedge 1
			halfedges[halfedgeIDs[0]].setTwin(halfedgeIDs[1]);
			halfedges[halfedgeIDs[1]].setTwin(halfedgeIDs[0]);

			// connect sortedHalfedge 2 and sortedHalfedge 3
			halfedges[halfedgeIDs[2]].setTwin(halfedgeIDs[3]);
			halfedges[halfedgeIDs[3]].setTwin(halfedgeIDs[2]);
		}
		else {
			// connect sortedHalfedge 0 and sortedHalfedge 3
			halfedges[halfedgeIDs[0]].setTwin(halfedgeIDs[3]);
			halfedges[halfedgeIDs[3]].setTwin(halfedgeIDs[0]);

			// connect sortedHalfedge 1 and sortedHalfedge 2
			halfedges[halfedgeIDs[1]].setTwin(halfedgeIDs[2]);
			halfedges[halfedgeIDs[2]].setTwin(halfedgeIDs[1]);
		}
	}

	__global__ void reconnectHalfedges(
		HalfedgeDeviceVector halfedges,
		HalfedgeDeviceHashTable hashTable,
		VertexDeviceVector vertices,
		UniformGrid<float> grid,
		float isoValue
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		// check bounds
		if (threadId >= hashTable.capacity()) return;
		if (hashTable.isBucketEmpty(threadId)) return;
		if (!isHashTableEntryNonManifold(threadId, halfedges, hashTable)) return;

		// Look up the hash-table
		glm::ivec4 edges = hashTable.getHalfedges(threadId);

		// get one of the halfedges that are currently packed together
		Halfedge& halfedge0 = halfedges[edges.x];

		// ##################################################################### //
		// ### Compute the Cell-Wall and the edge's intersection point in it ### //

		// get the edge vertices
		EdgePositions edge = computeEdgeVertices(halfedges, vertices, halfedge0, grid);

		// compute the direction of the cell-wall that is being crossed by the edge
		OffsetDirections offsets = computeOffsetDirections(edge.vertex0.cellIndex3D, edge.vertex1.cellIndex3D);

		// get the cell-wall's values
		float wallValues[4];
		computeCellWallValues(wallValues, edge.vertex0.cellIndex3D, edge.vertex1.cellIndex3D, grid, offsets);

		// get the corner points of the cell wall
		glm::vec3 planeCorners[4];
		computeCellWallVertices(planeCorners, edge.vertex0.cellIndex3D, edge.vertex1.cellIndex3D, grid, offsets);

		HalfedgeID sortedHalfedgeIDs[4] = { -1, -1, -1, -1 };
		mapHalfedgeToCellWallCorners(sortedHalfedgeIDs, planeCorners, edges, halfedges, vertices, offsets);

		for (int i = 0; i < 4; ++i) {
			if (sortedHalfedgeIDs[i] == -1) {
				// In this case, at least one quad couldn't be intersected with a cell-boundary-ray.
				// I don't know when this happens, but let's find out!
				printf("Take a peek at halfedge %d. Edge: %d -- %d\n", edges.x, edge.vertex0.vertexID, edge.vertex1.vertexID);
				return;
			}
		}

		// Now run the asymptotic decider
		AsymptoticDecider2D decider(wallValues[0], wallValues[1], wallValues[2], wallValues[3]);

		bool connect01 = !decider.quadrant0BelongsTogether(isoValue);

		reconnectHalfedgesUsingIDs(halfedges, sortedHalfedgeIDs, connect01);
	}


	// ##################################################################### //
	// ### Detect Non-Manifold Vertices #################################### //
	// ##################################################################### //

	struct VertexFaceFan {
		HalfedgeID fanKey = -1;
		VertexID originVertex = -1;

		HostDevice inline bool operator()(const VertexFaceFan& a, const VertexFaceFan& b) {
			if (a.originVertex == b.originVertex) {
				return a.fanKey < b.fanKey;
			}
			return a.originVertex < b.originVertex;
		}

		HostDevice inline friend bool operator==(const VertexFaceFan& a, const VertexFaceFan& b) {
			return a.originVertex == b.originVertex && a.fanKey == b.fanKey;
		}
	};

	__device__ inline HalfedgeID stepClockwiseAroundVertex(Halfedge start, HalfedgeDeviceVector& halfedges) {
		if (start.hasTwin()) {
			start = halfedges[start.getTwin()];
			return start.getNext();
		}
		return Halfedge::INVALID_ID;
	}

	__device__ inline HalfedgeID stepCounterClockwiseAroundVertex(Halfedge start, HalfedgeDeviceVector& halfedges) {
		start = halfedges[start.getNext()];
		start = halfedges[start.getNext()];
		start = halfedges[start.getNext()];
		if (start.hasTwin()) {
			return start.getTwin();
		}
		return Halfedge::INVALID_ID;
	}

	template <bool goClockwise>
	__device__ inline HalfedgeID stepAroundVertex(HalfedgeID halfedge, HalfedgeDeviceVector& halfedges) {
		if constexpr (goClockwise) {
			return stepClockwiseAroundVertex(halfedges[halfedge], halfedges);
		}
		return stepCounterClockwiseAroundVertex(halfedges[halfedge], halfedges);
	}

	/**
	 *	Iterates over neighboring faces and returns the smallest HalfedgeID of those that are the same as the initial halfedge's vertex.
	 */
	template <bool goClockwise>
	__device__ inline HalfedgeID iterateOverNeighboringFaces(HalfedgeID start, HalfedgeDeviceVector& halfedges) {
		// use the start halfedge as reference
		Halfedge currentHalfedge = halfedges[start];
		VertexID originVertexID = currentHalfedge.getOriginVertexID();
		HalfedgeID currentHalfedgeID = start;
		HalfedgeID result = start;
		// step either clockwise or counter-clockwise around the vertex.
		currentHalfedgeID = stepAroundVertex<goClockwise>(currentHalfedgeID, halfedges);
		// contine walking in the desired direction and save the smallest halfedge-id.
		while (currentHalfedgeID != Halfedge::INVALID_ID) {
			currentHalfedge = halfedges[currentHalfedgeID];

			// check if we ended up in a loop
			if (currentHalfedgeID == result) break;
			// Check if the origin vertex changed (which should not happen, but does happen :(( )
			// Apparently this happens only for Non-Manifold Edges with 3 Faces!
			if (originVertexID != currentHalfedge.getOriginVertexID()) break;

			if (currentHalfedgeID < result && currentHalfedgeID != Halfedge::INVALID_ID) {
				result = currentHalfedgeID;
			}

			currentHalfedgeID = stepAroundVertex<goClockwise>(currentHalfedgeID, halfedges);
		}

		return result;
	}

	__global__ void detectFaceFans(
		HalfedgeDeviceVector halfedges,
		LazyEngine::DataView<VertexFaceFan> fanOutput
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		HalfedgeID startingHalfedgeID = threadId;
		Halfedge startingHalfedge = halfedges[startingHalfedgeID];
		HalfedgeID smallestID = startingHalfedgeID;
		VertexID originVertexID = startingHalfedge.getOriginVertexID();
		// find the halfedge with the smallest ID that belongs to startingHalfedge's Face-Fan

		HalfedgeID smallestClockwise = iterateOverNeighboringFaces<true>(startingHalfedgeID, halfedges);
		HalfedgeID smallestCounterClockwise = iterateOverNeighboringFaces<false>(startingHalfedgeID, halfedges);
		smallestID = min(smallestID, min(smallestClockwise, smallestCounterClockwise));

		// the lowest HalfedgeID should be a sufficient key, as this halfedge only belongs to the current vertex anyways.
		HalfedgeID key = smallestID;
		// Output a struct containing the key and the vertexID at threadId in an array.
		fanOutput[threadId] = { key, originVertexID };
}

	__global__ void transformFansToNumbers(
		LazyEngine::DataView<VertexFaceFan> fans,
		LazyEngine::DataView<int> output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= fans.size()) return;

		if (threadId == 0) {
			output[threadId] = 1;
			return;
		}
		VertexFaceFan current = fans[threadId];
		VertexFaceFan previous = fans[threadId - 1];
		output[threadId] = (current == previous) ? 0 : 1;
	}

	__global__ void countFansPerVertex(
		LazyEngine::DataView<VertexFaceFan> fans,
		LazyEngine::DataView<int> transformedFanChanges,
		LazyEngine::DataView<int> outNumFansPerVertex
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= transformedFanChanges.size()) return;

		int change = transformedFanChanges[threadId];
		if (change == 0) return;
		VertexFaceFan fan = fans[threadId];

		atomicAdd(&outNumFansPerVertex[fan.originVertex], 1);
	}

	__global__ void updateVertexFlags(
		LazyEngine::DataView<int> numFansPerVertex,
		LazyEngine::DataView<Flags> vertexFlags
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertexFlags.size()) return;

		int numFans = numFansPerVertex[threadId];
		if (numFans > 1) {
			vertexFlags[threadId].setMultipleFansFlag();
		}
		else {
			vertexFlags[threadId].unsetMutlipleFansFlag();
		}
	}

	// ##################################################################### //
	// ### Host Code ####################################################### //
	// ##################################################################### //

	void ManifoldCreator::fixManifoldTwins(VertexHostVector& vertices, const UniformGridHost<float>& grid, float isoValue) {
		assert(m_halfedgeMesh != nullptr);
		
		// check how many non-manifold cases exist
		size_t numSplits = m_halfedgeMesh->getNumNonManifoldEdges4();

		auto hashTableDataView = m_halfedgeMesh->getHashTable().getDataView();
		reconnectHalfedges ToCudaArgs(hashTableDataView) (
			m_halfedgeMesh->getHalfedges(),
			m_halfedgeMesh->getHashTable(),
			vertices,
			grid,
			isoValue
			);
		cudaCheckError();
	}

	Performance ManifoldCreator::detectNonManifoldVertices() {
		assert(m_halfedgeMesh != nullptr);
		Performance performance = { "Detect Non-Manifold Vertices" };
		thrust::device_vector<VertexFaceFan> fans;
		thrust::device_vector<int> fanDuplicates;
		thrust::device_vector<int> numFansPerVertex;
		{
			PerformanceTimer timer("Allocate Memory", performance);
			fans = thrust::device_vector<VertexFaceFan>(m_halfedgeMesh->getHalfedges().size(), VertexFaceFan());
			fanDuplicates = thrust::device_vector<int>(fans.size(), 0);
			numFansPerVertex = thrust::device_vector<int>(m_halfedgeMesh->getHalfedgeVertices().getDataView().size(), 0);
			cudaCheckError();
		}
		// Step 1: Detect and save Fans of all Halfedges
		{
			PerformanceTimer timer("Detect Face Fans", performance);
			auto dataView = m_halfedgeMesh->getHalfedges().getDataView();
			detectFaceFans ToCudaArgs(dataView) (m_halfedgeMesh->getHalfedges(), fans);
			cudaCheckError();
		}
		// Step 2: Sort the results
		{
			PerformanceTimer timer("Sort Fan-Results", performance);
			thrust::sort(fans.begin(), fans.end(), VertexFaceFan());
			cudaCheckError();
		}
		// Step 3: transform the entries into 0 and 1, depending on their previous entry.
		auto fanDataView = LazyEngine::DataView<VertexFaceFan>(fans);
		{
			PerformanceTimer timer("Transform Fans to Numbers", performance);
			transformFansToNumbers ToCudaArgs(fanDataView) (fans, fanDuplicates);
			cudaCheckError();
		}
		// Step 4: Extract the amount of fans for each vertex
		{
			PerformanceTimer timer("Count Fans per Vertex", performance);
			countFansPerVertex ToCudaArgs(fanDataView) (fans, fanDuplicates, numFansPerVertex);
			cudaCheckError();
		}
		// Step 5: Update Vertex Flags:
		{
			PerformanceTimer timer("Update Vertex Flags", performance);
			auto numFansDataView = LazyEngine::DataView<int>(numFansPerVertex);
			updateVertexFlags ToCudaArgs(numFansDataView) (numFansPerVertex, m_halfedgeMesh->getHalfedgeVertices().getFlagsDataView());
			cudaCheckError();
		}

		return performance;
	}

	struct CountVertices {
		HostDevice
		inline bool operator()(const Flags& flags) {
			return flags.hasMultipleFans();
		}
	};

	int ManifoldCreator::getNumNonManifoldVertices() {
		auto dataView = m_halfedgeMesh->getHalfedgeVertices().getFlagsDataView();
		int numNonManifoldVertices = thrust::count_if(dataView.begin(), dataView.end(), CountVertices());
		return numNonManifoldVertices;
	}

}