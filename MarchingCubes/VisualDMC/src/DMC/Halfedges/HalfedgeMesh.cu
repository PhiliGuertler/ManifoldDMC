#include "HalfedgeMesh.h"

#include "../Utils/Intersections.inl"

#include <thrust/count.h>

namespace DMC {


	// ##################################################################### //
	// ### HalfedgeMesh #################################################### //
	// ##################################################################### //

	__global__ void collectHalfedgeElements(
		QuadrilateralDeviceVector quads,
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		HalfedgeVertexDeviceVector vertices,
		HalfedgeDeviceHashTable hashTable
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		// get the vertex ids of this quad
		const glm::ivec4 quad = quads.getQuad(threadId);

		// save halfedges
		// As there are four halfedges for each quad, the address is computed as follows
		size_t halfedgeAddress = 4 * threadId;

		// Ignore the twins for now
		// Halfedge 1
		Halfedge halfedge = Halfedge(quad.x, threadId, halfedgeAddress + 1);
		halfedges.insertHalfedge(halfedgeAddress, halfedge);
		// Halfedge 2
		halfedge = Halfedge(quad.y, threadId, halfedgeAddress + 2);
		halfedges.insertHalfedge(halfedgeAddress + 1, halfedge);
		// Halfedge 3
		halfedge = Halfedge(quad.z, threadId, halfedgeAddress + 3);
		halfedges.insertHalfedge(halfedgeAddress + 2, halfedge);
		// Halfedge 4
		halfedge = Halfedge(quad.w, threadId, halfedgeAddress);
		halfedges.insertHalfedge(halfedgeAddress + 3, halfedge);

		// Collect faces
		faces.addFace(threadId, halfedgeAddress, quads.getAttribute(threadId));

		// Collect vertices. Never mind race conditions, the last writer will get the index.
		vertices[quad.x] = halfedgeAddress;
		vertices[quad.y] = halfedgeAddress + 1;
		vertices[quad.z] = halfedgeAddress + 2;
		vertices[quad.w] = halfedgeAddress + 3;

		// Set Hash table for connectivity computations
		hashTable.addHalfedge(quad.x, quad.y, halfedgeAddress);
		hashTable.addHalfedge(quad.y, quad.z, halfedgeAddress + 1);
		hashTable.addHalfedge(quad.z, quad.w, halfedgeAddress + 2);
		hashTable.addHalfedge(quad.w, quad.x, halfedgeAddress + 3);
	}

	__global__ void refreshHashTableGPU(
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		HalfedgeDeviceHashTable hashTable
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.size()) return;

		// get the vertex ids of this quad and its halfedgeIDs aswell
		glm::ivec4 quad;
		HalfedgeID halfedgeHandles[4];
		HalfedgeID currentHalfedgeHandle = faces.getFirstHalfedgeID(threadId);
		Halfedge halfedge = halfedges[currentHalfedgeHandle];
		for (int i = 0; i < 4; ++i) {
			// save origin vertex id and halfedge id
			quad[i] = halfedge.getOriginVertexID();
			halfedgeHandles[i] = currentHalfedgeHandle;
			// get the next halfedge
			currentHalfedgeHandle = halfedge.getNext();
			halfedge = halfedges[currentHalfedgeHandle];
		}

		// Set Hash table for connectivity computations
		hashTable.addHalfedge(quad.x, quad.y, halfedgeHandles[0]);
		hashTable.addHalfedge(quad.y, quad.z, halfedgeHandles[1]);
		hashTable.addHalfedge(quad.z, quad.w, halfedgeHandles[2]);
		hashTable.addHalfedge(quad.w, quad.x, halfedgeHandles[3]);
	}

	__global__ void refreshNonManifoldFlags(
		HalfedgeDeviceHashTable hashTable,
		HalfedgeDeviceVector halfedges,
		HalfedgeVertexDeviceVector vertices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;
		if (hashTable.isBucketEmpty(threadId)) return;

		const int numFaces = hashTable.getNumFaces(threadId);
		if (numFaces != 4) return;

		const glm::ivec4 edges = hashTable.getHalfedges(threadId);
		for (int i = 0; i < 4; ++i) {
			VertexID vertexID = halfedges[edges[i]].getOriginVertexID();
			vertices.getFlagsOf(vertexID).setNonManifoldFlag();
		}
	}

	__global__ void refreshWeirdFlags(
		HalfedgeDeviceHashTable hashTable,
		HalfedgeDeviceVector halfedges,
		HalfedgeVertexDeviceVector vertices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;
		if (hashTable.isBucketEmpty(threadId)) return;

		const int numFaces = hashTable.getNumFaces(threadId);
		if (numFaces != 3) return;

		const glm::ivec4 edges = hashTable.getHalfedges(threadId);
		for (int i = 0; i < 3; ++i) {
			VertexID vertexID = halfedges[edges[i]].getOriginVertexID();
			vertices.getFlagsOf(vertexID).setWeirdFlag();
		}
	}

	__global__ void refreshBoundaryFlags(
		HalfedgeDeviceHashTable hashTable,
		HalfedgeDeviceVector halfedges,
		HalfedgeVertexDeviceVector vertices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;
		if (hashTable.isBucketEmpty(threadId)) return;

		const int numFaces = hashTable.getNumFaces(threadId);
		if (numFaces != 1) return;

		const glm::ivec4 edges = hashTable.getHalfedges(threadId);
		VertexID vertexID = halfedges[edges.x].getOriginVertexID();
		vertices.getFlagsOf(vertexID).setIsBoundaryFlag();
	}

	__global__ void refreshFaceFlagsGPU(
		HalfedgeDeviceHashTable hashTable,
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;

		if (hashTable.isBucketEmpty(threadId)) return;

		// get the halfedges at this bucket
		const int numFaces = hashTable.getNumFaces(threadId);
		if (numFaces != 4) return;
	
		// set all faces of these edges as non-manifold
		const glm::ivec4 edges = hashTable.getHalfedges(threadId);
		for (int i = 0; i < numFaces; ++i) {
			FaceID faceID = halfedges[edges[i]].getFace();
			faces.getAttributes(faceID).setNonManifoldFlag();
		}
	}

	__global__ void connectHalfedgeTwins(
		HalfedgeDeviceHashTable hashTable,
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		HalfedgeVertexDeviceVector vertices,
		QuadrilateralDeviceVector quads
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= hashTable.capacity()) return;

		if (hashTable.isBucketEmpty(threadId)) return;

		// get the number of faces for the given edge

		const int numFaces = hashTable.getNumFaces(threadId);
		const glm::ivec4 edges = hashTable.getHalfedges(threadId);

		switch (numFaces) {
		case 1:
			// There is only one face for this halfedge, which means it is a boundary edge?
			// Either way, set it to be non-manifold

#ifdef LAZYENGINE_DEBUG
		{
			if (edges.x == -1 || edges.x >= halfedges.size()) {
				printf("Out of Bounds Case 1: [%d, %d, %d, %d]\n", edges.x, edges.y, edges.z, edges.w);
			}
		}
#endif

#if 0
			vertices.getFlagsOf(halfedges[edges.x].getOriginVertexID()).setIsBoundaryFlag();
#endif

			//faces.getAttributes(halfedges[edges.x].getFace()).setIsBoundaryFlag();
			//quads.getAttribute(halfedges[edges.x].getFace()).setIsBoundaryFlag();
			break;
		case 2:
			// This is the regular case, where two halfedges are actually twins
			// connect them
#ifdef LAZYENGINE_DEBUG
			{
				if (edges.x == -1 || edges.x >= halfedges.size() || edges.y == -1 || edges.y >= halfedges.size()) {
					printf("Out of Bounds Case 2: [%d, %d, %d, %d]\n", edges.x, edges.y, edges.z, edges.w);
				}
			}
#endif
			halfedges[edges.x].setTwin(edges.y);
			halfedges[edges.y].setTwin(edges.x);
			break;
		case 3:
			//printf("Wow, a 3-Halfedge-Case!\n");
			halfedges[edges.x].setTwin(edges.y);
			halfedges[edges.y].setTwin(edges.z);
			halfedges[edges.z].setTwin(edges.x);

#if 0
			for (int i = 0; i < 3; ++i) {
				vertices.getFlagsOf(halfedges[edges[i]].getOriginVertexID()).setWeirdFlag();
			}
#endif
			break;
		case 4:
			// ############################################################# //
			// This is the interesting case, where non-manifold edges occur! //
			// ############################################################# //
			// via normalen gucken, welche kanten/quads zusammengehoeren
#ifdef LAZYENGINE_DEBUG
			{
				if (edges.x == -1 || edges.x >= halfedges.size() || edges.y == -1 || edges.y >= halfedges.size() || edges.z == -1 || edges.z >= halfedges.size() || edges.w == -1 || edges.w >= halfedges.size()) {
					printf("Out of Bounds Case 4: [%d, %d, %d, %d]\n", edges.x, edges.y, edges.z, edges.w);
				}
			}
			//printf("These Edges are contained in a non manifoldness: %d, %d, %d, %d\n", edges.x, edges.y, edges.z, edges.w);
#endif
			// The original hacky way of dealing with this:
#define HACKY_OLD_WAY 1
#if HACKY_OLD_WAY
			// get vertices
			VertexID v0 = halfedges[edges.x].getOriginVertexID();
			VertexID v1 = halfedges[edges.y].getOriginVertexID();
			VertexID v2 = halfedges[edges.z].getOriginVertexID();

#if 0
			for (int i = 0; i < 4; ++i) {
				vertices.getFlagsOf(halfedges[edges[i]].getOriginVertexID()).setNonManifoldFlag();
			}
#endif

			if (v0 != v1) {
				halfedges[edges.x].setTwin(edges.y);
				halfedges[edges.y].setTwin(edges.x);
				halfedges[edges.z].setTwin(edges.w);
				halfedges[edges.w].setTwin(edges.z);
			}
			else if (v0 != v2) {
				halfedges[edges.x].setTwin(edges.z);
				halfedges[edges.z].setTwin(edges.x);
				halfedges[edges.y].setTwin(edges.w);
				halfedges[edges.w].setTwin(edges.y);
			}
			else {
				halfedges[edges.x].setTwin(edges.w);
				halfedges[edges.w].setTwin(edges.x);
				halfedges[edges.y].setTwin(edges.z);
				halfedges[edges.z].setTwin(edges.y);
			}

			for (int i = 0; i < 4; ++i) {
				Halfedge currentHalfedge = halfedges[edges[i]];
				//printf("Setting %d's face (%d) to non-manifold!\n", edges[i], currentHalfedge.getFace());
				faces.getAttributes(currentHalfedge.getFace()).setNonManifoldFlag();
				quads.getAttribute(currentHalfedge.getFace()).setNonManifoldFlag();
			}
			break;
#endif
		}
	}

	/**
	 *	Computes Vertex Valence of each vertex and stores it in the vertex map
	 */
	__global__ void computeVertexValence(HalfedgeDeviceVector halfedges, VertexMapDevice vertexMap) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		// Get this thread's halfedge and origin vertex
		const Halfedge halfedge = halfedges[threadId];
		const VertexID vertex = halfedge.getOriginVertexID();
		// Increase the vertex's valence by one
		atomicAdd(&(vertexMap.vertexValence(vertex)), 1);
		if (!halfedge.hasTwin()) {
			// Increase the neighbors valence by 1, as this neighbor is connected to it only with the current halfedge.
			const VertexID neighbor = halfedges[halfedge.getNext()].getOriginVertexID();
			atomicAdd(&(vertexMap.vertexValence(neighbor)), 1);
		}
	}

	// ##################################################################### //
	// ### P3X3Y Kernels ################################################### //
	// ##################################################################### //

	__global__ void markElementsP3X3Y(QuadrilateralDeviceVector quads, VertexMapDevice vertexMap) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		QuadrilateralAttribute& attributes = quads.getAttribute(threadId);

		if (attributes.isNonManifold() || attributes.isBoundary()) return;

		attributes.clearPatternFlags();
		glm::ivec4 quad = quads.getQuad(threadId);

		int valences[4];
		for (int i = 0; i < 4; ++i) {
			valences[i] = vertexMap.vertexValence(quad[i]);
		}

		bool flag1 = (valences[0] == 3 && valences[1] >= 5 && valences[2] == 3 && valences[3] >= 5);
		bool flag2 = (valences[0] >= 5 && valences[1] == 3 && valences[2] >= 5 && valences[3] == 3);
		if (flag1 || flag2) {
			attributes.setPatternFlag();
		}
	}

	__global__ void countVerticesP3X3Y(QuadrilateralDeviceVector quads, VertexMapDevice vertexMap) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		QuadrilateralAttribute& attributes = quads.getAttribute(threadId);

		attributes.clearPatternFlags();
		glm::ivec4 quad = quads.getQuad(threadId);

		int valences[4];
		for (int i = 0; i < 4; ++i) {
			valences[i] = vertexMap.vertexValence(quad[i]);
		}

		if (valences[0] == 3 && valences[1] >= 5 && valences[2] == 3 && valences[3] >= 5) {
			atomicAdd(&(vertexMap.elementCount(quad.x)), 1);
			atomicAdd(&(vertexMap.elementCount(quad.z)), 1);
			attributes.setPatternFlag();
		}
		else if (valences[0] >= 5 && valences[1] == 3 && valences[2] >= 5 && valences[3] == 3) {
			atomicAdd(&(vertexMap.elementCount(quad.y)), 1);
			atomicAdd(&(vertexMap.elementCount(quad.w)), 1);
			attributes.setPatternFlag();
		}
	}

	__device__ inline void handleMerge(glm::ivec4 vertexIndices, VertexDeviceVector& vertices, VertexMapDevice& vertexMap, QuadrilateralAttribute& attribute) {
		// compute new position of the merged vertex
		glm::vec3 newPosition = vertices.getPosition(vertexIndices.y);
		newPosition = newPosition + 0.5f * (vertices.getPosition(vertexIndices.w) - newPosition);
		vertices.getPosition(vertexIndices.x) = newPosition;
		
		// compute new normal of the merged vertex
		glm::vec3 newNormal = vertices.getNormal(vertexIndices.y);
		newNormal = newNormal + 0.5f * (vertices.getNormal(vertexIndices.w) - newNormal);
		newNormal = glm::normalize(newNormal);
		vertices.getNormal(vertexIndices.x) = newNormal;

		// mark duplicate vertex as to be removed
		vertexMap.vertexType(vertexIndices.z) = VertexType::PatternRemovable;

		// set twin of duplicate vertex to the merged vertex, so it can be removed
		vertexMap.mappingTarget(vertexIndices.z) = vertexIndices.x;

		// set Remove-Flag to enforce removal
		attribute.setRemoveFlag();
	}


	__global__ void mergeVerticesP3X3YColor(
		QuadrilateralDeviceVector quads,
		VertexMapDevice vertexMap,
		VertexDeviceVector vertices,
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		QuadrilateralAttribute& attributes = quads.getAttribute(threadId);

		if (!attributes.isP3X3Y()) return;

		// get neighbors
		NeighborFaces neighbors = halfedges.getNeighboringFaces(faces.getFirstHalfedgeID(threadId));
		for (int i = 0; i < 4; ++i) {
			// Check for non-manifold cases and skip them
			QuadrilateralAttribute attribute = quads.getAttribute(neighbors[i]);
			if (attribute.isNonManifold() || attributes.isBoundary()) {
				return;
			}
		}

		// get neighbor colors and vertex valence patterns
		int color = attributes.getColor();
		int neighborColors[4];
		bool neighborPatterns[4];
		for (int i = 0; i < 4; ++i) {
			QuadrilateralAttribute attribute = quads.getAttribute(neighbors[i]);
			neighborColors[i] = attribute.getColor();
			neighborPatterns[i] = attribute.isP3X3Y();
		}

		// Case analysis
		bool dontStop = true;
		for (int i = 0; i < 4; ++i) {
			if (neighborPatterns[i] && (neighborColors[i] <= color)) dontStop = false;
		}
		if (!dontStop) return;

		glm::ivec4 quad = quads.getQuad(threadId);
		int valences[4];
		for (int i = 0; i < 4; ++i) {
			valences[i] = vertexMap.vertexValence(quad[i]);
		}

		if (valences[0] == 3 && valences[1] >= 5 && valences[2] == 3 && valences[3] >= 5 /*&&
			vertexMap.elementCount(quad.x) == 1 && vertexMap.elementCount(quad.z) == 1*/) {

			handleMerge({ quad.x, quad.y, quad.z, quad.w }, vertices, vertexMap, attributes);
		}
		else if (valences[0] >= 5 && valences[1] == 3 && valences[2] >= 5 && valences[3] == 3 /*&&
			vertexMap.elementCount(quad.y) == 1 && vertexMap.elementCount(quad.w) == 1*/) {

			handleMerge({ quad.y, quad.x, quad.w, quad.z }, vertices, vertexMap, attributes);
		}
	}


	/**
	 *	Merges vertices if possible
	 */
	__global__ void mergeVerticesP3X3Y(
		QuadrilateralDeviceVector quads,
		VertexMapDevice vertexMap,
		VertexDeviceVector vertices,
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		QuadrilateralAttribute& attributes = quads.getAttribute(threadId);

		if (attributes.isNonManifold() || attributes.isBoundary()) return;
		if (!attributes.isP3X3Y()) return;

		// get neighbors
		NeighborFaces neighbors = halfedges.getNeighboringFaces(faces.getFirstHalfedgeID(threadId));
		for (int i = 0; i < 4; ++i) {
			// Check for non-manifold cases and skip them
			QuadrilateralAttribute attribute = quads.getAttribute(neighbors[i]);
			if (attribute.isNonManifold() || attributes.isBoundary()) {
				return;
			}
		}

		glm::ivec4 quad = quads.getQuad(threadId);
		int valences[4];
		for (int i = 0; i < 4; ++i) {
			valences[i] = vertexMap.vertexValence(quad[i]);
		}

		if (valences[0] == 3 && valences[1] >= 5 && valences[2] == 3 && valences[3] >= 5 &&
			vertexMap.elementCount(quad.x) == 1 && vertexMap.elementCount(quad.z) == 1) {
			
			handleMerge({quad.x, quad.y, quad.z, quad.w}, vertices, vertexMap, attributes);
		}
		else if (valences[0] >= 5 && valences[1] == 3 && valences[2] >= 5 && valences[3] == 3 &&
			vertexMap.elementCount(quad.y) == 1 && vertexMap.elementCount(quad.w) == 1) {

			handleMerge({ quad.y, quad.x, quad.w, quad.z }, vertices, vertexMap, attributes);
		}

	}

	__global__ void removeVerticesP3X3Y(VertexDeviceVector vertices, VertexMapDevice vertexMap, VertexDeviceVector verticesOut) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.size()) return;

		// ignore removable vertices
		if (vertexMap.vertexType(threadId) == VertexType::PatternRemovable) return;
		
		// copy vertices that cannot be removed
		size_t address = verticesOut.addVertex(vertices.getPosition(threadId), vertices.getNormal(threadId));
		vertexMap.mappingAddress(threadId) = address;
	}

	__global__ void removeQuadrilateralsP3X3Y(QuadrilateralDeviceVector quads, VertexMapDevice vertexMap, QuadrilateralDeviceVector quadsOut) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		QuadrilateralAttribute& attributes = quads.getAttribute(threadId);
		// ignore removable quads
		if (attributes.isRemoveable()) return;

		glm::ivec4 quad = quads.getQuad(threadId);
		glm::ivec4 newQuad;


		for (int i = 0; i < 4; ++i) {
			if (vertexMap.vertexType(quad[i]) == VertexType::PatternRemovable) {
				newQuad[i] = vertexMap.mappingAddress(vertexMap.mappingTarget(quad[i]));
			}
			else {
				newQuad[i] = vertexMap.mappingAddress(quad[i]);
			}
		}

		size_t address = quadsOut.addQuadrilateral(newQuad);
		if (attributes.isNonManifold()) quadsOut.getAttribute(address).setNonManifoldFlag();
		if (attributes.isBoundary()) quadsOut.getAttribute(address).setIsBoundaryFlag();
	}

	// ##################################################################### //
	// ### P3333 Kernels ################################################### //
	// ##################################################################### //

	/**
	 *	Marks elements as type P-3333 and prepares them for removal
	 */
	__global__ void markElementsP3333(HalfedgeFaceDeviceVector faces, HalfedgeDeviceVector halfedges, VertexMapDevice vertexMap) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= faces.size()) return;

		HalfedgeFace& faceAttributes = faces.getAttributes(threadId);
		if (faceAttributes.isNonManifold() || faceAttributes.isBoundary()) return;
		faceAttributes.clearPatternFlags();

		// Find the vertices of this face
		Halfedge quadEdges[4];
		int vertexValences[4];

		quadEdges[0] = halfedges[faces.getFirstHalfedgeID(threadId)];
		for (int i = 0; i < 4; ++i) {
			if (i < 4) {
				// get the next halfedge of the quad from the current edge
				quadEdges[i + 1] = halfedges[quadEdges[i].getNext()];
			}
			// get the vertex valence of the current edge
			vertexValences[i] = vertexMap.vertexValence(quadEdges[i].getOriginVertexID());
			
			// check if the 3-3-3-3 pattern holds, or quit early
			if (vertexValences[i] != 3) return;
		}

		// Find element neighbors
		FaceID neighbors[4];
		VertexID neighborVertices[4];	// v4-v7
		int neighborValences[4];
		for (int i = 0; i < 4; ++i) {
			// Get the neighbor's face id
			HalfedgeID twin = quadEdges[i].getTwin();
			neighbors[i] = halfedges[twin].getFace();
			
			// check if the neighbor is non manifold and early out if so
			auto attributes = faces.getAttributes(neighbors[i]);
			if (attributes.isNonManifold() || attributes.isBoundary()) return;
			
			// Get the neighboring vertex that is not part of the quad
			VertexID helper = halfedges[twin].getNext();
			helper = halfedges[helper].getNext();
			neighborVertices[i] = halfedges[helper].getOriginVertexID();

			// get the neighbor-vertices' valences
			neighborValences[i] = vertexMap.vertexValence(neighborVertices[i]);

			// If the neighbor valence is 3, removal is impossible. Early out
			if (neighborValences[i] == 3) return;
		}

		// Check if neighbor is of same type
		bool isNotRemovable = false;
		for (int i = 0; i < 4; ++i) {
			// compare consecutive neighbors (0-1, 1-2, 2-3, 3-0)
			isNotRemovable |= (neighborValences[i] == 3 && neighborValences[(i+1)&0x3] == 3);
		}
		if (isNotRemovable) return;

		// check for special case, where removing the element would generate a non-manifold mesh
		isNotRemovable = true;
		for (int i = 0; i < 4; ++i) {
			isNotRemovable &= neighborValences[i] == 4;
		}
		if (isNotRemovable) return;

		// Test are done, mark the element as removable, etc
		// Mark element as type P_3333
		faceAttributes.setVertexValenceFlag();
		for (int i = 0; i < 4; ++i) {
			// Mark neighbors as removable, if possible
			faces.getAttributes(neighbors[i]).setRemoveFlag();

			// Set vertex types
			VertexID vertex = quadEdges[i].getOriginVertexID();
			vertexMap.vertexType(vertex) = VertexType::Pattern3333;

			// set vertex twins
			vertexMap.mappingTarget(vertex) = neighborVertices[i];
		}

	}

	/**
	 *	Removes Vertices for the P3333 Simplification pattern
	 *	@param vertices: Input Vertices, that will be untouched
	 *	@param vertexMap: A prefilled map, that stores which vertices are required and which can be removed
	 *	@param newVertices: Output Vertices, which should be an empty Vector with the size of original vertices.
	 */
	__global__ void removeVerticesP3333(VertexDeviceVector vertices, VertexMapDevice vertexMap, VertexDeviceVector newVertices) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= vertices.size()) return;

		// Skip Vertices that are tagged with the 3333-Pattern, only copy needed vertices
		if (vertexMap.vertexType(threadId) == VertexType::Pattern3333) return;

		// Add the Vertex to the new vertex vector
		size_t address = newVertices.addVertex(vertices.getPosition(threadId), vertices.getNormal(threadId));
		// Keep the address for mapping later on
		vertexMap.mappingAddress(threadId) = address;
	}

	/**
	 *	Removes Quads for the P3333 Simplification pattern.
	 *	@param quads: Input Quads, that will be untouched
	 *	@param faces: Face-List of the Halfedge Datastructure
	 *	@param vertexMap: A prefilled map, that stores which vertices are required and where non required vertices map to
	 *	@param newQuads: Output Quads, which should be an empty Vector with the size of original Quads
	 */
	__global__ void removeQuadsP3333(QuadrilateralDeviceVector quads, HalfedgeFaceDeviceVector faces, VertexMapDevice vertexMap, QuadrilateralDeviceVector newQuads) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= quads.size()) return;

		HalfedgeFace attributes = faces.getAttributes(threadId);
		// Skip this face if it is marked as removable. This way it won't be listed in the output.
		if (attributes.isRemoveable()) return;

		// Compute new quad
		glm::ivec4 quad = quads.getQuad(threadId);
		if (attributes.isP3333()) {
			for (int i = 0; i < 4; ++i) {
				quad[i] = vertexMap.mappingAddress(vertexMap.mappingTarget(quad[i]));
			}
		}
		else {
			for (int i = 0; i < 4; ++i) {
				quad[i] = vertexMap.mappingAddress(quad[i]);
			}
		}

		// add the element to the output
		size_t address = newQuads.addQuadrilateral(quad);
		if (attributes.isNonManifold()) newQuads.getAttribute(address).setNonManifoldFlag();
		if (attributes.isBoundary()) newQuads.getAttribute(address).setIsBoundaryFlag();
	}


	// ##################################################################### //
	// ### Rendering Kernels ############################################### //
	// ##################################################################### //

	__global__ void insertHalfedgeFaceIndices(
		HalfedgeDeviceVector halfedges,
		HalfedgeFaceDeviceVector faces,
		LazyEngine::DataView<uint32_t> indices
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();

		if (threadId >= faces.size()) return;

		// get the first halfedge of this face
		HalfedgeID firstHalfedge = faces.getFirstHalfedgeID(threadId);

		NeighborVertices vertices = halfedges.getFaceVertices(firstHalfedge);

		// Triangle 1
		int i = 0;
		indices[6*threadId + (i++)] = vertices.x;
		indices[6*threadId + (i++)] = vertices.y;
		indices[6*threadId + (i++)] = vertices.z;
		// Triangle 2
		indices[6*threadId + (i++)] = vertices.z;
		indices[6*threadId + (i++)] = vertices.w;
		indices[6*threadId + (i++)] = vertices.x;
	}

	__global__ void insertVertices(
		VertexDeviceVector vertices,
		HalfedgeFaceDeviceVector halfedgeFaces,
		HalfedgeVertexDeviceVector halfedgeVertices,
		HalfedgeDeviceVector halfedges,
		LazyEngine::DataView<Vertex> output
	) {
		LazyEngine::CUDAInfo info;
		unsigned int threadId = info.getGlobalThreadId();
		if (threadId >= halfedges.size()) return;

		// get the halfedge originating at the current vertex
		Halfedge halfedge = halfedges[threadId];
		Flags flags = halfedgeVertices.getFlagsOf(halfedge.getOriginVertexID());

		// get the halfedge's origin Vertex
		Vertex vertex;
		vertex.position = vertices.getPosition(halfedge.getOriginVertexID());
		vertex.normal = vertices.getNormal(halfedge.getOriginVertexID());
		// Default hue: Cyan
		float hue = 180.f;
		if (flags.isNonManifold()) {
			// Is part of a Non-Manifold Edge: Red
			hue = 0.f;
		}
		if (flags.isWeird()) {
			// Is part of a Non-Manifold Edge with 3 Faces instead of 4: Yellow
			hue = 60.f;
		}
		if (flags.hasMultipleFans()) {
			// Is a Non-Manifold Vertex by itself: Purple
			hue = 270.f;
		}
		if (flags.isWeird() && flags.isNonManifold()) {
			// Magenta
			hue = 300.f;
		}
		if (flags.isNonManifold() && flags.hasMultipleFans()) {
			// dark blue
			hue = 240.f;
		}
		vertex.color = glm::vec4(LazyEngine::Color::HSVtoRGB(hue, 0.8f, 0.8f), 1.f);

		output[halfedge.getOriginVertexID()] = vertex;
	}

	// ##################################################################### //
	// ### Host Code ####################################################### //
	// ##################################################################### //

	HalfedgeMesh::HalfedgeMesh()
		: m_halfedges(nullptr)
		, m_halfedgeFaces(nullptr)
		, m_halfedgeVertices(nullptr)
		, m_hashTable(nullptr)
	{
		// empty
	}

	HalfedgeMesh::~HalfedgeMesh() {
		m_hashTable.reset(nullptr);
		m_halfedgeVertices.reset(nullptr);
		m_halfedgeFaces.reset(nullptr);
		m_halfedges.reset(nullptr);
	}


	void HalfedgeMesh::initializeBuffers(Performance& performances, size_t numQuads, size_t numVertices) {
		Performance perf = { "InitializeBuffers" };
		{
			PerformanceTimer timer("Halfedge-Buffer Initialization", perf);

			// Initialize Buffers (Not sure if this should be done every single time!)
			// A Resize should suffice, actually
			m_halfedges = std::make_unique<HalfedgeHostVector>(4 * numQuads);
			m_halfedgeFaces = std::make_unique<HalfedgeFaceHostVector>(numQuads);
			m_halfedgeVertices = std::make_unique<HalfedgeVertexHostVector>(numVertices);
		}
		auto per = initializeHashTable();
		perf.addChild(per);
		performances.addChild(per);
	}

	Performance HalfedgeMesh::initializeHashTable() {
		Performance performance = { "Halfedge-HashTable Initialization" };
		{
			PerformanceTimer timer("Buffer allocation", performance);
			// Dirty hack: If there are too many halfedges, sometimes the gpu-memory is not enough.
			// This limits the memory usage somewhat, however this hack may lead to infinite loops for datasets that are too big.
			size_t numElements = std::min(50'000'000, m_halfedges->capacity());
			m_hashTable = std::make_unique<HalfedgeHostHashTable>(numElements);
		}
		return performance;
	}

	Performance HalfedgeMesh::freeHashTable() {
		Performance performance = { "Halfedge-HashTable free" };
		{
			PerformanceTimer timer("Buffer deallocation", performance);
			m_hashTable.reset(nullptr);
		}
		return performance;
	}

	void HalfedgeMesh::collectHalfedges(Performance& performances, QuadrilateralHostVector& quads) {
		assert(m_halfedges != nullptr && m_halfedgeFaces != nullptr && m_halfedgeVertices != nullptr && m_hashTable != nullptr);
		PerformanceTimer timer("Halfedge Collection", performances);

		auto dataView = quads.getDataView();
		collectHalfedgeElements ToCudaArgs(dataView) (quads, *m_halfedges, *m_halfedgeFaces, *m_halfedgeVertices, *m_hashTable);
	}

	void HalfedgeMesh::connectHalfedges(Performance& performances, QuadrilateralHostVector& quads) {
		Performance per = { "Connect Halfedges" };
		{
			PerformanceTimer timer("Halfedge Connection", per);

			auto dataView = m_hashTable->getDataView();
			connectHalfedgeTwins ToCudaArgs(dataView) (*m_hashTable, *m_halfedges, *m_halfedgeFaces, *m_halfedgeVertices, quads);
		}

		{
			PerformanceTimer timer("Refresh Vertex Flags", per);
			auto hashTableDataView = m_hashTable->getDataView();
			refreshNonManifoldFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			refreshWeirdFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			refreshBoundaryFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			cudaCheckError();
		}
		performances.addChild(per);
	}

	Performance HalfedgeMesh::initialize(size_t numQuads, size_t numVertices, QuadrilateralHostVector& quads) {
		Performance performances = { "HalfedgeMesh Initialization" };

		initializeBuffers(performances, numQuads, numVertices);
		collectHalfedges(performances, quads);
		connectHalfedges(performances, quads);


		return performances;
	}

	Performance HalfedgeMesh::simplify3X3Y(VertexHostVector& vertices, QuadrilateralHostVector& quads) {
		Performance performances = {"Simplification: 3X3Y Color based"};

		size_t numVertices = vertices.size();
		size_t numQuads = quads.size();
		std::unique_ptr<VertexMapHost> vertexMap;
		std::unique_ptr<VertexHostVector> tmpVertices;
		std::unique_ptr<QuadrilateralHostVector> tmpQuads;

		{
			PerformanceTimer timer("Initialize Data Structures", performances);
			vertexMap = std::make_unique<VertexMapHost>(numVertices);
			tmpVertices = std::make_unique<VertexHostVector>(numVertices);
			tmpQuads = std::make_unique<QuadrilateralHostVector>(numQuads);
		}

		{
			PerformanceTimer timer("Compute Vertex Valence", performances);
			auto dataView = m_halfedges->getDataView();
			computeVertexValence ToCudaArgs(dataView) (*m_halfedges, *vertexMap);
		}

		{
			PerformanceTimer timer("Mark elements and vertices for removal", performances);
			auto dataView = quads.getDataView();
			markElementsP3X3Y ToCudaArgs(dataView) (quads, *vertexMap);
		}

		{
			PerformanceTimer timer("Merge Vertices", performances);
			auto dataView = quads.getDataView();
			mergeVerticesP3X3YColor ToCudaArgs(dataView) (quads, *vertexMap, vertices, *m_halfedges, *m_halfedgeFaces);
		}

		{
			auto dataView = vertices.getDataView();
			PerformanceTimer timer("Remove Vertices from list", performances);
			removeVerticesP3X3Y ToCudaArgs(dataView) (vertices, *vertexMap, *tmpVertices);
		}

		{
			auto dataView = quads.getDataView();
			PerformanceTimer timer("Remove Quadrilaterals", performances);
			removeQuadrilateralsP3X3Y ToCudaArgs(dataView) (quads, *vertexMap, *tmpQuads);
		}

		{
			PerformanceTimer timer("Copy Simplified Data Back", performances);
			tmpVertices->copyTo(vertices);
			tmpQuads->copyTo(quads);
		}

		{
			Performance recompute = { "Recompute Halfedges" };
			auto init = initialize(quads.size(), vertices.size(), quads);
			recompute.addChild(init);
			performances.addChild(recompute);
		}

		vertexMap.reset(nullptr);
		tmpVertices.reset(nullptr);
		tmpQuads.reset(nullptr);

		return performances;
	}
	
	Performance HalfedgeMesh::simplify3X3YOld(VertexHostVector& vertices, QuadrilateralHostVector& quads) {
		Performance performances = {"Simplification: 3X3Y-old"};

		size_t numVertices = vertices.size();
		size_t numQuads = quads.size();
		std::unique_ptr<VertexMapHost> vertexMap;
		std::unique_ptr<VertexHostVector> tmpVertices;
		std::unique_ptr<QuadrilateralHostVector> tmpQuads;

		{
			PerformanceTimer timer("Initialize Data Structures", performances);
			vertexMap = std::make_unique<VertexMapHost>(numVertices);
			tmpVertices = std::make_unique<VertexHostVector>(numVertices);
			tmpQuads = std::make_unique<QuadrilateralHostVector>(numQuads);
		}

		{
			auto dataView = m_halfedges->getDataView();
			PerformanceTimer timer("Compute Vertex Valence", performances);
			computeVertexValence ToCudaArgs(dataView) (*m_halfedges, *vertexMap);
		}

		{
			auto dataView = quads.getDataView();
			PerformanceTimer timer("Count vertices", performances);
			countVerticesP3X3Y ToCudaArgs(dataView) (quads, *vertexMap);
		}

		{
			auto dataView = quads.getDataView();
			PerformanceTimer timer("Merge Vertices", performances);
			mergeVerticesP3X3Y ToCudaArgs(dataView) (quads, *vertexMap, vertices, *m_halfedges, *m_halfedgeFaces);
		}

		{
			auto dataView = vertices.getDataView();
			PerformanceTimer timer("Remove Vertices from list", performances);
			removeVerticesP3X3Y ToCudaArgs(dataView) (vertices, *vertexMap, *tmpVertices);
		}

		{
			auto dataView = quads.getDataView();
			PerformanceTimer timer("Remove Quadrilaterals", performances);
			removeQuadrilateralsP3X3Y ToCudaArgs(dataView) (quads, *vertexMap, *tmpQuads);
		}

		{
			PerformanceTimer timer("Copy Simplified Data Back", performances);
			tmpVertices->copyTo(vertices);
			tmpQuads->copyTo(quads);
		}

		{
			Performance recompute = { "Recompute Halfedges" };
			auto init = initialize(quads.size(), vertices.size(), quads);
			recompute.addChild(init);
			performances.addChild(recompute);
		}

		vertexMap.reset(nullptr);
		tmpVertices.reset(nullptr);
		tmpQuads.reset(nullptr);


		return performances;
	}
	
	Performance HalfedgeMesh::simplify3333(VertexHostVector& vertices, QuadrilateralHostVector& quads) {
		Performance performances = {"Simplification: 3333"};

		std::unique_ptr<VertexMapHost> vertexMap;
		std::unique_ptr<VertexHostVector> tmpVertices;
		std::unique_ptr<QuadrilateralHostVector> tmpQuads;

		{
			PerformanceTimer timer("Initialize Data Structures", performances);
			vertexMap = std::make_unique<VertexMapHost>(vertices.size());
			tmpVertices = std::make_unique<VertexHostVector>(vertices.size());
			tmpQuads = std::make_unique<QuadrilateralHostVector>(quads.size());
		}

		{
			PerformanceTimer timer("Compute Vertex Valence", performances);
			auto dataView = m_halfedges->getDataView();
			computeVertexValence ToCudaArgs(dataView) (*m_halfedges, *vertexMap);
		}

		{
			PerformanceTimer timer("Mark elements with 3333 Valence Pattern", performances);
			auto dataView = quads.getDataView();
			markElementsP3333 ToCudaArgs(dataView) (*m_halfedgeFaces, *m_halfedges, *vertexMap);
		}

		{
			PerformanceTimer timer("Remove Vertices", performances);
			auto dataView = vertices.getDataView();
			removeVerticesP3333 ToCudaArgs(dataView) (vertices, *vertexMap, *tmpVertices);
		}

		{
			PerformanceTimer timer("Remove Elements", performances);
			auto dataView = quads.getDataView();
			removeQuadsP3333 ToCudaArgs(dataView) (quads, *m_halfedgeFaces, *vertexMap, *tmpQuads);
		}

		{
			PerformanceTimer timer("Copy Simplified Data back", performances);
			tmpVertices->copyTo(vertices);
			tmpQuads->copyTo(quads);
		}

		{
			Performance recompute = { "Recompute Halfedges" };
			auto init = initialize(quads.size(), vertices.size(), quads);
			recompute.addChild(init);
			performances.addChild(recompute);
		}

		vertexMap.reset(nullptr);
		tmpVertices.reset(nullptr);
		tmpQuads.reset(nullptr);

		return performances;
	}

	struct ResetFaceFlags {
		HostDevice inline void operator()(QuadrilateralAttribute& faceAttributes) {
			faceAttributes.clearPatternFlags();
			faceAttributes.unsetNonManifoldFlag();
		}
	};

	struct ResetVertexFlags {
		HostDevice inline void operator()(Flags& flags) {
			flags.reset();
		}
	};

	Performance HalfedgeMesh::refreshHashTable() {
		Performance performances = { "Refresh Hash Table" };
		{
			PerformanceTimer timer("Clearing HashTable", performances);
			m_hashTable->clear();
			cudaCheckError();
		}
		{
			PerformanceTimer timer("Clearing Flags", performances);
			auto faceDataView = m_halfedgeFaces->getDataView();
			thrust::for_each(faceDataView.begin(), faceDataView.end(), ResetFaceFlags());
			cudaCheckError();
			auto vertexDataView = m_halfedgeVertices->getFlagsDataView();
			thrust::for_each(vertexDataView.begin(), vertexDataView.end(), ResetVertexFlags());
			cudaCheckError();
		}
		{
			PerformanceTimer timer("Recollecting Edges", performances);
			auto facesDataView = m_halfedgeFaces->getFirstHalfedgeIDDataView();
			refreshHashTableGPU ToCudaArgs(facesDataView) (*m_halfedges, *m_halfedgeFaces, *m_hashTable);
			cudaCheckError();
		}
		{
			PerformanceTimer timer("Refresh Vertex Flags", performances);
			auto hashTableDataView = m_hashTable->getDataView();
			refreshNonManifoldFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			refreshWeirdFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			refreshBoundaryFlags ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeVertices);
			cudaCheckError();
		}
		{
			PerformanceTimer timer("Refresh Face Flags", performances);
			auto hashTableDataView = m_hashTable->getDataView();
			refreshFaceFlagsGPU ToCudaArgs(hashTableDataView) (*m_hashTable, *m_halfedges, *m_halfedgeFaces);
			cudaCheckError();
		}
		return performances;
	}

	uint32_t HalfedgeMesh::toVertexAndIndexBuffer(VertexHostVector& vertices, LazyEngine::DataView<Vertex>& vertexBuffer, LazyEngine::DataView<uint32_t>& indexBuffer) {
		{
			// copy vertices with their respective colors into the vertex buffer
			auto dataView = m_halfedges->getDataView();
			insertVertices ToCudaArgs(dataView) (vertices, *m_halfedgeFaces, *m_halfedgeVertices, *m_halfedges, vertexBuffer);
			cudaCheckError();
		}

		{
			// copy halfedgeFaces into the index buffer
			auto dataView = m_halfedgeFaces->getDataView();
			insertHalfedgeFaceIndices ToCudaArgs(dataView) (*m_halfedges, *m_halfedgeFaces, indexBuffer);
			cudaCheckError();
		}
		unsigned int numTriangles = m_halfedgeFaces->size() / 3;
		return numTriangles;
	}

	uint32_t HalfedgeMesh::toVertexAndIndexBuffer(VertexHostVector& vertices, Mesh& mesh) {
		// resize buffers first
		size_t verticesBytes = vertices.size() * sizeof(Vertex);
		size_t numIndices = m_halfedgeFaces->size() * 6;
		mesh.resize(verticesBytes, numIndices);

		LazyEngine::ScopedCUDAInterop<Vertex> vertexInterop(mesh.getVertices());
		LazyEngine::ScopedCUDAInterop<uint32_t> indexInterop(mesh.getIndices());

		auto vertexBuffer = vertexInterop.getMapping();
		auto indexBuffer = indexInterop.getMapping();
		
		{
			// copy vertices with their respective colors into the vertex buffer
			auto dataView = m_halfedges->getDataView();
			insertVertices ToCudaArgs(dataView) (vertices, *m_halfedgeFaces, *m_halfedgeVertices, *m_halfedges, vertexBuffer);
		}

		{
			// copy halfedgeFaces into the index buffer
			auto dataView = m_halfedgeFaces->getDataView();
			insertHalfedgeFaceIndices ToCudaArgs(dataView) (*m_halfedges, *m_halfedgeFaces, indexBuffer);

		}
		unsigned int numTriangles = m_halfedgeFaces->size() / 3;
		return numTriangles;
	}

	struct CountNonManifolds {
		HostDevice
		inline bool operator()(const HalfedgeFace& a) {
			return a.isNonManifold();
		}
	};

	struct CountNonManifoldEdge {
		int desiredAmount = 4;

		HostDevice
		inline bool operator()(int numFaces) {
			return numFaces == desiredAmount;
		}
	};

	int HalfedgeMesh::getNumNonManifoldEdges4() {
		// run a count_if on the hashtable that counts all entries with 4 entries
		auto dataView = m_hashTable->getNumFacesDataView();
		auto counter = CountNonManifoldEdge();
		counter.desiredAmount = 4;
		int amount = thrust::count_if(dataView.begin(), dataView.end(), counter);
		return amount;
	}
	
	int HalfedgeMesh::getNumNonManifoldEdges3() {
		// run a count_if on the hashtable that counts all entries with 3 entries
		auto dataView = m_hashTable->getNumFacesDataView();
		auto counter = CountNonManifoldEdge();
		counter.desiredAmount = 3;
		int amount = thrust::count_if(dataView.begin(), dataView.end(), counter);
		return amount;
	}

	struct CountEdge {

		HostDevice
			inline bool operator()(int numFaces) {
			return numFaces > 0;
		}
	};

	int HalfedgeMesh::getNumEdges() {
		// run a count_if on the hashtable that counts all entries with 3 entries
		auto dataView = m_hashTable->getNumFacesDataView();
		auto counter = CountEdge();
		int amount = thrust::count_if(dataView.begin(), dataView.end(), counter);
		return amount;
	}

	struct CountNonManifoldEdges {
		HalfedgeDeviceVector halfedges;
		HalfedgeDeviceHashTable hashTable;
		int desiredAmount = 0;

		CountNonManifoldEdges(HalfedgeDeviceVector halfedges, HalfedgeDeviceHashTable hashTable)
			: halfedges(halfedges)
			, hashTable(hashTable)
		{}

		__device__
		inline bool operator()(HalfedgeID firstHalfedge) {
			if (firstHalfedge == INVALID_INDEX) return false;

			int numNonManifolds = 0;

			HalfedgeID currentHalfedgeID = firstHalfedge;
			Halfedge currentHalfedge = halfedges[currentHalfedgeID];
			Halfedge nextHalfedge = halfedges[currentHalfedge.getNext()];
			for (int i = 0; i < 4; ++i) {
				int numHalfedges = hashTable.getNumFaces(currentHalfedge.getOriginVertexID(), nextHalfedge.getOriginVertexID());
				if (numHalfedges > 2) {
					++numNonManifolds;
				}

				currentHalfedgeID = currentHalfedge.getNext();
				currentHalfedge = halfedges[currentHalfedgeID];
				nextHalfedge = halfedges[currentHalfedge.getNext()];
			}

			return (numNonManifolds == desiredAmount);
		}
	};

	int HalfedgeMesh::getNumQuadsWithNonManifoldEdges(int numNonManifoldEdges) {
		//refreshHashTable();
		auto dataView = m_halfedgeFaces->getFirstHalfedgeIDDataView();
		auto counter = CountNonManifoldEdges(*m_halfedges, *m_hashTable);
		counter.desiredAmount = numNonManifoldEdges;
		int amount = thrust::count_if(dataView.begin(), dataView.end(), counter);
		return amount;
	}
}