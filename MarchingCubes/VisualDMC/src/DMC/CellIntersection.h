#pragma once

#include <LazyEngine/LazyEngine.h>
#include "UniformGrid.h"
#include "DeviceHashTables.h"
#include "DeviceVectors.h"

namespace DMC {

	/**
	 *	returns the value of the bitIndex'th bit in input [0-1]
	 */
	__device__ inline int retrieveBit(unsigned long long input, int bitIndex) {
		return (input & (1ULL << (bitIndex))) >> (bitIndex);
	}

	/**
	 *	returns the value of the halfByteIndex'th half-byte in input [0-15]
	 */
	__device__ inline int retrieveHalfByte(unsigned short input, int halfByteIndex) {
		return (input >> (4 * halfByteIndex)) & 0xF;
	}


	/**
	 *	Defines the Contour of the iso-surface on the cell-walls
	 */
	class Contour {
	public:
		/**
		 *	constructor
		 */
		__host__ __device__ inline Contour() : m_contourSize(0xFFFFFFFFFFFF0000ULL)
		{
			// empty
		}

		/**
		 *	destructor
		 */
		__host__ __device__ inline ~Contour() = default;

		/**
		 *	Sets the half-byte (4 bits) at a given index to 0x0.
		 *  @param index: The index of the half-byte [0-15]
		 */
		__host__ __device__ inline void unsetHalfByte(int index) {
			m_contourSize &= ~(0xF << (4 * index));
		}

		/**
		 *	Sets the half-byte (4 bits) at a given index to @param value.
		 *  @param index: The index of the half-byte [0-15]
		 *  @param value: The value that should be entered [0-15].
		 */
		__host__ __device__ inline void setHalfByte(int index, int value) {
			unsetHalfByte(index);
			m_contourSize |= (value << (4 * index));
		}

		/**
		 *	Returns the half-byte (4 bits) at a given index
		 *  @param index: The index of the half-byte [0-15]
		 *  @returns The value of the half-byte (0-15)
		 */
		__host__ __device__ inline int getHalfByte(int index) {
			return static_cast<int>((m_contourSize & (0xF << 4 * index)) >> 4 * index);
		}

		/**
		 *	Sets the size of the marching cubes polygon, apparently
		 *  @param contour: apparently the index (0-15) of a contour
		 *  @param size: the size that should be stored for that contour
		 */
		__host__ __device__ inline void setContourSize(int contour, int size) {
			setHalfByte(contour, size);
		}

		/**
		 *	Returns a contour size for a given contour index (0-15)
		 *  @param contour: the index of the contour whose size should be returned
		 *  @returns the size stored at that contour
		 */
		__host__ __device__ inline int getContourSize(int contour) {
			return getHalfByte(contour);
		}

		/**
		 *	Computes a magic edge index?
		 */
		__host__ __device__ inline int computeEdgeIndex(int contour, int position) {
			const unsigned int mask[4] = { 0x0, 0xF, 0xFF, 0xFFF };
			const unsigned int contourSize = static_cast<unsigned int>(m_contourSize & mask[contour]);
			return 16 + 4 * (
				(contourSize & 0xF) +
				((contourSize & 0xF0) >> 4) +
				((contourSize & 0xF00) >> 8) +
				position);
		}

		/**
		 *	Sets a contour edge?
		 */
		__host__ __device__ inline void setContourEdge(int contour, int position, int value) {
			int edgeIndex = computeEdgeIndex(contour, position);

			// unset the half-byte at edge's position in m_contourSize
			m_contourSize &= ~(static_cast<unsigned long long>(0xF) << edgeIndex);
			// set the new value in the half-byte
			m_contourSize |= (static_cast<unsigned long long>(value) << edgeIndex);
		}

		/**
		 *	Gets a contour edge?
		 */
		__host__ __device__ inline int getContourEdge(int contour, int position) {
			int edgeIndex = computeEdgeIndex(contour, position);
			// is this correct?
			//return getHalfByte(edgeIndex >> 2);

			return static_cast<int>((m_contourSize >> edgeIndex) & 0xF);
		}

		/**
		 *	casts the contour to its underlying unsigned long long
		 */
		__host__ __device__ inline operator unsigned long long() const {
			return m_contourSize;
		}

	protected:
		// it is effectively used as an array of 16 4-bit wide values.
		// Layout:
		//	- Half-Bytes 0-3: The Sizes (0-15) of the contours (0x0000 by default)
		//	- Half-Bytes 4-15: The edge-indices making up the contours (0-12), where 0xF is an invalid edgeIndex
		unsigned long long m_contourSize;
	};

	/**
	 *	Describes a Contour Segment
	 */
	class ContourSegment {
	public:
		__host__ __device__ static inline void connect(ContourSegment& in, ContourSegment& out) {
			in.setOutgoingSegment(out);
			out.setIncomingSegment(in);
		}

		/**
		 *	constructor
		 */
		__host__ __device__ inline ContourSegment(unsigned char index)
			: m_segment(0xFF)
			, m_index(index)
		{
			// empty
		}

		__host__ __device__ inline ~ContourSegment() = default;

		/**
		 *	Sets the incoming contour segment's index
		 */
		__host__ __device__ inline void setIncomingSegment(ContourSegment& incoming) {
			// clear this segment's high half-byte
			m_segment &= 0xF;
			// set the high half-byte of this segment to incoming's index
			m_segment |= (incoming.m_index << 4);
		}

		/**
		 *	Sets the outgoing contour segment's index
		 */
		__host__ __device__ inline void setOutgoingSegment(ContourSegment& outgoing) {
			// clear this segment's low half-byte
			m_segment &= 0xF0;
			// set the low half-byte of this segment to outgoing's index
			m_segment |= (outgoing.m_index & 0xF);
		}

		/**
		 *	Returns the index of the incoming segment
		 */
		__host__ __device__ inline int getIncomingSegment() {
			return static_cast<int>((m_segment >> 4) & 0xF);
		}

		/**
		 *	Returns the index of the outgoing segment
		 */
		__host__ __device__ inline int getOutgoingSegment() {
			return static_cast<int>(m_segment & 0xF);
		}

		/**
		 *	returns true, if the segment is set
		 */
		__host__ __device__ inline bool isSegmentSet() const {
			return m_segment != 0xFF;
		}

		/**
		 *	resets this segment to an invalid one
		 */
		__host__ __device__ inline void reset() {
			m_segment = 0xFF;
		}


	protected:
		// A single byte that stores two values:
		// - The high half-byte stores the index of the corresponding outgoing [something]
		// - The low half-byte stores the index of the corresponding incoming [something]
		unsigned char m_segment;
		// The index of this Contour Segment
		unsigned char m_index;
	};

	/**
	 *	Encodes two vertices of a Cell-Intersection
	 */
	class EdgeVertices {
	public:
		/**
		 *	constructor
		 */
		__host__ __device__ inline EdgeVertices(unsigned char encodedVertices)
			: m_encodedVertices(encodedVertices)
		{
			// empty
		}

		/**
		 *	destructor
		 */
		__host__ __device__ inline ~EdgeVertices() = default;

		/**
		 *	Returns the index of vertex-0
		 */
		__host__ __device__ inline int getVertexID0() const {
			return m_encodedVertices & 0xF;
		}

		/**
		 *	Returns the index of vertex-1
		 */
		__host__ __device__ inline int getVertexID1() const {
			return (m_encodedVertices >> 4) & 0xF;
		}

	protected:
		// encodes the indices of the two vertices that make up this edge.
		// One vertex index is stored in the high and the other the low half-byte
		const unsigned char m_encodedVertices;
	};

	struct GradientShiftResult {
		bool breakNow = false;
		bool xClamped = false;
		bool yClamped = false;
		bool zClamped = false;
	};

	/**
	 *	Moves a Vertex to the surface of its cell iteratively
	 */
	class VertexToSurfaceIterator {
	public:
		__device__
		inline VertexToSurfaceIterator(float isoValue, const float *scalars, float step = 0.05f);
		
		__device__
		inline void movePointToSurface(glm::vec3& point);

		/**
		 *	Computes the gradient at a point in the cell given by the three trilinear factors uvw
		 */
		__device__
		inline glm::vec3 computeGradient(const glm::vec3& uvw);

		/**
		 *	Sets the maximum amount of iterations that will be performed during the point movement step.
		 */
		__device__
		inline void setNumMaxIterations(int numIterations) {
			m_numMaxIterations = numIterations;
		}

		__device__
		inline GradientShiftResult updateGradient(
			glm::vec3& point,
			glm::vec3& gradient,
			glm::vec3& interpolation,
			glm::vec3& interpolation2,
			float& value1,
			float& value2);

		__device__
		inline void setEnablePrints(bool value) {
			m_enablePrints = value;
		}

	protected:
		int m_numMaxIterations;
		const float *m_scalars;
		float m_isoValue;
		float m_step;

		bool m_enablePrints;
	};

	/**
	 *	Takes care of the intersection of a cell and the iso-surface.
	 */
	class CellIntersection {
	protected:
		float m_isoValue;
		int m_intersectionCase;
		glm::ivec3 m_index3D;
		const float *m_scalars;
		const char *m_polygonPatterns;
		const char *m_triangleAmbiguities;
		char m_MC_AMBIGUOUS;

		int m_numMaxIterations;

		bool m_enableSurfaceToVertexIteration;

		DMC::UniformGrid<float>& m_grid;
		DMC::QuadrilateralDeviceHashTable& m_hashTable;
		DMC::VertexDeviceVector& m_vertices;

	public:

		__device__ inline CellIntersection(
			float isoValue,
			int intersectionCase,
			glm::ivec3 index3D,
			DMC::UniformGrid<float>& grid,
			DMC::QuadrilateralDeviceHashTable& hashTable,
			DMC::VertexDeviceVector& vertices,
			float scalars[8],
			char polygonPatterns[],
			char triangleAmbiguities[],
			char MC_AMBIGUOUS,
			bool enableSurfaceToVertexIteration
		);

		/**
		 *	Intersects the cell with the isosurface and generates vertex representatives
		 */
		__device__ inline void sliceP();

		__device__ inline void setNumMoveIterations(int numIterations) {
			m_numMaxIterations = numIterations;
		}

	protected:
		/**
		 *	computes the global index of an edge of the cell
		 */
		__device__ inline int computeGlobalIndex(int edgeIndex);
	
		/**
		 *	computes a local coordinate offset?
		 */
		__device__ inline glm::vec3 computeLocalCoordinateOffset(EdgeVertices edge, int edgeIndex);

		__device__ inline float asymptoticDecider(float a, float b, float c, float d);
		
		__device__ inline float asymptoticDecider(float values[4]);

		__device__ inline void handleAsymptoticFaceCase(int faceCase, ContourSegment segments[12], int edges[4], float scalarValues[4]);

		/**
		 *	computes the marching cubes polygon using an asymptotic decider.
		 */
		__device__ inline int marchingCubesAsymptoticPolygon(Contour& contour);

		/**
		 *	computes the marching cubes polygon using a simple direct approach.
		 */
		__device__ inline int marchingCubesSimplePolygon(Contour& contour);
	
		__device__ inline int computeVertexPosition(int edgeIndex, int offset);
	
	};

}

#include "CellIntersection.inl"