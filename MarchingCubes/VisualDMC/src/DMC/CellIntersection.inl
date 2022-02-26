#pragma once

#include "CellIntersection.h"
#include "Utils/Interpolation.h"

namespace DMC {

	// ##################################################################### //
	// ### VertexToSurfaceIterator ######################################### //
	// ##################################################################### //

	__device__ inline VertexToSurfaceIterator::VertexToSurfaceIterator(float isoValue, const float* scalars, float step)
		: m_numMaxIterations(30)
		, m_scalars(scalars)
		, m_isoValue(isoValue)
		, m_step(step)
		, m_enablePrints(false)
	{
		// empty
	}

	__device__ inline void VertexToSurfaceIterator::movePointToSurface(glm::vec3& point) {
		glm::vec3 interpolation = point;
		float value1 = Interpolation<float>::interpolateTrilinearly(m_scalars, interpolation);
		float value2 = 0.f;
		for (int i = 0; i < m_numMaxIterations; ++i) {
			glm::vec3 gradient = computeGradient(interpolation);
			if (value1 > m_isoValue) {
				// invert the gradient
				gradient = -gradient;
			}

			if (m_enablePrints) {
				printf("[%d]: Gradient: {%.2f,%.2f,%.2f}, Point: {%.2f,%.2f,%.2f}\n", i, VECCC(gradient), VECCC(point));
			}

			glm::vec3 interpolation2 = interpolation;

			auto movementResult = updateGradient(point, gradient, interpolation, interpolation2, value1, value2);
			if (m_enablePrints) {
				printf("interpolations: {%.2f,%.2f,%.2f}, {%.2f,%.2f,%.2f}\n", VECCC(interpolation), VECCC(interpolation2));
			}

			if (movementResult.breakNow) {
				break;
			}

			if (movementResult.xClamped || movementResult.yClamped || movementResult.zClamped) {
				// as we have reached at least one wall, stop walking into it and instead move perpendicular.
				if (movementResult.xClamped) {
					gradient.x = 0.f;
				}
				else if (movementResult.yClamped) {
					gradient.y = 0.f;
				}
				else if (movementResult.zClamped) {
					gradient.z = 0.f;
				}

				auto secondMovement = updateGradient(point, gradient, interpolation, interpolation2, value1, value2);
				if (secondMovement.breakNow) {
					break;
				}
			}

			interpolation = interpolation2;
			value1 = value2;
		}

		// clamp the interpolation to stay inside of the cell!
		// Seems like this version of the vertex iterator doesn't ensure this behaviour!
		point = glm::clamp(point, glm::vec3(LazyEngine::Constants::EPSILON), glm::vec3(1.f - LazyEngine::Constants::EPSILON));
	}

	__device__ inline glm::vec3 VertexToSurfaceIterator::computeGradient(const glm::vec3& uvw) {
		glm::vec3 gradient;
		gradient.x = (1 - uvw.z) * ((1 - uvw.y) * (m_scalars[1] - m_scalars[0]) + uvw.y * (m_scalars[3] - m_scalars[2])) + uvw.z * ((1 - uvw.y) * (m_scalars[5] - m_scalars[4]) + uvw.y * (m_scalars[7] - m_scalars[6]));
		gradient.y = (1 - uvw.z) * ((1 - uvw.x) * (m_scalars[2] - m_scalars[0]) + uvw.x * (m_scalars[3] - m_scalars[1])) + uvw.z * ((1 - uvw.x) * (m_scalars[6] - m_scalars[4]) + uvw.x * (m_scalars[7] - m_scalars[5]));
		gradient.z = (1 - uvw.y) * ((1 - uvw.x) * (m_scalars[4] - m_scalars[0]) + uvw.x * (m_scalars[5] - m_scalars[1])) + uvw.y * ((1 - uvw.x) * (m_scalars[6] - m_scalars[2]) + uvw.x * (m_scalars[7] - m_scalars[3]));
		return gradient;
	}

	__device__ inline GradientShiftResult VertexToSurfaceIterator::updateGradient(
		glm::vec3& point,
		glm::vec3& gradient,
		glm::vec3& interpolation,
		glm::vec3& interpolation2,
		float& value1,
		float& value2
	) {
		GradientShiftResult result;

		// normalize the gradient
		gradient = glm::normalize(gradient);
		interpolation2 = interpolation + m_step * gradient;

		// check if we are within the cell
		result.xClamped = interpolation2.x <= 0.f || interpolation2.x >= 1.f;
		result.yClamped = interpolation2.y <= 0.f || interpolation2.y >= 1.f;
		result.zClamped = interpolation2.z <= 0.f || interpolation2.z >= 1.f;

		// clamp the interpolation factors to be inside of the cell
		interpolation2 = glm::clamp(interpolation2, glm::vec3(0.f), glm::vec3(1.f));

		// interpolate trilinearly
		value2 = Interpolation<float>::interpolateTrilinearly(m_scalars, interpolation2);
		if ((value1 <= m_isoValue && m_isoValue <= value2) || (value2 <= m_isoValue && m_isoValue <= value1)) {
			// isoValue is between value1 and value2
			float e = value1;
			if (value1 != value2) {
				e = (m_isoValue - value1) / (value2 - value1);
			}

			// move the input point
			point = interpolation + e * (interpolation2 - interpolation);
			result.breakNow = true;
		}
		else {
			result.breakNow = false;
		}
		return result;
	}


	// ##################################################################### //
	// ### CellIntersection ################################################ //
	// ##################################################################### //

	__device__ inline CellIntersection::CellIntersection(
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
	)
		: m_isoValue(isoValue)
		, m_intersectionCase(intersectionCase)
		, m_index3D(index3D)
		, m_grid(grid)
		, m_hashTable(hashTable)
		, m_vertices(vertices)
		, m_scalars(scalars)
		, m_polygonPatterns(polygonPatterns)
		, m_triangleAmbiguities(triangleAmbiguities)
		, m_MC_AMBIGUOUS(MC_AMBIGUOUS)
		, m_numMaxIterations(30)
		, m_enableSurfaceToVertexIteration(enableSurfaceToVertexIteration)
	{
		// empty
		// Print Scalars that contain nonsensical data
		for (int i = 0; i < 8; ++i) {
			if (scalars[i] != scalars[i]) {
				printf("Scalar %d is NAN\n", i);
			}
		}
	}

	__device__
	inline int CellIntersection::computeGlobalIndex(int edgeIndex) {
		const unsigned long long magicPattern = 670526590282893600ULL;
		glm::ivec3 newIndex3D = m_index3D;
		newIndex3D.x += (int)((magicPattern >> (5 * edgeIndex)) & 1);
		newIndex3D.y += (int)((magicPattern >> (5 * edgeIndex + 1)) & 1);
		newIndex3D.z += (int)((magicPattern >> (5 * edgeIndex + 2)) & 1);
		const int offs = (int)((magicPattern >> (5 * edgeIndex + 3)) & 3);
		return (3 * m_grid.getGlobalIndex(newIndex3D) + offs);
	}


	__device__
	inline glm::vec3 CellIntersection::computeLocalCoordinateOffset(EdgeVertices edge, int edgeIndex) {
		int vertexID0 = edge.getVertexID0();
		int vertexID1 = edge.getVertexID1();
		float interpolation = (m_isoValue - m_scalars[vertexID0]) / (m_scalars[vertexID1] - m_scalars[vertexID0]);
		// These are some magic numbers for quick lookups and bitshifts
		const unsigned long long int magicE = 75059404284193ULL;
		const unsigned long long magicC = 38552806359568ULL;

		glm::vec3 result = glm::vec3(0.f, 0.f, 0.f);

		float value = retrieveBit(magicE, 4 * edgeIndex);
		float contour = retrieveBit(magicC, 4 * edgeIndex);
		result.x += interpolation * value + contour;

		value = retrieveBit(magicE, 4 * edgeIndex + 1);
		contour = retrieveBit(magicC, 4 * edgeIndex + 1);
		result.y += interpolation * value + contour;

		value = retrieveBit(magicE, 4 * edgeIndex + 2);
		contour = retrieveBit(magicC, 4 * edgeIndex + 2);
		result.z += interpolation * value + contour;

		return result;
	}

	__device__
	inline float CellIntersection::asymptoticDecider(float a, float b, float c, float d) {
		return (a * d - b * c) / (a + d - b - c);
	}

	__device__
	inline float CellIntersection::asymptoticDecider(float values[4]) {
		return asymptoticDecider(values[0], values[1], values[2], values[3]);
	}

	__device__
	inline void CellIntersection::handleAsymptoticFaceCase(int faceCase, ContourSegment segments[12], int edges[4], float scalarValues[4]) {
		float decision = 0.f;

		switch (faceCase) {
		case 1:
			ContourSegment::connect(segments[edges[0]], segments[edges[3]]);
			break;
		case 2:
			ContourSegment::connect(segments[edges[1]], segments[edges[0]]);
			break;
		case 3:
			ContourSegment::connect(segments[edges[1]], segments[edges[3]]);
			break;
		case 4:
			ContourSegment::connect(segments[edges[3]], segments[edges[2]]);
			break;
		case 5:
			ContourSegment::connect(segments[edges[0]], segments[edges[2]]);
			break;
		case 7:
			ContourSegment::connect(segments[edges[1]], segments[edges[2]]);
			break;
		case 8:
			ContourSegment::connect(segments[edges[2]], segments[edges[1]]);
			break;
		case 10:
			ContourSegment::connect(segments[edges[2]], segments[edges[0]]);
			break;
		case 11:
			ContourSegment::connect(segments[edges[2]], segments[edges[3]]);
			break;
		case 12:
			ContourSegment::connect(segments[edges[3]], segments[edges[1]]);
			break;
		case 13:
			ContourSegment::connect(segments[edges[0]], segments[edges[1]]);
			break;
		case 14:
			ContourSegment::connect(segments[edges[3]], segments[edges[0]]);
			break;
		case 6:
			decision = asymptoticDecider(scalarValues);
			if (decision >= m_isoValue) {
				ContourSegment::connect(segments[edges[3]], segments[edges[0]]);
				ContourSegment::connect(segments[edges[1]], segments[edges[2]]);
			}
			else {
				ContourSegment::connect(segments[edges[1]], segments[edges[0]]);
				ContourSegment::connect(segments[edges[3]], segments[edges[2]]);
			}
			break;
		case 9:
			decision = asymptoticDecider(scalarValues);
			if (decision >= m_isoValue) {
				ContourSegment::connect(segments[edges[0]], segments[edges[1]]);
				ContourSegment::connect(segments[edges[2]], segments[edges[3]]);
			}
			else {
				ContourSegment::connect(segments[edges[0]], segments[edges[3]]);
				ContourSegment::connect(segments[edges[2]], segments[edges[1]]);
			}
			break;
		default:
			break;
		}
	}

	__device__
	inline int CellIntersection::marchingCubesAsymptoticPolygon(Contour& contour) {
		// compute oriented contours by first building segments, then connecting them
		ContourSegment segments[12] = { 0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11 };

		// "In order to compute oriented segments, the hexahedron has to be flatten.
		// The insides of the faces of the hexahedron have to be all at the same
		// side of the flattend hexa. This requires changing the order of the
		// edges when reading from the faces"
		// code edges at face
		unsigned short faceEdgeLookup[6]{ 0x0123, 0x4765, 0x4908, 0x2A6B, 0x83B7, 0x95A1 };
		// code vertices at face
		unsigned short faceVertexLookup[6]{ 0x3120, 0x6475, 0x1504, 0x7362, 0x2064, 0x7531 };

		for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
			// classify the face
			unsigned int faceCase = 0;

			int vertices[4];
			int edges[4];
			float scalarValues[4];

			for (int i = 0; i < 4; ++i) {
				vertices[i] = retrieveHalfByte(faceVertexLookup[faceIndex], i);
				edges[i] = retrieveHalfByte(faceEdgeLookup[faceIndex], i);
				scalarValues[i] = m_scalars[vertices[i]];

				faceCase |= (scalarValues[i] >= m_isoValue) << i;
			}

			handleAsymptoticFaceCase(faceCase, segments, edges, scalarValues);
		}

		// Connect oriented segments into oriented contours
		int numberOfContours = 0;
		for (unsigned int edge = 0; edge < 12; ++edge) {
			ContourSegment currentSegment = segments[edge];
			if (currentSegment.isSegmentSet()) {
				int edgeOut = currentSegment.getOutgoingSegment();
				int edgeIn = currentSegment.getIncomingSegment();
				unsigned int edgeStart = edge;
				unsigned int position = 0;
				contour.setContourEdge(numberOfContours, position, edgeStart);
				while (edgeOut != edgeStart) {
					// Iterate over the contour segments and update the contour
					++position;
					contour.setContourEdge(numberOfContours, position, edgeOut);
					edgeIn = edgeOut;
					edgeOut = segments[edgeIn].getOutgoingSegment();
					segments[edgeIn].reset();
				}
				// set the contour length
				contour.setContourSize(numberOfContours, position + 1);
				++numberOfContours;
			}
		}

		return numberOfContours;
	}

	__device__
	inline int CellIntersection::marchingCubesSimplePolygon(Contour& contour) {
		int currentContour = 0;
		// get the line of the current case from the plygonPatterns lookup table
		const char* polygonCase = &m_polygonPatterns[17 * m_intersectionCase];
		unsigned char position2 = polygonCase[0] + 1;
		// loop over contours
		for (int c = 1; c <= polygonCase[0]; ++c) {
			char polygon = polygonCase[c];
			// set polygon size
			contour.setContourSize(currentContour, polygon);
			// save the vertex for all these edges
			for (int i = 0; i < polygon; ++i) {
				const unsigned int edge = polygonCase[position2++];
				contour.setContourEdge(currentContour, i, edge);
			}
			currentContour++;
		}
		return currentContour;
	}

	__device__
	inline int CellIntersection::computeVertexPosition(int edgeIndex, int offset) {
		const unsigned long long e_table = 240177437832960ULL;
		// retrieve the lowest two bits of a specific table position
		return (e_table >> (4 * edgeIndex + 2 * offset)) & 3;
	}

	__device__ inline void CellIntersection::sliceP() {
		Contour contourSize;
		unsigned int contour = 0;
		if (m_triangleAmbiguities[m_intersectionCase] == m_MC_AMBIGUOUS) {
			// create the polygon for ambiguous cases using an asymptotic decider
			//printf("An asymptotic case arised!\n");
			contour = marchingCubesAsymptoticPolygon(contourSize);
		}
		else {
			// create the polygon in the simple case
			contour = marchingCubesSimplePolygon(contourSize);
		}


		// lists edges
		EdgeVertices edges[12]{ 16, 49, 50, 32, 84, 117, 118, 100, 64, 81, 115, 98 };

		// compute normals at cell corners
		unsigned short edgy = 0;
		glm::vec3 normals[8];
		m_grid.computeGradients(normals, m_scalars, m_index3D);

		// Compute the actual vertices
		for (int t = 0; t < contour; ++t) {
			glm::vec3 vertexPosition = glm::vec3(0.f, 0.f, 0.f);
			int currentContourSize = contourSize.getContourSize(t);
			for (int i = 0; i < currentContourSize; ++i) {
				// compute the new vertex position depending on the contour
				const unsigned int edgeIndex = contourSize.getContourEdge(t, i);
				vertexPosition += computeLocalCoordinateOffset(edges[edgeIndex], edgeIndex);

				// "set edge case to construct oriented quadilateral"
				if (m_scalars[edges[edgeIndex].getVertexID0()] < m_isoValue) {
					edgy |= (1 << edgeIndex);
				}
			}

			// normalize vertex position
			vertexPosition /= static_cast<float>(currentContourSize);

			// move the vertex to the isosurface
			if (m_enableSurfaceToVertexIteration) {
				VertexToSurfaceIterator pointMover(m_isoValue, m_scalars);
				pointMover.movePointToSurface(vertexPosition);
			}

			// compute the normal at the mesh vertex
			glm::vec3 normal = Interpolation<glm::vec3>::interpolateTrilinearly(normals, vertexPosition);
			normal = glm::normalize(normal);

			// compute the point in world space
			vertexPosition = m_grid.getOrigin() + (glm::vec3(m_index3D) + vertexPosition) * m_grid.getDeltas();

			int vertexAddress = m_vertices.addVertex(vertexPosition, normal);

			// "for all these edges save the vertex"
			for (int i = 0; i < currentContourSize; ++i) {
				const unsigned int edgeIndex = contourSize.getContourEdge(t, i);
				int color = -1;
				const int baseColor = ((m_index3D.x & 1) | (m_index3D.y & 1) << 1 | (m_index3D.z & 1) << 2);
				if (edgeIndex == 0) color = 3 * baseColor;
				if (edgeIndex == 3) color = 3 * baseColor + 1;
				if (edgeIndex == 8) color = 3 * baseColor + 2;

				// compute unique edge id
				const int globalId = computeGlobalIndex(edgeIndex);
				const int position = computeVertexPosition(edgeIndex, (edgy >> edgeIndex) & 1);

				m_hashTable.addVertex(globalId, position, vertexAddress, color);
			}
		}
	}
}