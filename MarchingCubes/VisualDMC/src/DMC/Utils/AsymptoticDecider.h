#pragma once

#include <LazyEngine/LazyEngine.h>
#include "Interpolation.h"

namespace DMC {

	struct AsymptoticIntersection {
		glm::vec2 uv;
		float value;
	};

	class AsymptoticDecider2D {
	public:
		/**
		 *	constructor
		 *	@param u0v0: the iso-value at point uv = {0,0}
		 *	@param u1v0: the iso-value at point uv = {1,0}
		 *	@param u1v1: the iso-value at point uv = {1,1}
		 *	@param u0v1: the iso-value at point uv = {0,1}
		 */
		HostDevice inline AsymptoticDecider2D(float u0v0, float u1v0, float u1v1, float u0v1)
			: m_00(u0v0)
			, m_10(u1v0)
			, m_11(u1v1)
			, m_01(u0v1)
		{
			// empty
		}

		/**
		 *	Computes the value at the point where the asymptotes intersect.
		 */
		HostDevice inline AsymptoticIntersection computeAsymptoticIntersection() {
			const float A = m_10 - m_00;
			const float B = m_01 - m_00;
			const float C = m_00 - m_10 - m_01 + m_11;
			const float delta = (m_00 * m_11 - m_10 * m_01) / C;
			glm::vec2 uv = { -B / C, -A / C };
			return { uv, delta };
		}

		/**
		 *	computes the quadrant of a uv-point.
		 */
		HostDevice inline int computeQuadrantOfPoint(const glm::vec2& uv, const glm::vec2& asymptoteIntersection) {
			bool isToTheRight = uv.x > asymptoteIntersection.x;
			bool isToTheTop = uv.y > asymptoteIntersection.y;
			int result = 0;
			result |= (isToTheRight ? 1 : 0);
			result |= (isToTheTop ? 1 : 0) << 1;
			return result;
		}

		/**
		 *	returns true if both points are in the same quadrant.
		 */
		HostDevice inline bool pointsBelongTogether(const glm::vec2& uv0, const glm::vec2& uv1) {
			AsymptoticIntersection intersection = computeAsymptoticIntersection();
			// check if both uv coordinates are in the same part of the domain, relative to the intersection point.
			int quadrant0 = computeQuadrantOfPoint(uv0, intersection.uv);
			int quadrant1 = computeQuadrantOfPoint(uv1, intersection.uv);

			return quadrant0 == quadrant1;
		}

		HostDevice inline bool pointsBelongTogetherV2(const glm::vec2& uv0, const glm::vec2& uv1) {
			// don't check the intersections points themselves, but instead two points that are inset a bit
			glm::vec2 interpol0 = Interpolation<glm::vec2>::interpolateLinearly(uv0, uv1, 0.1f);
			glm::vec2 interpol1 = Interpolation<glm::vec2>::interpolateLinearly(uv1, uv0, 0.1f);

			return pointsBelongTogether(interpol0, interpol1);
		}

		HostDevice inline bool quadrant0BelongsTogether(float isoValue) {
			AsymptoticIntersection intersection = computeAsymptoticIntersection();

			return (m_00 < isoValue && intersection.value >= isoValue) || (m_00 >= isoValue && intersection.value < isoValue);
		}

	protected:
		float m_00, m_10, m_11, m_01;
	};

}