#pragma once

#include <LazyEngine/LazyEngine.h>

namespace DMC {

	template <typename T>
	class Interpolation {
	public:
		/**
		 *	Interpolates linearly between two values
		 */
		HostDevice static T interpolateLinearly(const T& a, const T& b, float factor);

		/**
		 *	Interpolates bilinearly on a quad given by four values	
		 */
		HostDevice static T interpolateBilinearly(const T values[4], const glm::vec2& uv);

		/**
		 *	Interpolates trilinearly in a cube given by eight values
		 */
		HostDevice static T interpolateTrilinearly(const T values[8], const glm::vec3& uvw);

	};

}

#include "Interpolation.inl"