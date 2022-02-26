#include "Interpolation.h"

#include "../Mesh.h"

namespace DMC {

	template <typename T>
	HostDevice inline T Interpolation<T>::interpolateLinearly(const T& a, const T& b, float factor) {
		return glm::mix(a, b, factor);
	}

	/**
	 *	specialized for floats, as glm::mix doesn't support them
	 */
	template <>
	HostDevice inline float Interpolation<float>::interpolateLinearly(const float& a, const float& b, float factor) {
		return a * (1.f - factor) + b * factor;
	}

	/**
	 *	specialized for doubles, as glm::mix doesn't support them
	 */
	template <>
	HostDevice inline double Interpolation<double>::interpolateLinearly(const double& a, const double& b, float factor) {
		return a * static_cast<double>(1.0 - factor) + b * static_cast<double>(factor);
	}

	/**
	 *	specialized for Vertices
	 */
	template <>
	HostDevice inline Vertex Interpolation<Vertex>::interpolateLinearly(const Vertex& a, const Vertex& b, float factor) {
		Vertex result;
		result.position = glm::mix(a.position, b.position, factor);
		result.normal = glm::mix(a.normal, b.normal, factor);
		result.color = glm::mix(a.color, b.color, factor);
		return result;
	}

	template <typename T>
	HostDevice inline T Interpolation<T>::interpolateBilinearly(const T values[4], const glm::vec2& uv) {
		T a = interpolateLinearly(values[0], values[1], uv.x);
		T b = interpolateLinearly(values[3], values[2], uv.x);
		return interpolateLinearly(a, b, uv.y);
	}

	template <typename T>
	HostDevice inline T Interpolation<T>::interpolateTrilinearly(const T values[8], const glm::vec3& uvw) {
		return (1.f - uvw.z) * (
			(1.f - uvw.y) * (values[0] + uvw.x * (values[1] - values[0])) +
			uvw.y * (values[2] + uvw.x * (values[3] - values[2]))
			) + uvw.z * (
				(1.f - uvw.y) * (values[4] + uvw.x * (values[5] - values[4])) +
				uvw.y * (values[6] + uvw.x * (values[7] - values[6]))
				);
	}

}