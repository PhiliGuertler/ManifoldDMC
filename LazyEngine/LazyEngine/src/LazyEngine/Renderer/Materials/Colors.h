#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Core/Constants.h"
#include "LazyEngine/platform/CUDA/CUDAUtils.h"
#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>

namespace LazyEngine {

	namespace Color {

		/**
		 *	converts an HSV-color into an RGB-color
		 */
		HostDevice static inline glm::vec3 HSVtoRGB(const glm::vec3& hsv) {
			return glm::rgbColor(hsv);
		}

		/**
		 *	converts an HSV-color into an RGB-color
		 */
		HostDevice static inline glm::vec3 HSVtoRGB(float hue, float saturation, float value) {
			return HSVtoRGB({ hue, saturation, value });
		}

		/**
		 *	converts an RGB-color into an HSV-color
		 */
		HostDevice static inline glm::vec3 RGBtoHSV(const glm::vec3& rgb) {
			return glm::hsvColor(rgb);
		}

		/**
		 *	converts an RGB-color into an HSV-color
		 */
		HostDevice static inline glm::vec3 RGBtoHSV(float red, float green, float blue) {
			return RGBtoHSV({ red, green, blue });
		}

		/**
		 *	returns the luminosity of an RGB color
		 */
		HostDevice static inline float getLuminosityRGB(const glm::vec3& rgb) {
			return glm::luminosity(rgb);
		}

		/**
		 *	returns the luminosity of an RGB color
		 */
		HostDevice static inline float getLuminosityRGB(float red, float green, float blue) {
			return getLuminosityRGB({ red, green, blue });
		}

		// TODO: extend using sRGB, HSL, etc.
	}

}