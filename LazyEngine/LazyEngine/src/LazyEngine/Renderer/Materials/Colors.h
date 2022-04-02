#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Core/Constants.h"
#include "LazyEngine/platform/CUDA/CUDAUtils.h"
#include <glm/glm.hpp>

namespace LazyEngine {

	namespace Color {

		/**
		 *	converts an HSV-color into an RGB-color
		 */
		HostDevice static inline glm::vec3 HSVtoRGB(const glm::vec3& hsv) {
			float r, g, b;

			float f = hsv.x / 60.0f;
			float hi = floorf(f);
			f = f - hi;
			float p = hsv.z * (1 - hsv.y);
			float q = hsv.z * (1 - hsv.y * f);
			float t = hsv.z * (1 - hsv.y * (1 - f));

			if (hi == 0.0f || hi == 6.0f) {
				r = hsv.z;
				g = t;
				b = p;
			}
			else if (hi == 1.0f) {
				r = q;
				g = hsv.z;
				b = p;
			}
			else if (hi == 2.0f) {
				r = p;
				g = hsv.z;
				b = t;
			}
			else if (hi == 3.0f) {
				r = p;
				g = q;
				b = hsv.z;
			}
			else if (hi == 4.0f) {
				r = t;
				g = p;
				b = hsv.z;
			}
			else {
				r = hsv.z;
				g = p;
				b = q;
			}
			return glm::vec3(r, g, b);

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
			glm::vec3 out;

			float min = rgb.r < rgb.g ? rgb.r : rgb.g;
			min = min < rgb.b ? min : rgb.b;

			float max = rgb.r > rgb.g ? rgb.r : rgb.g;
			max = max > rgb.b ? max : rgb.b;

			out.z = max;                                // v
			float delta = max - min;
			if (delta < 0.00001)
			{
				out.y = 0;
				out.x = 0; // undefined, maybe nan?
				return out;
			}
			if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
				out.y = (delta / max);                  // s
			}
			else {
				// if max is 0, then r = g = b = 0              
				// s = 0, h is undefined
				out.y = 0.0;
				out.x = NAN;                            // its now undefined
				return out;
			}
			if (rgb.r >= max) {                           // > is bogus, just keeps compilor happy
				out.x = (rgb.g - rgb.b) / delta;        // between yellow & magenta
			} else {
				if (rgb.g >= max) {
					out.x = 2.f + (rgb.b - rgb.r) / delta;  // between cyan & yellow
				}
				else {
					out.x = 4.f + (rgb.r - rgb.g) / delta;  // between magenta & cyan
				}
			}

			out.x *= 60.f;                              // degrees

			if (out.x < 0.f) {
				out.x += 360.f;
			}

			return out;
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
			return 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
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