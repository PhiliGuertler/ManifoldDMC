#pragma once

#include "Constants.h"

namespace LazyEngine {

	namespace Util {

		template <typename T>
		inline int sgn(T val) {
			return (T(0) < val) - (val < T(0));
		}

		template <typename T>
		inline T InchesToCentimeters(T inches) {
			constexpr T InchesToCentimeters = static_cast<T>(2.54);
			return inches * InchesToCentimeters;
		}

		template <typename T>
		inline T CentimetersToInches(T centimeters) {
			constexpr T CentimetersToInches = static_cast<T>(1)/static_cast<T>(2.54);
			return centimeters * CENTIMETERS_TO_INCHES;
		}

		template <typename T>
		inline T radiansToDegrees(T radians) {
			constexpr T RadToDeg = static_cast<T>(180) / static_cast<T>(LazyEngine::Constants::PI);
			return RadToDeg * radians;
		}

		template <typename T>
		inline T degreesToRadians(T degrees) {
			constexpr T DegToRad = static_cast<T>(LazyEngine::Constants::PI) / static_cast<T>(180);
			return DegToRad * degrees;
		}
	}

}