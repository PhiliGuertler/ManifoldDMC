#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Renderer/Shader.h"

namespace LazyEngine {

	class Material {
	public:
		Material() = default;
		virtual ~Material() = default;

		/**
		 *	returns the shader that is representing this material
		 */
		virtual Ref<Shader> getShader() const = 0;

		/**
		 *	updates this material's shader's uniforms
		 */
		virtual void updateUniforms() = 0;

	};

}