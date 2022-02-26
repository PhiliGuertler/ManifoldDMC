// ######################################################################### //
// ### GeometricPrimitives.cpp ############################################# //
// ### Defines geometric primitives like quads and vertices              ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"

#include "GeometricPrimitives.h"

#include "glm/gtc/matrix_transform.hpp"

#include "LazyEngine/Profiling/Profiler.h"
#include "LazyEngine/Core/Constants.h"

namespace LazyEngine {


	// ######################################################################## //
	// ### Quad ############################################################### //
	// ######################################################################## //

	// defines the vertices of a quad without a transform. It has a width and height of 1 and is centered in the origin.
	const glm::vec3 Quad::s_quadPositions[4] = { {-0.5f,-0.5f,0.f},{0.5f,-0.5f,0.f},{0.5f,0.5f,0.f},{-0.5f,0.5f,0.f} };
	const std::array<glm::vec2, 4> Quad::s_textureCoordinates = { glm::vec2(0.f,0.f), glm::vec2(1.f,0.f), glm::vec2(1.f,1.f), glm::vec2(0.f,1.f) };

	Quad::Quad(glm::vec3 position, glm::vec2 size, glm::vec4 color, float angleRadians, Ref<Texture2D> texture)
		: m_position(position)
		, m_size(size)
		, m_color(color)
		, m_angleRadians(angleRadians)
		, m_texture(texture)
		, m_textureCoordinates(Quad::s_textureCoordinates)
	{
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();
	}

	Quad::Quad(glm::vec3 position, glm::vec2 size, glm::vec4 color, float angleRadians, Ref<SubTexture2D> subtexture)
		: m_position(position)
		, m_size(size)
		, m_color(color)
		, m_angleRadians(angleRadians)
		, m_texture(subtexture->getTexture())
		, m_textureCoordinates(subtexture->getTextureCoordinates())
	{
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();
	}

	const glm::mat4 Quad::computeModelMatrix() const {
		LAZYENGINE_PROFILE_RENDERER_FUNCTION();

		glm::mat4 model = glm::mat4(1.f);
		model = glm::translate(model, glm::vec3(m_position.x, m_position.y, 0.f));
		if (fabs(m_angleRadians) > Constants::EPSILON) {
			model = glm::rotate(model, m_angleRadians, glm::vec3(0.f, 0.f, 1.f));
		}
		model = glm::scale(model, glm::vec3(m_size.x, m_size.y, 1.f));

		return model;
	}

}